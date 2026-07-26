#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol NeutronResidencySet
- (void)addAllocation:(id)allocation;
- (void)commit;
- (void)requestResidency;
- (void)endResidency;
- (void)removeAllAllocations;
@end
@interface NSObject (NeutronResidencyDevice)
- (id<NeutronResidencySet>)newResidencySetWithDescriptor:(id)descriptor error:(NSError**)error;
- (void)addResidencySet:(id<NeutronResidencySet>)set;
- (void)removeResidencySet:(id<NeutronResidencySet>)set;
@end

#include "neutron/engine.hpp"
#include "neutron/native/kv_cache.hpp"
#include "neutron/native/metal_ffn.hpp"
#include "neutron/native/model.hpp"
#include "neutron/native/multimodal.hpp"
#include "neutron/native/tokenizer.hpp"
#include "neutron/speculative.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <cstring>
#include <mutex>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>

namespace neutron {
namespace {
using Clock=std::chrono::steady_clock;
struct MatArgs{uint32_t cols,rows,batch;uint32_t pad;uint64_t weight_offset;};
struct PairMatArgs{uint32_t cols,rows,batch,pad;uint64_t weight0_offset,weight1_offset;};
struct QKVArgs{uint32_t cols,q_rows,kv_rows,batch;uint64_t q_offset,k_offset,v_offset;};
struct EmbArgs{uint32_t width,batch;uint64_t weight_offset;float scale;uint32_t pad;};
struct NormArgs{uint32_t width,rows;float eps;};
struct HeadNormArgs{uint32_t dim,heads,batch;float eps;};
struct NormRopeArgs{uint32_t dim,heads,batch,pos0;float eps,base;uint32_t use_factors,capacity;};
struct RopeArgs{uint32_t dim,heads,batch,pos0;float base;uint32_t use_factors;};
struct KVArgs{uint32_t dim,kv_heads,batch,pos0,capacity;};
struct AttnArgs{uint32_t dim,heads,kv_heads,batch,pos0,capacity,window;};
struct FlashArgs{
    uint32_t dim,heads,kv_heads,batch,pos0,capacity,window,span,causal;
    uint32_t sparse_sink_blocks,sparse_recent_block,sparse_stride,sparse_selected_blocks;
};
constexpr uint32_t kFusedLongDecodeSpan=2048;
constexpr uint32_t kMaxThreadgroupAttentionSpan=8000;
// The direct FP16 attention kernels are tiled over the query dimension and do
// not have a 768-token correctness limit.  Keeping the old cutoff forced a
// prompt just above the default chunk size through the score-buffer fallback,
// or through an almost-empty second model pass.  2048 is the largest prefill
// batch exposed by the runtime and remains bounded by the existing workspaces.
constexpr uint32_t kMaxDirectPrefillBatch=2048;

std::string default_kv_cache_directory(){
    const char*home=std::getenv("HOME");
    return home&&*home?std::string(home)+"/Library/Caches/neutron-star/kv":"";
}

uint64_t kv_cache_fingerprint(const native::GGUF&gguf,const Config&c){
    uint64_t hash=native::metal_ffn_fingerprint(gguf);
    auto mix=[&](uint64_t value){
        hash^=value+0x9e3779b97f4a7c15ULL+(hash<<6)+(hash>>2);
    };
    if(c.sparse_context){
        mix(0x676c6f62616c0001ULL);
        mix(c.sparse_context_threshold);mix(c.sparse_context_sink);
        mix(c.sparse_context_window);mix(c.sparse_context_stride);
    }
    if(c.sparse_swa){
        mix(0x7377610000000001ULL);
        mix(c.sparse_swa_threshold);mix(c.sparse_swa_sink);
        mix(c.sparse_swa_recent);mix(c.sparse_swa_stride);
    }
    if(!c.exact_ffn)mix(0x66666e3136000001ULL);
    return hash;
}

class NativeMetalEngine final:public Engine{
public:
    explicit NativeMetalEngine(const Config&c):cfg(c),gguf(c.model_path,c.metal_ffn_path.empty()),model(gguf),tok(gguf){
        if(c.context>model.context)throw std::runtime_error("requested context exceeds the Gemma model limit");
        NSArray<id<MTLDevice>>*ds=MTLCopyAllDevices();dev=ds.firstObject;if(!dev)throw std::runtime_error("Metal unavailable");
        NSError*err=nil;NSURL*url=[NSURL fileURLWithPath:@NEUTRON_METALLIB_PATH];lib=[dev newLibraryWithURL:url error:&err];if(!lib)throw std::runtime_error([[err localizedDescription]UTF8String]);queue=[dev newCommandQueue];
        if(!queue)throw std::runtime_error("Metal command queue creation failed");
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if(@available(macOS 26.0,*))metal4_submission=[dev supportsFamily:MTLGPUFamilyMetal4]&&std::getenv("NEUTRON_LEGACY_PREFILL_SUBMISSION")==nullptr;
#endif
        if(!c.metal_ffn_path.empty()){metal_ffn=std::make_unique<native::MetalFfnFile>(c.metal_ffn_path,gguf,model);weights=[dev newBufferWithBytesNoCopy:(void*)metal_ffn->mapped_data() length:metal_ffn->file_size() options:MTLResourceStorageModeShared deallocator:nil];}else weights=[dev newBufferWithBytesNoCopy:(void*)gguf.mapped_data() length:gguf.file_size() options:MTLResourceStorageModeShared deallocator:nil];if(!weights)throw std::runtime_error("Metal cannot map model weights");ones=[dev newBufferWithLength:512*4 options:MTLResourceStorageModeShared];std::fill_n((float*)ones.contents,512,1.0f);
        legacy_attention=std::getenv("NEUTRON_LEGACY_ATTENTION")!=nullptr;online_attention=std::getenv("NEUTRON_ONLINE_ATTENTION")!=nullptr;legacy_post_proj=std::getenv("NEUTRON_LEGACY_POST_PROJ")!=nullptr;full_causal_attention=std::getenv("NEUTRON_FULL_CAUSAL_ATTENTION")!=nullptr;reload_attention_q=std::getenv("NEUTRON_RELOAD_ATTENTION_Q")!=nullptr;qt8_attention=std::getenv("NEUTRON_QT8_ATTENTION")!=nullptr;swa_qt32_attention=std::getenv("NEUTRON_SWA_QT16_ATTENTION")==nullptr;swa_flash4_64=std::getenv("NEUTRON_SWA_K32_ATTENTION")==nullptr;swa_llama_q8=std::getenv("NEUTRON_SWA_LLAMA_Q8")!=nullptr;legacy_swa_f32_attention=std::getenv("NEUTRON_LEGACY_SWA_F32_ATTENTION")!=nullptr;llama_global_attention=std::getenv("NEUTRON_LEGACY_GLOBAL_PV_ATTENTION")==nullptr;compact_qt8=std::getenv("NEUTRON_COMPACT_QT8")!=nullptr;k32_matmul=std::getenv("NEUTRON_K32_MATMUL")!=nullptr;legacy_global_attention=std::getenv("NEUTRON_LEGACY_GLOBAL_ATTENTION")!=nullptr;k64_gate=std::getenv("NEUTRON_K64_GATE")!=nullptr;split_gate_up=std::getenv("NEUTRON_SPLIT_GATE_UP")!=nullptr;rm32_gate_up=metal_ffn&&std::getenv("NEUTRON_SERIAL_GATE_UP")==nullptr&&!split_gate_up&&std::getenv("NEUTRON_CONCURRENT_GATE_UP")==nullptr;legacy_decode_ffn=std::getenv("NEUTRON_LEGACY_DECODE_FFN")!=nullptr;concurrent_gate_up=std::getenv("NEUTRON_CONCURRENT_GATE_UP")!=nullptr||(metal_ffn&&std::getenv("NEUTRON_SERIAL_GATE_UP")==nullptr&&!rm32_gate_up&&!split_gate_up);persistent_packed_ffn=std::getenv("NEUTRON_PERSISTENT_PACKED_FFN")!=nullptr;legacy_prefill_ffn=std::getenv("NEUTRON_PACKED_FFN")==nullptr&&!persistent_packed_ffn&&!concurrent_gate_up;legacy_qkv=std::getenv("NEUTRON_FUSED_QKV")==nullptr;fast_verify=std::getenv("NEUTRON_EXACT_VERIFY")==nullptr;profile_graph=std::getenv("NEUTRON_PROFILE_GRAPH")!=nullptr;profile_sync=std::getenv("NEUTRON_PROFILE_SYNC")!=nullptr;profile_stages=std::getenv("NEUTRON_PROFILE_STAGES")!=nullptr;
        max_batch=cfg.batch;const size_t b=(max_batch+63)&~size_t(63);auto alloc=[&](size_t n){id<MTLBuffer>x=[dev newBufferWithLength:n*4 options:MTLResourceStorageModePrivate];if(!x)throw std::runtime_error("Metal workspace allocation failed");return x;};
        ids=[dev newBufferWithLength:std::max<size_t>(b,cfg.context)*4 options:MTLResourceStorageModeShared];x=alloc(b*3840);norm=alloc(b*3840);qbuf=alloc(b*8192);kbuf=alloc(b*2048);vbuf=alloc(b*2048);attn=alloc(b*8192);proj=alloc(b*3840);attn_out=alloc(b*3840);gate=alloc((concurrent_gate_up?b:4)*15360);up=alloc((concurrent_gate_up?b:4)*15360);mid=[dev newBufferWithLength:std::max<size_t>(b*15360*2,15360*4) options:MTLResourceStorageModePrivate];ffout=alloc(b*3840);halfbuf=[dev newBufferWithLength:b*8192*2 options:MTLResourceStorageModePrivate];logits=[dev newBufferWithLength:model.vocab*4 options:MTLResourceStorageModeShared];verify_logits=[dev newBufferWithLength:4*model.vocab*4 options:MTLResourceStorageModeShared];target_hidden=[dev newBufferWithLength:4*3840*4 options:MTLResourceStorageModeShared];if(!c.mmproj_path.empty())media_input=[dev newBufferWithLength:b*3840*4 options:MTLResourceStorageModeShared];if(!ids||!mid||!halfbuf||!logits||!verify_logits||!target_hidden||(!c.mmproj_path.empty()&&!media_input))throw std::runtime_error("Metal workspace allocation failed");
        if(metal_ffn&&!concurrent_gate_up&&!rm32_gate_up&&!split_gate_up){
            size_t scale_bytes=0;for(const auto&l:model.layer)for(const native::Tensor*t:{l.gate,l.up}){scale_bytes=(scale_bytes+255)&~size_t(255);q4_scale_offsets.emplace(t->offset,scale_bytes);scale_bytes+=size_t(t->shape[0])*t->shape[1]/32*4;}
            q4_scales=[dev newBufferWithLength:scale_bytes options:MTLResourceStorageModePrivate];if(!q4_scales)throw std::runtime_error("Metal Q4 scale sidecar allocation failed");id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];for(const auto&l:model.layer)for(const native::Tensor*t:{l.gate,l.up}){MatArgs a{uint32_t(t->shape[0]),uint32_t(t->shape[1]),0,0,metal_ffn->offset(*t)};[e setComputePipelineState:pipeline("gemma_q4k_expand_scales_metal")];buf(e,weights,0);buf(e,q4_scales,1,q4_scale_offsets.at(t->offset));[e setBytes:&a length:sizeof(a) atIndex:2];[e dispatchThreads:MTLSizeMake(size_t(a.cols)*a.rows/32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Metal Q4 scale preprocessing failed");
        }else if(!metal_ffn&&legacy_prefill_ffn){
            size_t scale_bytes=0;for(const auto&l:model.layer)for(const native::Tensor*t:{l.gate,l.up}){scale_bytes=(scale_bytes+255)&~size_t(255);q4_scale_offsets.emplace(t->offset,scale_bytes);scale_bytes+=size_t(t->shape[0])*t->shape[1]/32*4;}
            q4_scales=[dev newBufferWithLength:scale_bytes options:MTLResourceStorageModePrivate];if(!q4_scales)throw std::runtime_error("Metal Q4 scale sidecar allocation failed");id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];for(const auto&l:model.layer)for(const native::Tensor*t:{l.gate,l.up}){MatArgs a{uint32_t(t->shape[0]),uint32_t(t->shape[1]),0,0,gguf.data_offset()+t->offset};[e setComputePipelineState:pipeline("gemma_q4k_expand_scales")];buf(e,weights,0);buf(e,q4_scales,1,q4_scale_offsets.at(t->offset));[e setBytes:&a length:sizeof(a) atIndex:2];[e dispatchThreads:MTLSizeMake(size_t(a.cols)*a.rows/32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Metal Q4 scale preprocessing failed");
        }else if(!metal_ffn){
            const auto&t=*model.layer.front().gate;q4_packed_stride=(size_t(t.shape[0])*t.shape[1]/32*20+255)&~size_t(255);const size_t slots=persistent_packed_ffn?2*model.layer.size():2;q4_packed=[dev newBufferWithLength:slots*q4_packed_stride options:MTLResourceStorageModePrivate];if(!q4_packed)throw std::runtime_error("Metal Q4 packed FFN workspace allocation failed");
            if(persistent_packed_ffn){id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];for(size_t il=0;il<model.layer.size();++il){const auto&l=model.layer[il];size_t slot=2*il;for(const native::Tensor*w:{l.gate,l.up}){const size_t off=slot++*q4_packed_stride;q4_packed_offsets.emplace(w->offset,off);MatArgs a{uint32_t(w->shape[0]),uint32_t(w->shape[1]),0,0,gguf.data_offset()+w->offset};[e setComputePipelineState:pipeline("gemma_q4k_pack_rm64")];buf(e,weights,0);buf(e,q4_packed,1,off);[e setBytes:&a length:sizeof(a) atIndex:2];[e dispatchThreads:MTLSizeMake(size_t(a.cols)*a.rows/32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}}[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("persistent Q4 FFN packing failed");}
        }
        if(!persistent_packed_ffn&&!metal_ffn&&!std::getenv("NEUTRON_NO_RESIDENCY")){if(@available(macOS 15.0,*)){Class cls=NSClassFromString(@"MTLResidencySetDescriptor");SEL sel=NSSelectorFromString(@"newResidencySetWithDescriptor:error:");if(cls&&[dev respondsToSelector:sel]){id d=[[cls alloc]init];[d setValue:@"neutron-hot-weights" forKey:@"label"];[d setValue:@2 forKey:@"initialCapacity"];NSError*re=nil;residency=(id<NeutronResidencySet>)[(id)dev newResidencySetWithDescriptor:d error:&re];[d release];if(residency){[residency addAllocation:weights];if(q4_scales)[residency addAllocation:q4_scales];[residency commit];[residency requestResidency];[(id)queue addResidencySet:residency];}}}}
        for(const auto&l:model.layer){uint32_t cap=l.swa?std::min<uint32_t>(cfg.context,model.window+max_batch):cfg.context;size_t n=size_t(l.kv_heads)*cap*l.head_dim*2;id<MTLBuffer>kb=[dev newBufferWithLength:n options:MTLResourceStorageModePrivate],vb=[dev newBufferWithLength:n options:MTLResourceStorageModePrivate];if(!kb||!vb)throw std::runtime_error("KV allocation failed");kc.push_back(kb);vc.push_back(vb);}
        const std::string kv_cache_dir=c.kv_cache?(c.kv_cache_dir.empty()?default_kv_cache_directory():c.kv_cache_dir):"";if(!kv_cache_dir.empty())kv_disk=std::make_unique<native::KvCacheIndex>(kv_cache_dir,gguf.file_size(),kv_cache_fingerprint(gguf,c),cfg.context,max_batch,c.kv_cache_entries);
        description=metal_ffn?"Gemma 4 12B native Metal Q4_K_M + K-major FFN":"Gemma 4 12B native Metal Q4_K_M";
        if(!c.draft_model_path.empty()){
            draft_gguf=std::make_unique<native::GGUF>(c.draft_model_path);draft_model=std::make_unique<native::Gemma4AssistantModel>(*draft_gguf);for(auto&dl:draft_model->layer){dl.kv_source=-1;for(size_t i=model.layer.size();i-->0;)if(model.layer[i].kv_source<0&&model.layer[i].swa==dl.swa){dl.kv_source=int32_t(i);break;}if(dl.kv_source<0)throw std::runtime_error("assistant has no compatible target KV source");}
            if(draft_model->vocab!=model.vocab||gguf.meta<std::vector<std::string>>("tokenizer.ggml.tokens")!=draft_gguf->meta<std::vector<std::string>>("tokenizer.ggml.tokens"))throw std::runtime_error("target and assistant vocabularies differ");
            draft_weights=[dev newBufferWithBytesNoCopy:(void*)draft_gguf->mapped_data() length:draft_gguf->file_size() options:MTLResourceStorageModeShared deallocator:nil];
            auto da=[&](size_t n,MTLResourceOptions o=MTLResourceStorageModePrivate){id<MTLBuffer>v=[dev newBufferWithLength:n*4 options:o];if(!v)throw std::runtime_error("Metal assistant workspace allocation failed");return v;};
            draft_embedding=da(3840);draft_concat=da(7680);draft_x=da(1024);draft_norm=da(1024);draft_q=da(8192);draft_attn=da(8192);draft_proj=da(1024);draft_attn_out=da(1024);draft_mid=da(8192);draft_ffout=da(1024);draft_hnext=da(3840);draft_logits=da(model.vocab,MTLResourceStorageModeShared);
            description+=" + Gemma 4 Assistant MTP";
        }
        if(metal4_submission)description+=" + Metal 4 pipelined submission";
        if(kv_disk)description+=" + indexed disk KV";
        if(c.sparse_context)description+=" + opt-in sparse long context";
        if(c.sparse_swa)description+=" + approximate sparse SWA";
        if(!c.exact_ffn)description+=" + selective FP16 SWA FFN";
        if(!c.mmproj_path.empty()){
            multimodal=std::make_unique<native::MultimodalProcessor>(c.model_path,c.mmproj_path);
            description+=" + Gemma 4 vision/audio/video";
        }
    }
    ~NativeMetalEngine()override{finish_kv_save();multimodal.reset();drain_prefill();if(residency){if(@available(macOS 15.0,*)){[(id)queue removeResidencySet:residency];[residency endResidency];[residency removeAllAllocations];}[(id)residency release];}for(auto&[_,v]:pipes)[v release];for(auto v:kc)[v release];for(auto v:vc)[v release];for(id<MTLBuffer>v:{ids,x,norm,qbuf,kbuf,vbuf,attn,proj,attn_out,gate,up,mid,ffout,halfbuf,media_input,q4_scales,q4_packed,scorebuf,logits,verify_logits,target_hidden,ones,weights,draft_embedding,draft_concat,draft_x,draft_norm,draft_q,draft_attn,draft_proj,draft_attn_out,draft_mid,draft_ffout,draft_hnext,draft_logits,draft_weights})[v release];[queue release];[lib release];}

    GenerationResult generate(const std::string&prompt,const SamplingParams&sp,const TokenCallback&cb)override{
        std::lock_guard lock(mu);if(residency){if(@available(macOS 15.0,*))[residency requestResidency];}auto input=tok.encode(prompt);if(input.size()+sp.max_tokens>cfg.context)throw std::runtime_error("context exceeded");GenerationResult out;out.stats.prompt_tokens=input.size();size_t prefix=0;while(prefix<input.size()&&prefix<cached.size()&&input[prefix]==cached[prefix])++prefix;if(kv_disk&&prefix<input.size())try{if(auto hit=kv_disk->find_longest(input,prefix)){restore_kv_cache(*hit);prefix=hit->token_count();cached.assign(hit->tokens().begin(),hit->tokens().end());last_saved_tokens=cached.size();last_saved_hash=native::KvCacheIndex::token_hash(cached);}}catch(const std::exception&e){std::cerr<<"disk KV restore ignored: "<<e.what()<<'\n';}if(prefix==input.size()&&prefix)--prefix;cached.resize(prefix);out.stats.cached_tokens=prefix;
        const bool spec_active=draft_model&&sp.temperature<=0&&sp.repeat_penalty==1.0f;const bool need_softcap=sp.temperature>0||sp.repeat_penalty!=1.0f;const size_t prefill_end=input.size();auto ps=Clock::now();for(size_t off=prefix;off<prefill_end;){uint32_t n=std::min<size_t>(max_batch,prefill_end-off);eval(input.data()+off,n,off,off+n==prefill_end,need_softcap);cached.insert(cached.end(),input.begin()+off,input.begin()+off+n);off+=n;}out.stats.prefill_ms=ms(ps);
        std::mt19937 rng(sp.seed==0xffffffffu?std::random_device{}():sp.seed);auto gs=Clock::now();std::string pending;if(sp.max_tokens<=0){out.stats.finish_reason="length";save_kv_cache();return out;}
        if(spec_active){
            bool done=false;auto emit=[&](int32_t token){if(tok.is_eog(token)){out.stats.finish_reason="stop";return false;}std::string piece=tok.decode(token);pending+=piece;out.stats.generated_tokens++;size_t at=std::string::npos;for(const auto&s:sp.stop)if(!s.empty())at=std::min(at,pending.find(s));bool stop=at!=std::string::npos;if(stop){piece=pending.substr(0,at);pending.clear();}else piece=std::exchange(pending,{});out.text+=piece;if(cb&&!piece.empty()&&!cb(piece)){out.stats.finish_reason="cancelled";return false;}if(stop){out.stats.finish_reason="stop";return false;}return true;};
            int32_t last=sample(sp,rng);if(emit(last)){if(out.stats.generated_tokens>=uint64_t(sp.max_tokens))out.stats.finish_reason="length";while(out.stats.generated_tokens<uint64_t(sp.max_tokens)&&!done){const uint32_t remaining=sp.max_tokens-out.stats.generated_tokens;if(remaining==1){eval(&last,1,cached.size(),true,false);cached.push_back(last);int32_t t=sample(sp,rng);if(!emit(t))break;out.stats.finish_reason="length";break;}const uint32_t n=std::min<uint32_t>(cfg.speculative_tokens,remaining-1);std::vector<int32_t>draft(n);id<MTLBuffer>h=target_hidden;int32_t in=last;auto ds=Clock::now();for(uint32_t i=0;i<n;++i){draft[i]=draft_one(in,cached.size(),h);in=draft[i];h=draft_hnext;}double dms=ms(ds);auto vs=Clock::now();auto verified=verify_mtp_target(last,draft.data(),n,cached.size(),false);double vms=ms(vs);const size_t hidden_row=verified.accepted;std::memmove(target_hidden.contents,(float*)target_hidden.contents+hidden_row*3840,3840*4);cached.push_back(last);if(std::getenv("NEUTRON_SPEC_TRACE")){std::cerr<<"mtp pos="<<cached.size()-1<<" draft=[";for(size_t i=0;i<draft.size();++i)std::cerr<<(i?",":"")<<draft[i];std::cerr<<"] accepted="<<verified.accepted<<" next="<<verified.next_token<<" draft_ms="<<dms<<" verify_ms="<<vms<<'\n';}for(size_t i=0;i<verified.accepted;++i){int32_t t=draft[i];cached.push_back(t);last=t;if(!emit(t)){done=true;break;}}if(done||out.stats.generated_tokens>=uint64_t(sp.max_tokens))break;last=verified.next_token;if(!emit(last))break;if(out.stats.generated_tokens>=uint64_t(sp.max_tokens)){out.stats.finish_reason="length";break;}}}
            out.stats.generation_ms=ms(gs);save_kv_cache();return out;
        }
        for(int step=0;step<sp.max_tokens;++step){int32_t token=sample(sp,rng);if(std::getenv("NEUTRON_TOKEN_TRACE"))std::cerr<<"token pos="<<cached.size()<<" id="<<token<<'\n';if(tok.is_eog(token)){out.stats.finish_reason="stop";break;}std::string piece=tok.decode(token);pending+=piece;out.stats.generated_tokens++;
            bool stop=false;size_t at=std::string::npos;for(const auto&s:sp.stop)if(!s.empty())at=std::min(at,pending.find(s));if(at!=std::string::npos){piece=pending.substr(0,at);pending.clear();stop=true;}else{piece=std::exchange(pending,{});}out.text+=piece;if(cb&&!piece.empty()&&!cb(piece)){out.stats.finish_reason="cancelled";break;}if(stop){out.stats.finish_reason="stop";break;}if(step+1<sp.max_tokens){eval(&token,1,cached.size(),true,need_softcap);cached.push_back(token);}else out.stats.finish_reason="length";
        }out.stats.generation_ms=ms(gs);save_kv_cache();return out;
    }
    GenerationResult generate_multimodal(const std::string&prompt,const std::vector<MediaInput>&media,const SamplingParams&sp,const TokenCallback&cb)override{
        std::lock_guard lock(mu);
        if(!multimodal)throw std::runtime_error("multimodal input requires --mmproj");
        if(residency){if(@available(macOS 15.0,*))[residency requestResidency];}
        auto chunks=multimodal->encode(prompt,media);
        size_t total=0;for(const auto&chunk:chunks)total+=chunk.size();
        if(total+sp.max_tokens>cfg.context)throw std::runtime_error("multimodal prompt exceeds configured context");
        GenerationResult out;out.stats.prompt_tokens=total;cached.clear();cached.reserve(total+sp.max_tokens);
        const bool need_softcap=sp.temperature>0||sp.repeat_penalty!=1.0f;
        auto ps=Clock::now();size_t pos=0;
        for(size_t ci=0;ci<chunks.size();++ci){
            const auto&chunk=chunks[ci];size_t off=0,nchunk=chunk.size();
            if(chunk.non_causal&&nchunk>max_batch)throw std::runtime_error("a vision chunk exceeds the native non-causal batch");
            while(off<nchunk){
                uint32_t n=chunk.non_causal?uint32_t(nchunk):std::min<size_t>(max_batch,nchunk-off);
                bool last=ci+1==chunks.size()&&off+n==nchunk;
                if(chunk.tokens.empty()){
                    eval(nullptr,n,pos,last,need_softcap,0,chunk.embeddings.data()+off*3840,chunk.non_causal);
                    cached.insert(cached.end(),n,-1);
                }else{
                    eval(chunk.tokens.data()+off,n,pos,last,need_softcap);
                    cached.insert(cached.end(),chunk.tokens.begin()+off,chunk.tokens.begin()+off+n);
                }
                pos+=n;off+=n;
            }
        }
        out.stats.prefill_ms=ms(ps);
        std::mt19937 rng(sp.seed==0xffffffffu?std::random_device{}():sp.seed);
        auto gs=Clock::now();std::string pending;
        if(sp.max_tokens<=0){out.stats.finish_reason="length";return out;}
        for(int step=0;step<sp.max_tokens;++step){
            int32_t token=sample(sp,rng);
            if(tok.is_eog(token)){out.stats.finish_reason="stop";break;}
            std::string piece=tok.decode(token);pending+=piece;out.stats.generated_tokens++;
            bool stop=false;size_t at=std::string::npos;
            for(const auto&s:sp.stop)if(!s.empty())at=std::min(at,pending.find(s));
            if(at!=std::string::npos){piece=pending.substr(0,at);pending.clear();stop=true;}
            else piece=std::exchange(pending,{});
            out.text+=piece;
            if(cb&&!piece.empty()&&!cb(piece)){out.stats.finish_reason="cancelled";break;}
            if(stop){out.stats.finish_reason="stop";break;}
            if(step+1<sp.max_tokens){eval(&token,1,cached.size(),true,need_softcap);cached.push_back(token);}
            else out.stats.finish_reason="length";
        }
        out.stats.generation_ms=ms(gs);return out;
    }
    bool supports_media(const std::string&type)const override{return multimodal&&multimodal->supports(type);}
    std::string media_marker()const override{return multimodal?multimodal->marker():"<__media__>";}
    std::string model_description()const override{return description;}uint64_t model_size()const override{return gguf.file_size();}uint32_t context_size()const override{return cfg.context;}
private:
    static double ms(Clock::time_point s){return std::chrono::duration<double,std::milli>(Clock::now()-s).count();}
    void configure_sparse(FlashArgs& a)const{
        if(!cfg.sparse_context||!a.causal||a.span<cfg.sparse_context_threshold)return;
        constexpr uint32_t block=128;
        const uint32_t blocks=(a.span+block-1)/block;
        const uint32_t sink=std::min(blocks,(cfg.sparse_context_sink+block-1)/block);
        const uint32_t recent_token=a.pos0>cfg.sparse_context_window?a.pos0-cfg.sparse_context_window:0;
        const uint32_t recent=std::min(blocks,std::max(sink,recent_token/block));
        const uint32_t middle=recent-sink;
        const uint32_t sampled=(middle+cfg.sparse_context_stride-1)/cfg.sparse_context_stride;
        const uint32_t selected=sink+sampled+(blocks-recent);
        if(selected>=blocks)return;
        a.sparse_sink_blocks=sink;
        a.sparse_recent_block=recent;
        a.sparse_stride=cfg.sparse_context_stride;
        a.sparse_selected_blocks=selected;
    }
    void configure_swa_sparse(FlashArgs&a)const{
        if(!cfg.sparse_swa||!a.causal||a.span<cfg.sparse_swa_threshold)return;
        constexpr uint32_t block=64;
        a.sparse_sink_blocks=(cfg.sparse_swa_sink+block-1)/block;
        a.sparse_recent_block=(cfg.sparse_swa_recent+block-1)/block;
        a.sparse_stride=cfg.sparse_swa_stride;
        a.sparse_selected_blocks=1;
    }
    id<MTLCommandBuffer> command_buffer(){return metal4_submission?[queue commandBufferWithUnretainedReferences]:[queue commandBuffer];}
    void drain_prefill(){
        if(pending_prefill.empty())return;
        [pending_prefill.back() waitUntilCompleted];
        for(id<MTLCommandBuffer>cb:pending_prefill)[cb release];
        pending_prefill.clear();
    }
    void finish_command(id<MTLCommandBuffer>command,bool defer){
        [command commit];
        if(defer){[command retain];pending_prefill.push_back(command);return;}
        [command waitUntilCompleted];
        std::string error;
        for(id<MTLCommandBuffer>cb:pending_prefill){
            if(cb.status!=MTLCommandBufferStatusCompleted&&error.empty())error=cb.error?cb.error.localizedDescription.UTF8String:"deferred Metal command failed";
            [cb release];
        }
        pending_prefill.clear();
        if(command.status!=MTLCommandBufferStatusCompleted&&error.empty())error=command.error?command.error.localizedDescription.UTF8String:"Metal command failed";
        if(!error.empty())throw std::runtime_error("native Metal graph failed: "+error);
    }
    uint32_t kv_capacity(size_t layer_index)const{
        const auto&l=model.layer[layer_index];
        return l.swa?std::min<uint32_t>(cfg.context,model.window+max_batch):cfg.context;
    }
    void restore_kv_cache(const native::KvCacheMapping&mapping){
        size_t expected=0;for(const auto&l:model.layer)if(l.kv_source<0)++expected;
        if(mapping.layers().size()!=expected)throw std::runtime_error("disk KV layer count mismatch");
        id<MTLBuffer>source=[dev newBufferWithBytesNoCopy:(void*)mapping.data() length:mapping.size() options:MTLResourceStorageModeShared deallocator:nil];
        if(!source)throw std::runtime_error("Metal cannot map the disk KV snapshot");
        id<MTLCommandBuffer>command=[queue commandBuffer];id<MTLBlitCommandEncoder>blit=[command blitCommandEncoder];size_t entry=0;
        for(size_t il=0;il<model.layer.size();++il){const auto&l=model.layer[il];if(l.kv_source>=0)continue;const auto&e=mapping.layers()[entry++];const uint32_t cap=kv_capacity(il),slots=std::min<uint32_t>(mapping.token_count(),cap);const uint64_t bytes_per_head=uint64_t(slots)*l.head_dim*2,head_stride=uint64_t(cap)*l.head_dim*2;if(e.layer!=il||e.capacity!=cap||e.dim!=l.head_dim||e.heads!=l.kv_heads||e.slots!=slots||e.bytes_per_head!=bytes_per_head){[blit endEncoding];[source release];throw std::runtime_error("disk KV layer layout mismatch");}for(uint32_t h=0;h<l.kv_heads;++h){[blit copyFromBuffer:source sourceOffset:mapping.payload_offset()+e.key_offset+uint64_t(h)*bytes_per_head toBuffer:kc[il] destinationOffset:uint64_t(h)*head_stride size:bytes_per_head];[blit copyFromBuffer:source sourceOffset:mapping.payload_offset()+e.value_offset+uint64_t(h)*bytes_per_head toBuffer:vc[il] destinationOffset:uint64_t(h)*head_stride size:bytes_per_head];}}
        [blit endEncoding];[command commit];[command waitUntilCompleted];const bool ok=command.status==MTLCommandBufferStatusCompleted;std::string error=ok?"":(command.error?command.error.localizedDescription.UTF8String:"Metal disk KV restore failed");[source release];if(!ok)throw std::runtime_error(error);
    }
    void finish_kv_save(){
        if(kv_save.valid())kv_save.get();
    }
    void save_kv_cache(){
        if(!kv_disk||cached.size()<cfg.kv_cache_min_tokens)return;const uint64_t hash=native::KvCacheIndex::token_hash(cached);if(last_saved_tokens==cached.size()&&last_saved_hash==hash)return;if(kv_save.valid()){if(kv_save.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return;finish_kv_save();}drain_prefill();std::vector<native::KvCacheLayer>layers;uint64_t payload_bytes=0;
        for(size_t il=0;il<model.layer.size();++il){const auto&l=model.layer[il];if(l.kv_source>=0)continue;const uint32_t cap=kv_capacity(il),slots=std::min<uint32_t>(cached.size(),cap);const uint64_t bytes_per_head=uint64_t(slots)*l.head_dim*2,key_offset=payload_bytes;payload_bytes+=bytes_per_head*l.kv_heads;const uint64_t value_offset=payload_bytes;payload_bytes+=bytes_per_head*l.kv_heads;layers.push_back({uint32_t(il),cap,l.head_dim,l.kv_heads,slots,0,bytes_per_head,key_offset,value_offset});}
        if(!payload_bytes||payload_bytes>[dev maxBufferLength]){std::cerr<<"disk KV snapshot skipped: "<<payload_bytes<<" bytes exceed the Metal buffer limit\n";return;}id<MTLBuffer>stage=[dev newBufferWithLength:payload_bytes options:MTLResourceStorageModeShared];if(!stage){std::cerr<<"disk KV snapshot skipped: staging allocation failed\n";return;}id<MTLCommandBuffer>command=[queue commandBuffer];id<MTLBlitCommandEncoder>blit=[command blitCommandEncoder];for(const auto&e:layers){const uint64_t head_stride=uint64_t(e.capacity)*e.dim*2;for(uint32_t h=0;h<e.heads;++h){[blit copyFromBuffer:kc[e.layer] sourceOffset:uint64_t(h)*head_stride toBuffer:stage destinationOffset:e.key_offset+uint64_t(h)*e.bytes_per_head size:e.bytes_per_head];[blit copyFromBuffer:vc[e.layer] sourceOffset:uint64_t(h)*head_stride toBuffer:stage destinationOffset:e.value_offset+uint64_t(h)*e.bytes_per_head size:e.bytes_per_head];}}[blit endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted){std::cerr<<"disk KV snapshot ignored: Metal copy failed\n";[stage release];return;}last_saved_tokens=cached.size();last_saved_hash=hash;auto tokens=cached;auto*index=kv_disk.get();kv_save=std::async(std::launch::async,[index,tokens=std::move(tokens),layers=std::move(layers),stage,payload_bytes]{@autoreleasepool{try{index->store(tokens,layers,stage.contents,payload_bytes);}catch(const std::exception&e){std::cerr<<"disk KV snapshot ignored: "<<e.what()<<'\n';}[stage release];}});
    }
    id<MTLComputePipelineState> pipeline(const char*n){auto i=pipes.find(n);if(i!=pipes.end())return i->second;NSError*e=nil;id<MTLFunction>f=[lib newFunctionWithName:[NSString stringWithUTF8String:n]];id<MTLComputePipelineState>p=[dev newComputePipelineStateWithFunction:f error:&e];[f release];if(!p)throw std::runtime_error([[e localizedDescription]UTF8String]);pipes.emplace(n,p);return p;}
    static void buf(id<MTLComputeCommandEncoder>e,id<MTLBuffer>b,NSUInteger i,NSUInteger off=0){[e setBuffer:b offset:off atIndex:i];}
    uint64_t weight_offset(const native::Tensor&w)const{return metal_ffn?metal_ffn->offset(w):gguf.data_offset()+w.offset;}
    void dispatch1(id<MTLComputeCommandEncoder>e,const char*n,size_t count,id<MTLBuffer>a,id<MTLBuffer>b,id<MTLBuffer>c=nil){[e setComputePipelineState:pipeline(n)];buf(e,a,0);buf(e,b,1);if(c)buf(e,c,2);[e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(256,count),1,1)];}
    void matmul(id<MTLComputeCommandEncoder>&e,const native::Tensor&w,id<MTLBuffer>in,id<MTLBuffer>out,uint32_t batch,bool half_input=false){
        const bool q4=w.type==native::TensorType::Q4_K,reordered=metal_ffn&&metal_ffn->reordered(w);MatArgs a{uint32_t(w.shape[0]),uint32_t(w.shape[1]),batch,0,weight_offset(w)};
        if(reordered&&half_input&&batch>4){const char*n=q4?"gemma_q4k_mma_metal_f16":"gemma_q6k_mma_metal_f16";[e setComputePipelineState:pipeline(n)];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:(64*64+32*64)*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.rows+63)/64,(batch+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];return;}
        if(batch==1){const char*n=q4?(reordered?"gemma_q4k_mv_fast_metal":"gemma_q4k_mv_fast"):(reordered?"gemma_q6k_mv_fast_metal":"gemma_q6k_mv_fast");const uint32_t rows_per_group=q4?16:4,threads=q4?128:64;[e setComputePipelineState:pipeline(n)];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((a.rows+rows_per_group-1)/rows_per_group,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];return;}
        if(batch<=4){const char*n;if(half_input)n=q4?(reordered?"gemma_q4k_mm_fast_f16_metal":"gemma_q4k_mm_fast_f16"):(reordered?"gemma_q6k_mm_fast_f16_metal":"gemma_q6k_mm_fast_f16");else n=q4?(reordered?"gemma_q4k_mm_fast_metal":"gemma_q4k_mm_fast"):(reordered?"gemma_q6k_mm_fast_metal":"gemma_q6k_mm_fast");[e setComputePipelineState:pipeline(n)];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((a.rows+(q4?15:3))/(q4?16:4),(batch+(q4?3:1))/(q4?4:2),1) threadsPerThreadgroup:MTLSizeMake(q4?128:64,1,1)];return;}
        const bool k64=half_input&&!k32_matmul;const char*n=q4?(k64?"gemma_q4k_mma_swizzled_k64_f16":(half_input?"gemma_q4k_mma_swizzled_f16":"gemma_q4k_mma_swizzled")):(k64?"gemma_q6k_mma_swizzled_k64_f16":(half_input?"gemma_q6k_mma_swizzled_f16":"gemma_q6k_mma_swizzled"));[e setComputePipelineState:pipeline(n)];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:(k64?(64*64+32*64):(64*32+32*32))*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.rows+63)/64,(batch+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    void draft_matmul(id<MTLComputeCommandEncoder>e,const native::Tensor&w,id<MTLBuffer>in,id<MTLBuffer>out){const bool q4=w.type==native::TensorType::Q4_K;MatArgs a{uint32_t(w.shape[0]),uint32_t(w.shape[1]),1,0,draft_gguf->data_offset()+w.offset};[e setComputePipelineState:pipeline(q4?"gemma_q4k_mv_fast":"gemma_q6k_mv_fast")];buf(e,draft_weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((a.rows+(q4?15:3))/(q4?16:4),1,1) threadsPerThreadgroup:MTLSizeMake(q4?128:64,1,1)];}
    void draft_rms(id<MTLComputeCommandEncoder>e,id<MTLBuffer>in,id<MTLBuffer>out,const native::Tensor&w){[e setComputePipelineState:pipeline("gemma_rmsnorm")];buf(e,in,0);buf(e,draft_weights,1,draft_gguf->data_offset()+w.offset);buf(e,out,2);NormArgs a{1024,1,draft_model->epsilon};[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:32 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    void draft_rms_add(id<MTLComputeCommandEncoder>e,id<MTLBuffer>in,id<MTLBuffer>residual,id<MTLBuffer>out,const native::Tensor&w,const native::Tensor*scale=nullptr){[e setComputePipelineState:pipeline(scale?"gemma_rmsnorm_add_scale":"gemma_rmsnorm_add")];buf(e,in,0);buf(e,draft_weights,1,draft_gguf->data_offset()+w.offset);buf(e,residual,2);buf(e,out,3);NormArgs a{1024,1,draft_model->epsilon};[e setBytes:&a length:sizeof(a) atIndex:4];if(scale)buf(e,draft_weights,5,draft_gguf->data_offset()+scale->offset);[e setThreadgroupMemoryLength:32 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    void draft_gate_up(id<MTLComputeCommandEncoder>e,const native::Tensor&g,const native::Tensor&u,id<MTLBuffer>in,id<MTLBuffer>out){PairMatArgs a{uint32_t(g.shape[0]),uint32_t(g.shape[1]),1,0,draft_gguf->data_offset()+g.offset,draft_gguf->data_offset()+u.offset};[e setComputePipelineState:pipeline("gemma_q4k_gate_up_geglu_mv2")];buf(e,draft_weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((a.rows+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];}
    void qkv(id<MTLComputeCommandEncoder>e,const native::Gemma4Layer&l,id<MTLBuffer>in,uint32_t batch){QKVArgs a{uint32_t(l.wq->shape[0]),uint32_t(l.wq->shape[1]),uint32_t(l.wk->shape[1]),batch,weight_offset(*l.wq),weight_offset(*l.wk),weight_offset(l.wv?*l.wv:*l.wk)};const char*n=!l.wv?"gemma_qkv_tied_v_f16":(l.wv->type==native::TensorType::Q6_K?"gemma_qkv_q6v_f16":"gemma_qkv_q4_f16");[e setComputePipelineState:pipeline(n)];buf(e,weights,0);buf(e,in,1);buf(e,qbuf,2);buf(e,kbuf,3);buf(e,vbuf,4);[e setBytes:&a length:sizeof(a) atIndex:5];[e setThreadgroupMemoryLength:(3*64*32+16*32)*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.q_rows+63)/64,(batch+15)/16,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];}
    void gate_up_geglu(id<MTLComputeCommandEncoder>e,const native::Tensor&gate_w,const native::Tensor&up_w,id<MTLBuffer>in,id<MTLBuffer>out,uint32_t batch){
        if(batch<=4){PairMatArgs a{uint32_t(gate_w.shape[0]),uint32_t(gate_w.shape[1]),batch,0,weight_offset(gate_w),weight_offset(up_w)};[e setComputePipelineState:pipeline(metal_ffn?"gemma_q4k_gate_up_geglu_mm_f16_metal":"gemma_q4k_gate_up_geglu_mm_f16")];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((a.rows+7)/8,(batch+3)/4,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];return;}
        if(metal_ffn){PairMatArgs a{uint32_t(gate_w.shape[0]),uint32_t(gate_w.shape[1]),batch,0,metal_ffn->offset(gate_w),metal_ffn->offset(up_w)};[e setComputePipelineState:pipeline("gemma_q4k_gate_up_geglu_metal_f16")];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];buf(e,q4_scales,4,q4_scale_offsets.at(gate_w.offset));buf(e,q4_scales,5,q4_scale_offsets.at(up_w.offset));[e setThreadgroupMemoryLength:10240*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.rows+63)/64,(batch+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];return;}
        if(legacy_prefill_ffn){PairMatArgs a{uint32_t(gate_w.shape[0]),uint32_t(gate_w.shape[1]),batch,0,gguf.data_offset()+gate_w.offset,gguf.data_offset()+up_w.offset};const char*n=rm128_gate?"gemma_q4k_gate_up_geglu_scaled_rm128_f16":(k64_gate?"gemma_q4k_gate_up_geglu_scaled_k64_f16":"gemma_q4k_gate_up_geglu_scaled_f16");[e setComputePipelineState:pipeline(n)];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];buf(e,q4_scales,4,q4_scale_offsets.at(gate_w.offset));buf(e,q4_scales,5,q4_scale_offsets.at(up_w.offset));[e setThreadgroupMemoryLength:(rm128_gate?8704:(k64_gate?10240:8192))*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.rows+(rm128_gate?127:63))/(rm128_gate?128:64),(batch+(rm128_gate?15:31))/(rm128_gate?16:32),1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];return;}
        size_t gate_off=0,up_off=q4_packed_stride;if(persistent_packed_ffn){gate_off=q4_packed_offsets.at(gate_w.offset);up_off=q4_packed_offsets.at(up_w.offset);}else for(const auto&[w,off]:{std::pair<const native::Tensor*,size_t>{&gate_w,gate_off},std::pair<const native::Tensor*,size_t>{&up_w,up_off}}){MatArgs pa{uint32_t(w->shape[0]),uint32_t(w->shape[1]),0,0,gguf.data_offset()+w->offset};[e setComputePipelineState:pipeline("gemma_q4k_pack_rm64")];buf(e,weights,0);buf(e,q4_packed,1,off);[e setBytes:&pa length:sizeof(pa) atIndex:2];[e dispatchThreads:MTLSizeMake(size_t(pa.cols)*pa.rows/32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}PairMatArgs a{uint32_t(gate_w.shape[0]),uint32_t(gate_w.shape[1]),batch,0,gate_off,up_off};[e setComputePipelineState:pipeline("gemma_q4k_gate_up_geglu_packed_f16")];buf(e,q4_packed,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:8192*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.rows+63)/64,(batch+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    void gate_up_geglu_split(id<MTLComputeCommandEncoder>e,const native::Tensor&gate_w,const native::Tensor&up_w,id<MTLBuffer>in,id<MTLBuffer>out,uint32_t batch){PairMatArgs a{uint32_t(gate_w.shape[0]),uint32_t(gate_w.shape[1]),batch,0,metal_ffn->offset(gate_w),metal_ffn->offset(up_w)};[e setComputePipelineState:pipeline("gemma_q4k_gate_up_geglu_metal_split_f16")];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:10240*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.rows+63)/64,(batch+31)/32,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    void gate_up_geglu_rm32(id<MTLComputeCommandEncoder>e,const native::Tensor&gate_w,const native::Tensor&up_w,id<MTLBuffer>in,id<MTLBuffer>out,uint32_t batch,bool fast_swa){PairMatArgs a{uint32_t(gate_w.shape[0]),uint32_t(gate_w.shape[1]),batch,0,metal_ffn->offset(gate_w),metal_ffn->offset(up_w)};const char*n=fast_swa?"gemma_q4k_gate_up_geglu_metal_rm32_f16acc_f16":"gemma_q4k_gate_up_geglu_metal_rm32_f16";[e setComputePipelineState:pipeline(n)];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:6144*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.rows+31)/32,(batch+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];}
    void ffn_down(id<MTLComputeCommandEncoder>&e,const native::Tensor&w,id<MTLBuffer>in,id<MTLBuffer>out,uint32_t batch,bool half_input,bool fast_swa=false){if(metal_ffn&&half_input&&batch>4){const bool q4=w.type==native::TensorType::Q4_K;MatArgs a{uint32_t(w.shape[0]),uint32_t(w.shape[1]),batch,0,metal_ffn->offset(w)};const char*n=fast_swa?(q4?"gemma_q4k_mma_metal_f16acc_f16":"gemma_q6k_mma_metal_f16acc_f16"):(q4?"gemma_q4k_mma_metal_f16":"gemma_q6k_mma_metal_f16");[e setComputePipelineState:pipeline(n)];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:(64*64+32*64)*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((a.rows+63)/64,(batch+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];return;}matmul(e,w,in,out,batch,half_input);}
    void gate_up_geglu_decode(id<MTLComputeCommandEncoder>e,const native::Tensor&gate_w,const native::Tensor&up_w,id<MTLBuffer>in,id<MTLBuffer>out){PairMatArgs a{uint32_t(gate_w.shape[0]),uint32_t(gate_w.shape[1]),1,0,weight_offset(gate_w),weight_offset(up_w)};[e setComputePipelineState:pipeline(metal_ffn?"gemma_q4k_gate_up_geglu_mv2_metal":"gemma_q4k_gate_up_geglu_mv2")];buf(e,weights,0);buf(e,in,1);buf(e,out,2);[e setBytes:&a length:sizeof(a) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((a.rows+7)/8,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];}
    void rms(id<MTLComputeCommandEncoder>e,id<MTLBuffer>in,id<MTLBuffer>out,const native::Tensor&w,uint32_t width,uint32_t rows,NSUInteger inoff=0){[e setComputePipelineState:pipeline("gemma_rmsnorm")];buf(e,in,0,inoff);buf(e,weights,1,weight_offset(w));buf(e,out,2);NormArgs a{width,rows,model.epsilon};[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:32 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    void rms_half(id<MTLComputeCommandEncoder>e,id<MTLBuffer>in,const native::Tensor&w,uint32_t width,uint32_t rows){[e setComputePipelineState:pipeline("gemma_rmsnorm_f16")];buf(e,in,0);buf(e,weights,1,weight_offset(w));buf(e,halfbuf,2);NormArgs a{width,rows,model.epsilon};[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:32 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    void to_half(id<MTLComputeCommandEncoder>e,id<MTLBuffer>in,size_t count){[e setComputePipelineState:pipeline("gemma_f32_to_f16")];buf(e,in,0);buf(e,halfbuf,1);[e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    void ensure_scorebuf(size_t bytes){if(scorebuf&&scorebuf_capacity>=bytes)return;id<MTLBuffer>next=[dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];if(!next)throw std::runtime_error("Metal attention score allocation failed");if(scorebuf)[scorebuf release];scorebuf=next;scorebuf_capacity=bytes;}
    void rms_add(id<MTLComputeCommandEncoder>e,id<MTLBuffer>in,id<MTLBuffer>residual,id<MTLBuffer>out,const native::Tensor&w,uint32_t rows,const native::Tensor*scale=nullptr){[e setComputePipelineState:pipeline(scale?"gemma_rmsnorm_add_scale":"gemma_rmsnorm_add")];buf(e,in,0);buf(e,weights,1,weight_offset(w));buf(e,residual,2);buf(e,out,3);NormArgs a{3840,rows,model.epsilon};[e setBytes:&a length:sizeof(a) atIndex:4];if(scale)buf(e,weights,5,weight_offset(*scale));[e setThreadgroupMemoryLength:32 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    void rms_add_then_half(id<MTLComputeCommandEncoder>e,id<MTLBuffer>in,id<MTLBuffer>residual,id<MTLBuffer>out,const native::Tensor&w,const native::Tensor&next_w,uint32_t rows,const native::Tensor*scale=nullptr){[e setComputePipelineState:pipeline(scale?"gemma_rmsnorm_add_scale_then_f16":"gemma_rmsnorm_add_then_f16")];buf(e,in,0);buf(e,weights,1,weight_offset(w));buf(e,residual,2);buf(e,out,3);buf(e,weights,4,weight_offset(next_w));buf(e,halfbuf,5);NormArgs a{3840,rows,model.epsilon};[e setBytes:&a length:sizeof(a) atIndex:6];if(scale)buf(e,weights,7,weight_offset(*scale));[e setThreadgroupMemoryLength:32 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    void headnorm(id<MTLComputeCommandEncoder>e,id<MTLBuffer>b,const native::Tensor*w,uint32_t dim,uint32_t heads,uint32_t batch){[e setComputePipelineState:pipeline("gemma_head_rmsnorm")];buf(e,b,0);buf(e,w?weights:ones,1,w?weight_offset(*w):0);buf(e,b,2);HeadNormArgs a{dim,heads,batch,model.epsilon};[e setBytes:&a length:sizeof(a) atIndex:3];[e setThreadgroupMemoryLength:32 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(heads*batch,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
    int32_t draft_one(int32_t token,uint32_t pos,id<MTLBuffer>hidden){
        if(pos+1>kMaxThreadgroupAttentionSpan)ensure_scorebuf(size_t(16)*(pos+1)*sizeof(float));
        *(int32_t*)ids.contents=token;id<MTLCommandBuffer>command=[queue commandBuffer];id<MTLComputeCommandEncoder>e=[command computeCommandEncoder];
        [e setComputePipelineState:pipeline("gemma_q6k_embedding")];buf(e,weights,0);buf(e,ids,1);buf(e,draft_embedding,2);EmbArgs ea{3840,1,weight_offset(model.embedding),std::sqrt(3840.0f),0};[e setBytes:&ea length:sizeof(ea) atIndex:3];[e dispatchThreads:MTLSizeMake(3840,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e setComputePipelineState:pipeline("gemma_mtp_concat")];buf(e,draft_embedding,0);buf(e,hidden,1);buf(e,draft_concat,2);[e dispatchThreads:MTLSizeMake(7680,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];draft_matmul(e,draft_model->pre_projection,draft_concat,draft_x);
        for(const auto&l:draft_model->layer){draft_rms(e,draft_x,draft_norm,*l.attn_norm);draft_matmul(e,*l.wq,draft_norm,draft_q);uint32_t cap=l.swa?std::min<uint32_t>(cfg.context,model.window+max_batch):cfg.context;NormRopeArgs na{l.head_dim,16,1,pos,draft_model->epsilon,l.swa?10000.0f:1000000.0f,l.swa?0u:1u,cap};[e setComputePipelineState:pipeline("gemma_q_norm_rope")];buf(e,draft_q,0);buf(e,draft_weights,1,draft_gguf->data_offset()+l.q_norm->offset);buf(e,draft_weights,2,draft_gguf->data_offset()+draft_model->rope_freqs.offset);[e setBytes:&na length:sizeof(na) atIndex:3];[e setThreadgroupMemoryLength:8*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            if(pos+1>kMaxThreadgroupAttentionSpan&&!l.swa){FlashArgs fa{l.head_dim,16,model.layer[l.kv_source].kv_heads,1,pos,cap,0,pos+1,1,0,0,0,0};[e setComputePipelineState:pipeline("gemma_flash_qk")];buf(e,draft_q,0);buf(e,kc[l.kv_source],1);buf(e,scorebuf,2);[e setBytes:&fa length:sizeof(fa) atIndex:3];[e setThreadgroupMemoryLength:(8*32+32*32)*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((fa.span+31)/32,16,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e setComputePipelineState:pipeline("gemma_flash_softmax")];buf(e,scorebuf,0);[e setBytes:&fa length:sizeof(fa) atIndex:1];[e setThreadgroupMemoryLength:8*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e setComputePipelineState:pipeline("gemma_flash_pv")];buf(e,scorebuf,0);buf(e,vc[l.kv_source],1);buf(e,draft_attn,2);[e setBytes:&fa length:sizeof(fa) atIndex:3];[e setThreadgroupMemoryLength:(64*32+8*32)*sizeof(uint16_t) atIndex:0];[e dispatchThreadgroups:MTLSizeMake((l.head_dim+63)/64,16,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];}else{AttnArgs aa{l.head_dim,16,model.layer[l.kv_source].kv_heads,1,pos,cap,l.swa?model.window:0};[e setComputePipelineState:pipeline("gemma_mtp_attention")];buf(e,draft_q,0);buf(e,kc[l.kv_source],1);buf(e,vc[l.kv_source],2);buf(e,draft_attn,3);[e setBytes:&aa length:sizeof(aa) atIndex:4];[e setThreadgroupMemoryLength:(size_t(std::min<uint32_t>(pos,l.swa?model.window:cfg.context))+8)*4 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
            draft_matmul(e,*l.wo,draft_attn,draft_proj);draft_rms_add(e,draft_proj,draft_x,draft_attn_out,*l.attn_post_norm);draft_rms(e,draft_attn_out,draft_norm,*l.ffn_norm);draft_gate_up(e,*l.gate,*l.up,draft_norm,draft_mid);draft_matmul(e,*l.down,draft_mid,draft_ffout);draft_rms_add(e,draft_ffout,draft_attn_out,draft_x,*l.ffn_post_norm,l.out_scale);
        }
        draft_rms(e,draft_x,draft_norm,draft_model->output_norm);draft_matmul(e,draft_model->embedding,draft_norm,draft_logits);draft_matmul(e,draft_model->post_projection,draft_norm,draft_hnext);[e endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Gemma 4 Assistant graph failed: "+std::string(command.error.localizedDescription.UTF8String));float*l=(float*)draft_logits.contents;l[258882]=l[258883]=-INFINITY;return std::max_element(l,l+draft_model->vocab)-l;
    }
    void eval(const int32_t*tokens,uint32_t batch,uint32_t pos,bool output_logits=true,bool apply_softcap=true,uint32_t output_rows=0,const float*embeddings=nullptr,bool non_causal=false){if(output_rows>batch||output_rows>4)throw std::runtime_error("invalid target verification row count");if(non_causal&&(batch<=4||batch>768))throw std::runtime_error("native non-causal media attention requires 5..768 tokens");if(embeddings){if(!media_input)throw std::runtime_error("multimodal input buffer is unavailable");std::memcpy(media_input.contents,embeddings,size_t(batch)*3840*4);}else{if(!tokens)throw std::runtime_error("missing token or embedding input");std::memcpy((char*)ids.contents+size_t(pos)*4,tokens,batch*4);}const bool prefill=batch>4||(fast_verify&&output_rows>=2);if(prefill&&(legacy_attention||(!online_attention&&batch>kMaxDirectPrefillBatch)))ensure_scorebuf(size_t(batch)*16*(pos+batch)*(legacy_attention?4:2));if(!prefill&&pos+batch>kMaxThreadgroupAttentionSpan)ensure_scorebuf(size_t(batch)*16*(pos+batch)*sizeof(float));id<MTLCommandBuffer>command=command_buffer();current=command;id<MTLComputeCommandEncoder>e=[command computeCommandEncoder];auto profile_stage=[&](const char*stage,size_t layer,id<MTLBuffer>inspect,size_t count){if(!profile_stages)return;[e endEncoding];if(batch<=4){id<MTLBlitCommandEncoder>blit=[command blitCommandEncoder];[blit copyFromBuffer:inspect sourceOffset:0 toBuffer:verify_logits destinationOffset:0 size:count*4];[blit endEncoding];}[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("profiled Metal stage failed");float maxabs=0;if(batch<=4){const float*v=(const float*)verify_logits.contents;for(size_t i=0;i<count;++i){if(!std::isfinite(v[i]))throw std::runtime_error("non-finite target state after "+std::string(stage)+" layer "+std::to_string(layer)+" at element "+std::to_string(i));maxabs=std::max(maxabs,std::abs(v[i]));}}std::cerr<<"profile-stage pos="<<pos<<" batch="<<batch<<" layer="<<layer<<" stage="<<stage<<" maxabs="<<maxabs<<" gpu_ms="<<1000.0*(command.GPUEndTime-command.GPUStartTime)<<'\n';command=command_buffer();current=command;e=[command computeCommandEncoder];};
        if(embeddings){dispatch1(e,"gemma_copy",size_t(batch)*3840,media_input,x);}else{[e setComputePipelineState:pipeline("gemma_q6k_embedding")];buf(e,weights,0);buf(e,ids,1,size_t(pos)*4);buf(e,x,2);EmbArgs ea{3840,batch,weight_offset(model.embedding),std::sqrt(3840.0f),0};[e setBytes:&ea length:sizeof(ea) atIndex:3];[e dispatchThreads:MTLSizeMake(3840,batch,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
        for(size_t il=0;il<model.layer.size();++il){const auto&l=model.layer[il];const bool shared_kv=l.kv_source>=0;const size_t kvi=shared_kv?size_t(l.kv_source):il;const bool global_half_attention=prefill&&batch>4&&batch<=kMaxDirectPrefillBatch&&!l.swa&&l.head_dim==512&&!legacy_global_attention&&!legacy_attention&&!full_causal_attention&&!reload_attention_q;const bool swa_half_attention=prefill&&batch>4&&batch<=kMaxDirectPrefillBatch&&l.swa&&l.head_dim==256&&swa_qt32_attention&&!legacy_swa_f32_attention&&!legacy_attention&&!full_causal_attention&&!reload_attention_q;const bool direct_half_attention=global_half_attention||swa_half_attention;if(profile_graph){if(il){[e endEncoding];e=[command computeCommandEncoder];}e.label=[NSString stringWithFormat:@"layer-%02zu-%@",il,l.swa?@"swa":@"global"];}if(prefill){if(il==0)rms_half(e,x,*l.attn_norm,3840,batch);}else rms(e,x,norm,*l.attn_norm,3840,batch);id<MTLBuffer>act=prefill?halfbuf:norm;if(shared_kv)matmul(e,*l.wq,act,qbuf,batch,prefill);else if(prefill&&!legacy_qkv)qkv(e,l,act,batch);else{matmul(e,*l.wq,act,qbuf,batch,prefill);matmul(e,*l.wk,act,kbuf,batch,prefill);if(l.wv)matmul(e,*l.wv,act,vbuf,batch,prefill);else dispatch1(e,"gemma_copy",size_t(batch)*l.kv_heads*l.head_dim,kbuf,vbuf);}
            uint32_t cap=l.swa?std::min<uint32_t>(cfg.context,model.window+max_batch):cfg.context;if(shared_kv){NormRopeArgs na{l.head_dim,16,batch,pos,model.epsilon,l.swa?10000.0f:1000000.0f,l.swa?0u:1u,cap};[e setComputePipelineState:pipeline("gemma_q_norm_rope")];buf(e,qbuf,0);buf(e,l.q_norm?weights:ones,1,l.q_norm?gguf.data_offset()+l.q_norm->offset:0);buf(e,weights,2,gguf.data_offset()+model.rope_freqs.offset);[e setBytes:&na length:sizeof(na) atIndex:3];[e setThreadgroupMemoryLength:8*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(16*batch,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}else if(legacy_post_proj){headnorm(e,qbuf,l.q_norm,l.head_dim,16,batch);headnorm(e,kbuf,l.k_norm,l.head_dim,l.kv_heads,batch);headnorm(e,vbuf,nullptr,l.head_dim,l.kv_heads,batch);RopeArgs ra{l.head_dim,16,batch,pos,l.swa?10000.0f:1000000.0f,l.swa?0u:1u};[e setComputePipelineState:pipeline("gemma_rope_neox")];buf(e,qbuf,0);buf(e,weights,1,gguf.data_offset()+model.rope_freqs.offset);[e setBytes:&ra length:sizeof(ra) atIndex:2];[e dispatchThreads:MTLSizeMake(l.head_dim/2,16,batch) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256,l.head_dim/2),1,1)];ra.heads=l.kv_heads;[e setBytes:&ra length:sizeof(ra) atIndex:2];buf(e,kbuf,0);[e dispatchThreads:MTLSizeMake(l.head_dim/2,l.kv_heads,batch) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(256,l.head_dim/2),1,1)];KVArgs ka{l.head_dim,l.kv_heads,batch,pos,cap};[e setComputePipelineState:pipeline("gemma_store_kv")];buf(e,kbuf,0);buf(e,vbuf,1);buf(e,kc[kvi],2);buf(e,vc[kvi],3);[e setBytes:&ka length:sizeof(ka) atIndex:4];[e dispatchThreads:MTLSizeMake(l.head_dim,l.kv_heads,batch) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}else{NormRopeArgs na{l.head_dim,16,batch,pos,model.epsilon,l.swa?10000.0f:1000000.0f,l.swa?0u:1u,cap};[e setComputePipelineState:pipeline("gemma_q_norm_rope")];buf(e,qbuf,0);buf(e,l.q_norm?weights:ones,1,l.q_norm?gguf.data_offset()+l.q_norm->offset:0);buf(e,weights,2,gguf.data_offset()+model.rope_freqs.offset);[e setBytes:&na length:sizeof(na) atIndex:3];[e setThreadgroupMemoryLength:8*sizeof(float) atIndex:0];[e dispatchThreadgroups:MTLSizeMake(16*batch,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];na.heads=l.kv_heads;[e setComputePipelineState:pipeline("gemma_kv_norm_rope_store")];buf(e,kbuf,0);buf(e,vbuf,1);buf(e,l.k_norm?weights:ones,2,l.k_norm?gguf.data_offset()+l.k_norm->offset:0);buf(e,weights,3,gguf.data_offset()+model.rope_freqs.offset);buf(e,kc[kvi],4);buf(e,vc[kvi],5);[e setBytes:&na length:sizeof(na) atIndex:6];[e setThreadgroupMemoryLength:8*sizeof(float)*2 atIndex:0];[e dispatchThreadgroups:MTLSizeMake(l.kv_heads*batch,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}profile_stage("qkv-post",il,qbuf,size_t(batch)*l.wq->shape[1]);
            if (prefill) {
              uint32_t base = l.swa && pos >= model.window
                                  ? pos - model.window + 1
                                  : 0,
                       span = pos + batch - base;
              FlashArgs fa{l.head_dim,
                           16,
                           l.kv_heads,
                           batch,
                           pos,
                           cap,
                           l.swa ? model.window : 0,
                           span,
                           non_causal ? 0u : 1u,
                           0, 0, 0, 0};
              if(l.swa)configure_swa_sparse(fa);else configure_sparse(fa);
              if (direct_half_attention) {
                const bool llama_global = global_half_attention &&
                                          llama_global_attention &&
                                          fa.span > 1024 &&
                                          fa.span + 127 <= fa.capacity;
                const uint32_t qt = swa_half_attention ? (swa_llama_q8 ? 8 : ((swa_q16_compact||swa_register_q16) ? 16 : 32)) : (llama_global ? 8 : 16),
                               kt = swa_half_attention && !swa_flash4_64 ? 32 : 64;
                [e setComputePipelineState:
                        pipeline(swa_half_attention
                                     ? (swa_llama_q8
                                            ? "gemma_flash_llama_swa_q8_f16"
                                            : (swa_register_q16
                                            ? "gemma_flash_llama_swa_q16_register_f16"
                                            : (swa_q16_compact
                                            ? "gemma_flash_online_causal_swa_q16_compact_f16"
                                            : (swa_q32_reload
                                            ? "gemma_flash_online_causal_swa_q32_reload_f16"
                                            : (swa_flash4_64
                                            ? "gemma_flash_online_causal_swa_q32_k64_f16"
                                            : "gemma_flash_online_causal_swa_q32_f16")))))
                                     : (llama_global
                                            ? "gemma_flash_llama_causal_q8_f16"
                                            : "gemma_flash4_online_causal_q16_f16"))];
                buf(e, qbuf, 0);
                buf(e, kc[kvi], 1);
                buf(e, vc[kvi], 2);
                buf(e, halfbuf, 3);
                [e setBytes:&fa length:sizeof(fa) atIndex:4];
                [e setThreadgroupMemoryLength:(swa_half_attention&&swa_register_q16
                                                   ? (16*256+64*32)*sizeof(uint16_t)+(16*64+2*64+3*16)*sizeof(float)
                                                   : (swa_half_attention&&swa_q32_reload
                                                   ? (32*32+32*32)*sizeof(uint16_t)+(32*32+3*32)*sizeof(float)
                                                   : (swa_half_attention&&swa_llama_q8
                                                   ? (8*256+64*32)*sizeof(uint16_t)+(8*64+8*256+3*8)*sizeof(float)
                                                   : (llama_global
                                                   ? qt*l.head_dim*sizeof(float)
                                                     : (qt * l.head_dim + kt * 32) *
                                                         sizeof(uint16_t) +
                                                     (qt * kt + 3 * qt) * sizeof(float)))))
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(
                                            ((batch + qt - 1) / qt) * 16, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(swa_half_attention&&(swa_llama_q8||swa_register_q16)?128:256, 1, 1)];
              } else if (legacy_attention) {
                [e setComputePipelineState:pipeline("gemma_flash_qk")];
                buf(e, qbuf, 0);
                buf(e, kc[kvi], 1);
                buf(e, scorebuf, 2);
                [e setBytes:&fa length:sizeof(fa) atIndex:3];
                [e setThreadgroupMemoryLength:(8 * 32 + 32 * 32) *
                                              sizeof(uint16_t)
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((span + 31) / 32,
                                                    ((batch + 7) / 8) * 16, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e setComputePipelineState:pipeline("gemma_flash_softmax")];
                buf(e, scorebuf, 0);
                [e setBytes:&fa length:sizeof(fa) atIndex:1];
                [e setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(batch * 16, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e setComputePipelineState:pipeline("gemma_flash_pv")];
                buf(e, scorebuf, 0);
                buf(e, vc[kvi], 1);
                buf(e, attn, 2);
                [e setBytes:&fa length:sizeof(fa) atIndex:3];
                [e setThreadgroupMemoryLength:(64 * 32 + 8 * 32) *
                                              sizeof(uint16_t)
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((l.head_dim + 63) / 64,
                                                    ((batch + 7) / 8) * 16, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
              } else if (online_attention || batch <= 768) {
                bool cache_q = !full_causal_attention && !reload_attention_q,
                     q16 = cache_q && !qt8_attention,
                     q32 = swa_qt32_attention && l.swa && q16;
                uint32_t qt = q32 ? 32 : (q16 ? 16 : 8);
                [e setComputePipelineState:
                        pipeline(
                            full_causal_attention
                                ? "gemma_flash_online"
                                : (q32 ? "gemma_flash_online_causal_swa_q32"
                                       : (q16 ? "gemma_flash_online_causal_q16"
                                              : (cache_q ? "gemma_flash_online_"
                                                           "causal_cached"
                                                         : "gemma_flash_online_"
                                                           "causal"))))];
                buf(e, qbuf, 0);
                buf(e, kc[kvi], 1);
                buf(e, vc[kvi], 2);
                buf(e, attn, 3);
                [e setBytes:&fa length:sizeof(fa) atIndex:4];
                [e setThreadgroupMemoryLength:((cache_q ? qt * l.head_dim
                                                        : 8 * 32) +
                                               32 * 32) *
                                                  sizeof(uint16_t) +
                                              (qt * 32 + 3 * qt) * sizeof(float)
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(
                                            ((batch + qt - 1) / qt) * 16, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
              } else {
                bool cache_q = !full_causal_attention && !reload_attention_q,
                     q16 = cache_q && !compact_qt8;
                [e setComputePipelineState:
                        pipeline(
                            full_causal_attention
                                ? "gemma_flash_qk_softmax_f16"
                                : (q16 ? "gemma_flash_qk_softmax_f16_causal_q16"
                                       : (cache_q ? "gemma_flash_qk_softmax_"
                                                    "f16_causal_cached"
                                                  : "gemma_flash_qk_softmax_"
                                                    "f16_causal")))];
                buf(e, qbuf, 0);
                buf(e, kc[kvi], 1);
                buf(e, scorebuf, 2);
                [e setBytes:&fa length:sizeof(fa) atIndex:3];
                [e setThreadgroupMemoryLength:((cache_q ? (q16 ? 16 : 8) *
                                                              l.head_dim
                                                        : 8 * 32) +
                                               32 * 32) *
                                                  sizeof(uint16_t) +
                                              (cache_q ? (q16 ? 16 : 8) * 32 *
                                                             sizeof(float)
                                                       : 0)
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(((batch + (q16 ? 15 : 7)) /
                                                     (q16 ? 16 : 8)) *
                                                        16,
                                                    1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e setComputePipelineState:
                        pipeline(full_causal_attention
                                     ? "gemma_flash_pv_f16"
                                     : (q16 ? "gemma_flash_pv_f16_causal_q16"
                                            : "gemma_flash_pv_f16_causal"))];
                buf(e, scorebuf, 0);
                buf(e, vc[kvi], 1);
                buf(e, attn, 2);
                [e setBytes:&fa length:sizeof(fa) atIndex:3];
                [e setThreadgroupMemoryLength:(64 * 32 + (q16 ? 16 : 8) * 32) *
                                              sizeof(uint16_t)
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((l.head_dim + 63) / 64,
                                                    ((batch + (q16 ? 15 : 7)) /
                                                     (q16 ? 16 : 8)) *
                                                        16,
                                                    1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
              }
            } else if (pos + batch > kFusedLongDecodeSpan && !l.swa) {
              uint32_t span = pos + batch;
              FlashArgs fa{l.head_dim, 16,  l.kv_heads, batch,
                           pos,        cap, 0,          span, 1, 0, 0, 0, 0};
              configure_sparse(fa);
              if (batch == 1 && l.head_dim == 512 &&
                  span + 127 <= cap &&
                  !std::getenv("NEUTRON_LEGACY_LONG_DECODE")) {
                [e setComputePipelineState:
                       pipeline("gemma_flash_llama_causal_q8_f16")];
                buf(e, qbuf, 0);
                buf(e, kc[kvi], 1);
                buf(e, vc[kvi], 2);
                buf(e, halfbuf, 3);
                [e setBytes:&fa length:sizeof(fa) atIndex:4];
                [e setThreadgroupMemoryLength:8 * l.head_dim * sizeof(float)
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(16, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e setComputePipelineState:pipeline("gemma_f16_to_f32")];
                buf(e, halfbuf, 0);
                buf(e, attn, 1);
                [e dispatchThreads:MTLSizeMake(16 * l.head_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
              } else {
                ensure_scorebuf(size_t(batch) * 16 * span * sizeof(float));
                [e setComputePipelineState:pipeline("gemma_flash_qk")];
                buf(e, qbuf, 0);
                buf(e, kc[kvi], 1);
                buf(e, scorebuf, 2);
                [e setBytes:&fa length:sizeof(fa) atIndex:3];
                [e setThreadgroupMemoryLength:(8 * 32 + 32 * 32) *
                                              sizeof(uint16_t)
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((span + 31) / 32,
                                                    ((batch + 7) / 8) * 16, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [e setComputePipelineState:pipeline("gemma_flash_softmax")];
                buf(e, scorebuf, 0);
                [e setBytes:&fa length:sizeof(fa) atIndex:1];
                [e setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(batch * 16, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e setComputePipelineState:pipeline("gemma_flash_pv")];
                buf(e, scorebuf, 0);
                buf(e, vc[kvi], 1);
                buf(e, attn, 2);
                [e setBytes:&fa length:sizeof(fa) atIndex:3];
                [e setThreadgroupMemoryLength:(64 * 32 + 8 * 32) *
                                              sizeof(uint16_t)
                                      atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((l.head_dim + 63) / 64,
                                                    ((batch + 7) / 8) * 16, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
              }
            } else {
              AttnArgs aa{l.head_dim,
                          16,
                          l.kv_heads,
                          batch,
                          pos,
                          cap,
                          l.swa ? model.window : 0};
              [e setComputePipelineState:pipeline("gemma_attention")];
              buf(e, qbuf, 0);
              buf(e, kc[kvi], 1);
              buf(e, vc[kvi], 2);
              buf(e, attn, 3);
              [e setBytes:&aa length:sizeof(aa) atIndex:4];
              [e setThreadgroupMemoryLength:(size_t(std::min<uint32_t>(
                                                 pos + batch,
                                                 l.swa ? model.window
                                                       : cfg.context)) +
                                             8) *
                                            4
                                    atIndex:0];
              [e dispatchThreadgroups:MTLSizeMake(16 * batch, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            profile_stage("attention", il,
                          direct_half_attention ? halfbuf : attn,
                          size_t(batch) * l.wq->shape[1]);
            if(prefill&&!direct_half_attention)to_half(e,attn,size_t(batch)*l.wq->shape[1]);
            act=prefill?halfbuf:attn;matmul(e,*l.wo,act,proj,batch,prefill);
            if(prefill)rms_add_then_half(e,proj,x,attn_out,*l.attn_post_norm,*l.ffn_norm,batch);else{rms_add(e,proj,x,attn_out,*l.attn_post_norm,batch);rms(e,attn_out,norm,*l.ffn_norm,3840,batch);}profile_stage("attn-out",il,attn_out,size_t(batch)*3840);
            act=prefill?halfbuf:norm;
            const bool fast_swa_ffn=prefill&&l.swa&&!cfg.exact_ffn;
            if(prefill&&rm32_gate_up&&metal_ffn&&batch>4){gate_up_geglu_rm32(e,*l.gate,*l.up,act,mid,batch,fast_swa_ffn);}
            else if(prefill&&split_gate_up&&metal_ffn&&batch>4){gate_up_geglu_split(e,*l.gate,*l.up,act,mid,batch);}
            else if(prefill&&concurrent_gate_up&&batch>4){[e endEncoding];e=[command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];matmul(e,*l.gate,act,gate,batch,true);matmul(e,*l.up,act,up,batch,true);[e endEncoding];e=[command computeCommandEncoder];dispatch1(e,"gemma_geglu_f16",size_t(batch)*15360,gate,up,mid);}
            else if(prefill){gate_up_geglu(e,*l.gate,*l.up,act,mid,batch);}
            else if(batch>1||legacy_decode_ffn){matmul(e,*l.gate,act,gate,batch,false);matmul(e,*l.up,act,up,batch,false);dispatch1(e,"gemma_geglu",size_t(batch)*15360,gate,up,mid);}
            else gate_up_geglu_decode(e,*l.gate,*l.up,act,mid);
            profile_stage("gate-up",il,mid,size_t(batch)*15360);
            ffn_down(e,*l.down,mid,ffout,batch,prefill,fast_swa_ffn);profile_stage("down",il,ffout,size_t(batch)*3840);
            if(prefill&&il+1<model.layer.size())rms_add_then_half(e,ffout,attn_out,x,*l.ffn_post_norm,*model.layer[il+1].attn_norm,batch,l.out_scale);else rms_add(e,ffout,attn_out,x,*l.ffn_post_norm,batch,l.out_scale);profile_stage("down-post",il,x,size_t(batch)*3840);
            if(profile_sync){[e endEncoding];if(batch<=4){id<MTLBlitCommandEncoder>blit=[command blitCommandEncoder];[blit copyFromBuffer:x sourceOffset:0 toBuffer:target_hidden destinationOffset:0 size:size_t(batch)*3840*4];[blit endEncoding];}[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("profiled Metal layer failed");if(batch<=4){const float*h=(const float*)target_hidden.contents;for(size_t i=0;i<size_t(batch)*3840;++i)if(!std::isfinite(h[i]))throw std::runtime_error("non-finite target state after layer "+std::to_string(il)+" at element "+std::to_string(i));}std::cerr<<"profile pos="<<pos<<" batch="<<batch<<" layer="<<il<<" kind="<<(l.swa?"swa":"global")<<" gpu_ms="<<1000.0*(command.GPUEndTime-command.GPUStartTime)<<'\n';command=[queue commandBuffer];current=command;e=[command computeCommandEncoder];}}
        if(output_logits){const uint32_t rows=output_rows?output_rows:1;id<MTLBuffer>dst=output_rows?verify_logits:logits;rms(e,x,target_hidden,model.output_norm,3840,rows,size_t(batch-rows)*3840*4);matmul(e,model.embedding,target_hidden,dst,rows);if(apply_softcap){[e setComputePipelineState:pipeline("gemma_softcap")];buf(e,dst,0);float cap=model.logit_cap;[e setBytes:&cap length:4 atIndex:1];[e dispatchThreads:MTLSizeMake(size_t(model.vocab)*rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}}[e endEncoding];const bool stable_score_storage=!legacy_attention&&(online_attention||batch<=768);finish_command(command,metal4_submission&&prefill&&!output_logits&&!profile_sync&&!profile_stages&&stable_score_storage&&!embeddings);}
    MtpVerification verify_mtp_target(int32_t last,const int32_t*draft,uint32_t count,uint32_t pos,bool apply_softcap){
        if(count<1||count>3)throw std::invalid_argument("Gemma MTP draft length must be 1..3");
        auto argmax=[&](const float*l){int32_t best=-1;for(uint32_t i=0;i<model.vocab;++i)if(i!=258882&&i!=258883&&std::isfinite(l[i])&&(best<0||l[i]>l[best]))best=i;return best;};
        std::vector<int32_t>batch(count+1),target(count+1);batch[0]=last;std::copy_n(draft,count,batch.begin()+1);eval(batch.data(),count+1,pos,true,apply_softcap,count+1);float*rows=(float*)verify_logits.contents;
        const float*hidden=(const float*)target_hidden.contents;for(size_t i=0;i<size_t(count+1)*3840;++i)if(!std::isfinite(hidden[i]))throw std::runtime_error("target verification produced non-finite hidden state at element "+std::to_string(i));
        for(uint32_t i=0;i<=count;++i){target[i]=argmax(rows+size_t(i)*model.vocab);if(target[i]<0)throw std::runtime_error("target verification produced no finite logits");}
        return verify_mtp_greedy({draft,count},target);
    }
    int32_t sample(const SamplingParams&s,std::mt19937&rng){float*l=(float*)logits.contents;l[258882]=l[258883]=-INFINITY;if(s.repeat_penalty!=1.0f&&s.repeat_penalty>0){const size_t begin=cached.size()>64?cached.size()-64:0;for(size_t i=begin;i<cached.size();++i){const int32_t id=cached[i];if(id>=0&&uint32_t(id)<model.vocab)l[id]=l[id]<0?l[id]*s.repeat_penalty:l[id]/s.repeat_penalty;}}if(s.temperature<=0)return std::max_element(l,l+model.vocab)-l;size_t k=std::clamp(s.top_k,1,int(model.vocab));std::vector<int32_t>idx(model.vocab);std::iota(idx.begin(),idx.end(),0);std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){return l[a]>l[b];});idx.resize(k);float mx=l[idx[0]],sum=0;std::vector<float>p(k);for(size_t i=0;i<k;++i){p[i]=std::exp((l[idx[i]]-mx)/s.temperature);sum+=p[i];}for(float&v:p)v/=sum;const float floor=p[0]*std::max(0.0f,s.min_p);float cum=0;size_t keep=0;while(keep<k&&p[keep]>=floor){cum+=p[keep++];if(cum>=s.top_p)break;}keep=std::max<size_t>(1,keep);std::discrete_distribution<size_t>d(p.begin(),p.begin()+keep);return idx[d(rng)];}
    Config cfg;native::GGUF gguf;native::Gemma4Model model;native::Gemma4Tokenizer tok;std::unique_ptr<native::GGUF>draft_gguf;std::unique_ptr<native::Gemma4AssistantModel>draft_model;std::unique_ptr<native::MetalFfnFile>metal_ffn;std::unique_ptr<native::KvCacheIndex>kv_disk;std::future<void>kv_save;std::unique_ptr<native::MultimodalProcessor>multimodal;std::string description;uint32_t max_batch;bool metal4_submission=false,legacy_attention=false,online_attention=false,legacy_post_proj=false,full_causal_attention=false,reload_attention_q=false,qt8_attention=false,swa_qt32_attention=false,swa_flash4_64=false,swa_llama_q8=false,swa_q16_compact=std::getenv("NEUTRON_SWA_Q16_COMPACT")!=nullptr,swa_q32_reload=std::getenv("NEUTRON_SWA_Q32_RELOAD")!=nullptr,swa_register_q16=std::getenv("NEUTRON_SWA_REGISTER_Q16")!=nullptr,legacy_swa_f32_attention=false,llama_global_attention=false,compact_qt8=false,k32_matmul=false,legacy_global_attention=false,k64_gate=false,split_gate_up=false,rm32_gate_up=false,rm128_gate=std::getenv("NEUTRON_RM128_GATE")!=nullptr,legacy_decode_ffn=false,concurrent_gate_up=false,legacy_prefill_ffn=false,persistent_packed_ffn=false,legacy_qkv=false,fast_verify=false,profile_graph=false,profile_sync=false,profile_stages=false;std::vector<int32_t>cached;size_t last_saved_tokens=0;uint64_t last_saved_hash=0;std::vector<id<MTLCommandBuffer>>pending_prefill;std::mutex mu;
    id<MTLDevice>dev=nil;id<MTLLibrary>lib=nil;id<MTLCommandQueue>queue=nil;id<MTLCommandBuffer>current=nil;id<NeutronResidencySet>residency=nil;id<MTLBuffer>weights=nil,ones=nil,ids=nil,x=nil,norm=nil,qbuf=nil,kbuf=nil,vbuf=nil,attn=nil,proj=nil,attn_out=nil,gate=nil,up=nil,mid=nil,ffout=nil,halfbuf=nil,media_input=nil,q4_scales=nil,q4_packed=nil,scorebuf=nil,logits=nil,verify_logits=nil,target_hidden=nil,draft_weights=nil,draft_embedding=nil,draft_concat=nil,draft_x=nil,draft_norm=nil,draft_q=nil,draft_attn=nil,draft_proj=nil,draft_attn_out=nil,draft_mid=nil,draft_ffout=nil,draft_hnext=nil,draft_logits=nil;size_t q4_packed_stride=0,scorebuf_capacity=0;std::vector<id<MTLBuffer>>kc,vc;std::unordered_map<uint64_t,size_t>q4_scale_offsets,q4_packed_offsets;std::unordered_map<std::string,id<MTLComputePipelineState>>pipes;
};
}
std::unique_ptr<Engine>make_native_engine(const Config&c){return std::make_unique<NativeMetalEngine>(c);}
}
