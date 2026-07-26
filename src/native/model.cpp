#include "neutron/native/model.hpp"
#include <stdexcept>

namespace neutron::native {
namespace {
const Tensor* optional(const GGUF&g,const std::string&n){try{return &g.tensor(n);}catch(...){return nullptr;}}
void shape(const Tensor&t,uint64_t a,uint64_t b=0){if(t.shape.size()!=(b?2:1)||t.shape[0]!=a||(b&&t.shape[1]!=b))throw std::runtime_error("unexpected shape: "+t.name);}
void type(const Tensor&t,TensorType expected){if(t.type!=expected)throw std::runtime_error("unexpected tensor type: "+t.name);}
void q4_or_q6(const Tensor&t){if(t.type!=TensorType::Q4_K&&t.type!=TensorType::Q6_K)throw std::runtime_error("expected Q4_K or Q6_K tensor: "+t.name);}
}
Gemma4Model::Gemma4Model(const GGUF&g):gguf(g),embedding(g.tensor("token_embd.weight")),output_norm(g.tensor("output_norm.weight")),rope_freqs(g.tensor("rope_freqs.weight")){
    if(g.meta<std::string>("general.architecture")!="gemma4")throw std::runtime_error("native engine only accepts Gemma 4");
    layers=g.meta_u64("gemma4.block_count");embedding_dim=g.meta_u64("gemma4.embedding_length");ffn_dim=g.meta_u64("gemma4.feed_forward_length");heads=g.meta_u64("gemma4.attention.head_count");context=g.meta_u64("gemma4.context_length");window=g.meta_u64("gemma4.attention.sliding_window");vocab=embedding.shape.at(1);epsilon=g.meta_number("gemma4.attention.layer_norm_rms_epsilon");logit_cap=g.meta_number("gemma4.final_logit_softcapping");
    if(layers!=48||embedding_dim!=3840||ffn_dim!=15360||heads!=16)throw std::runtime_error("GGUF is not the supported Gemma 4 12B layout");shape(embedding,embedding_dim,vocab);shape(output_norm,embedding_dim);shape(rope_freqs,256);type(embedding,TensorType::Q6_K);
    const auto&kv=g.meta<std::vector<int32_t>>("gemma4.attention.head_count_kv");const auto&swa=g.meta<std::vector<uint8_t>>("gemma4.attention.sliding_window_pattern");if(kv.size()!=layers||swa.size()!=layers)throw std::runtime_error("invalid per-layer metadata");const uint32_t shared=g.has("gemma4.attention.shared_kv_layers")?g.meta_u64("gemma4.attention.shared_kv_layers"):0;if(shared>=layers)throw std::runtime_error("invalid shared KV layer count");const uint32_t first_shared=layers-shared;
    layer.reserve(layers);for(uint32_t i=0;i<layers;++i){const std::string p="blk."+std::to_string(i)+".";Gemma4Layer l{};l.swa=swa[i];l.kv_heads=kv[i];l.head_dim=l.swa?256:512;if(i>=first_shared){for(uint32_t j=first_shared;j-->0;)if(bool(swa[j])==l.swa){l.kv_source=int32_t(j);break;}if(l.kv_source<0)throw std::runtime_error("shared KV layer has no compatible source");}
        l.attn_norm=&g.tensor(p+"attn_norm.weight");l.wq=&g.tensor(p+"attn_q.weight");l.wk=&g.tensor(p+"attn_k.weight");l.wv=optional(g,p+"attn_v.weight");l.wo=&g.tensor(p+"attn_output.weight");l.q_norm=&g.tensor(p+"attn_q_norm.weight");l.k_norm=&g.tensor(p+"attn_k_norm.weight");l.attn_post_norm=&g.tensor(p+"post_attention_norm.weight");l.ffn_norm=&g.tensor(p+"ffn_norm.weight");l.gate=&g.tensor(p+"ffn_gate.weight");l.up=&g.tensor(p+"ffn_up.weight");l.down=&g.tensor(p+"ffn_down.weight");l.ffn_post_norm=&g.tensor(p+"post_ffw_norm.weight");l.out_scale=&g.tensor(p+"layer_output_scale.weight");
        shape(*l.wq,embedding_dim,uint64_t(heads)*l.head_dim);shape(*l.wk,embedding_dim,uint64_t(l.kv_heads)*l.head_dim);if(l.wv)shape(*l.wv,embedding_dim,uint64_t(l.kv_heads)*l.head_dim);shape(*l.wo,uint64_t(heads)*l.head_dim,embedding_dim);shape(*l.gate,embedding_dim,ffn_dim);shape(*l.up,embedding_dim,ffn_dim);shape(*l.down,ffn_dim,embedding_dim);
        type(*l.wq,TensorType::Q4_K);type(*l.wk,TensorType::Q4_K);type(*l.wo,TensorType::Q4_K);type(*l.gate,TensorType::Q4_K);type(*l.up,TensorType::Q4_K);if(l.wv)q4_or_q6(*l.wv);q4_or_q6(*l.down);layer.push_back(l);}
}

Gemma4AssistantModel::Gemma4AssistantModel(const GGUF&g):gguf(g),embedding(g.tensor("token_embd.weight")),output_norm(g.tensor("output_norm.weight")),rope_freqs(g.tensor("rope_freqs.weight")),pre_projection(g.tensor("mtp.pre_projection.weight")),post_projection(g.tensor("mtp.post_projection.weight")){
    if(g.meta<std::string>("general.architecture")!="gemma4_assistant")throw std::runtime_error("draft model is not a Gemma 4 assistant");
    layers=g.meta_u64("gemma4_assistant.block_count");embedding_dim=g.meta_u64("gemma4_assistant.embedding_length");backbone_dim=g.meta_u64("gemma4_assistant.n_embd_backbone");ffn_dim=g.meta_u64("gemma4_assistant.feed_forward_length");heads=g.meta_u64("gemma4_assistant.attention.head_count");window=g.meta_u64("gemma4_assistant.attention.sliding_window");vocab=embedding.shape.at(1);epsilon=g.meta_number("gemma4_assistant.attention.layer_norm_rms_epsilon");
    if(layers!=4||embedding_dim!=1024||backbone_dim!=3840||ffn_dim!=8192||heads!=16||vocab!=262144)throw std::runtime_error("unsupported Gemma 4 assistant layout");
    shape(embedding,embedding_dim,vocab);shape(output_norm,embedding_dim);shape(rope_freqs,256);shape(pre_projection,2*backbone_dim,embedding_dim);shape(post_projection,embedding_dim,backbone_dim);type(embedding,TensorType::Q6_K);type(pre_projection,TensorType::Q4_K);type(post_projection,TensorType::Q4_K);
    const auto&swa=g.meta<std::vector<uint8_t>>("gemma4_assistant.attention.sliding_window_pattern");if(swa.size()!=layers)throw std::runtime_error("invalid assistant layer metadata");
    layer.reserve(layers);for(uint32_t i=0;i<layers;++i){const std::string p="blk."+std::to_string(i)+".";Gemma4AssistantLayer l{};l.swa=swa[i];l.head_dim=l.swa?256:512;
        l.attn_norm=&g.tensor(p+"attn_norm.weight");l.wq=&g.tensor(p+"attn_q.weight");l.wo=&g.tensor(p+"attn_output.weight");l.q_norm=&g.tensor(p+"attn_q_norm.weight");l.attn_post_norm=&g.tensor(p+"post_attention_norm.weight");l.ffn_norm=&g.tensor(p+"ffn_norm.weight");l.gate=&g.tensor(p+"ffn_gate.weight");l.up=&g.tensor(p+"ffn_up.weight");l.down=&g.tensor(p+"ffn_down.weight");l.ffn_post_norm=&g.tensor(p+"post_ffw_norm.weight");l.out_scale=&g.tensor(p+"layer_output_scale.weight");
        shape(*l.wq,embedding_dim,uint64_t(heads)*l.head_dim);shape(*l.wo,uint64_t(heads)*l.head_dim,embedding_dim);shape(*l.gate,embedding_dim,ffn_dim);shape(*l.up,embedding_dim,ffn_dim);shape(*l.down,ffn_dim,embedding_dim);type(*l.wq,TensorType::Q4_K);type(*l.wo,TensorType::Q4_K);type(*l.gate,TensorType::Q4_K);type(*l.up,TensorType::Q4_K);q4_or_q6(*l.down);layer.push_back(l);}
}
}
