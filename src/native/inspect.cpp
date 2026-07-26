#include "neutron/native/gguf.hpp"
#include "neutron/native/tokenizer.hpp"
#include "neutron/native/model.hpp"

#include <iostream>

int main(int argc,char**argv){
    if(argc<2){std::cerr<<"usage: neutron-native-inspect MODEL.gguf [TEXT]\n";return 2;}
    try{
        neutron::native::GGUF g(argv[1],false);neutron::native::Gemma4Tokenizer tok(g);const auto arch=g.meta<std::string>("general.architecture");
        if(argc==4&&std::string(argv[2])=="--compare-vocab"){
            neutron::native::GGUF other(argv[3],false);const auto&a=g.meta<std::vector<std::string>>("tokenizer.ggml.tokens");const auto&b=other.meta<std::vector<std::string>>("tokenizer.ggml.tokens");size_t mismatches=0,first=0;for(size_t i=0;i<std::min(a.size(),b.size());++i)if(a[i]!=b[i]){if(!mismatches)first=i;++mismatches;}mismatches+=a.size()>b.size()?a.size()-b.size():b.size()-a.size();std::cout<<"vocab_a="<<a.size()<<" vocab_b="<<b.size()<<" mismatches="<<mismatches;if(mismatches&&first<std::min(a.size(),b.size()))std::cout<<" first="<<first<<" a="<<a[first]<<" b="<<b[first];std::cout<<'\n';return mismatches?1:0;
        }
        std::cout<<"architecture="<<arch<<" tensors="<<g.tensors().size()<<" vocab="<<tok.size()<<" data_offset="<<g.data_offset()<<"\n";
        if(arch=="gemma4_assistant"){
            std::cout<<"layers="<<g.meta_u64("gemma4_assistant.block_count")<<" embedding="<<g.meta_u64("gemma4_assistant.embedding_length")<<" backbone="<<g.meta_u64("gemma4_assistant.n_embd_backbone")<<" context="<<g.meta_u64("gemma4_assistant.context_length")<<" shared_kv_layers="<<g.meta_u64("gemma4_assistant.attention.shared_kv_layers")<<"\n";
            const auto&swa=g.meta<std::vector<uint8_t>>("gemma4_assistant.attention.sliding_window_pattern");std::cout<<"heads="<<g.meta_u64("gemma4_assistant.attention.head_count")<<" swa=";for(auto v:swa)std::cout<<int(v)<<',';std::cout<<"\n";
            if(argc>2&&std::string(argv[2])=="--layers")for(const auto&t:g.tensors()){std::cout<<t.name<<" shape=";for(size_t i=0;i<t.shape.size();++i)std::cout<<(i?"x":"")<<t.shape[i];std::cout<<" type="<<uint32_t(t.type)<<" bytes="<<t.bytes<<"\n";}
            return 0;
        }
        neutron::native::Gemma4Model model(g);
        std::cout<<"layers="<<g.meta_u64("gemma4.block_count")<<" embedding="<<g.meta_u64("gemma4.embedding_length")<<" context="<<g.meta_u64("gemma4.context_length")<<"\n";
        std::cout<<"native_layout=validated swa_layers=";size_t n=0;for(const auto&l:model.layer)n+=l.swa;std::cout<<n<<" global_layers="<<model.layers-n<<"\n";
        bool mtp=false;for(const auto&t:g.tensors())if(t.name.find("nextn")!=std::string::npos||t.name.find("eh_proj")!=std::string::npos||t.name.find("assistant")!=std::string::npos){mtp=true;break;}
        std::cout<<"embedded_mtp_tensors="<<(mtp?"present":"absent")<<"\n";
        if(argc>2&&std::string(argv[2])=="--layers"){for(size_t i=0;i<model.layer.size();++i){const auto&l=model.layer[i];std::cout<<"layer="<<i<<" swa="<<l.swa<<" q_rows="<<l.wq->shape[1]<<" kv_rows="<<l.wk->shape[1]<<" kv_source="<<l.kv_source<<" v="<<(l.wv?l.wv->type==neutron::native::TensorType::Q6_K?"q6":"q4":"tied")<<" down="<<(l.down->type==neutron::native::TensorType::Q6_K?"q6":"q4")<<"\n";}}else if(argc>2){auto ids=tok.encode(argv[2]);std::cout<<"tokens:";for(auto id:ids)std::cout<<' '<<id;std::cout<<"\ndecoded="<<tok.decode(ids,true)<<"\n";}
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}
}
