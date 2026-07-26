#pragma once
#include "neutron/native/gguf.hpp"
#include <cstdint>
#include <vector>
namespace neutron::native {
struct Gemma4Layer {
    const Tensor *attn_norm,*wq,*wk,*wv,*wo,*q_norm,*k_norm,*attn_post_norm;
    const Tensor *ffn_norm,*gate,*up,*down,*ffn_post_norm,*out_scale;
    uint32_t head_dim=0,kv_heads=0;int32_t kv_source=-1;bool swa=false;
};
class Gemma4Model {
public:
    explicit Gemma4Model(const GGUF&);
    const GGUF&gguf;const Tensor&embedding;const Tensor&output_norm;const Tensor&rope_freqs;
    uint32_t layers=0,embedding_dim=0,ffn_dim=0,heads=0,context=0,window=0,vocab=0;
    float epsilon=1e-6f,logit_cap=30;std::vector<Gemma4Layer>layer;
};

struct Gemma4AssistantLayer {
    const Tensor *attn_norm,*wq,*wo,*q_norm,*attn_post_norm;
    const Tensor *ffn_norm,*gate,*up,*down,*ffn_post_norm,*out_scale;
    uint32_t head_dim=0;int32_t kv_source=-1;bool swa=false;
};

class Gemma4AssistantModel {
public:
    explicit Gemma4AssistantModel(const GGUF&);
    const GGUF&gguf;const Tensor&embedding;const Tensor&output_norm;const Tensor&rope_freqs;
    const Tensor&pre_projection;const Tensor&post_projection;
    uint32_t layers=0,embedding_dim=0,backbone_dim=0,ffn_dim=0,heads=0,vocab=0,window=0;
    float epsilon=1e-6f;std::vector<Gemma4AssistantLayer>layer;
};
}
