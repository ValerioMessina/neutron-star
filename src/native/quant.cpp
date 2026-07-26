#include "neutron/native/quant.hpp"
#include <bit>
#include <cstring>
#include <stdexcept>

namespace neutron::native {
namespace {
float fp16(const std::byte * p){uint16_t u;std::memcpy(&u,p,2);return static_cast<float>(std::bit_cast<_Float16>(u));}
float q4(const std::byte *b,size_t i){
    const auto*u=reinterpret_cast<const uint8_t*>(b);const float d=fp16(b),dm=fp16(b+2);const uint8_t*s=u+4,*q=u+16;
    const size_t g=i/32,l=i%32;uint8_t sc,m;if(g<4){sc=s[g]&63;m=s[g+4]&63;}else{sc=(s[g+4]&15)|((s[g-4]>>6)<<4);m=(s[g+4]>>4)|((s[g]>>6)<<4);}const uint8_t v=(g&1)?q[(g/2)*32+l]>>4:q[(g/2)*32+l]&15;return d*sc*v-dm*m;
}
float q6(const std::byte *b,size_t i){
    const auto*u=reinterpret_cast<const uint8_t*>(b);const uint8_t*ql=u,*qh=u+128;const int8_t*sc=reinterpret_cast<const int8_t*>(u+192);const float d=fp16(b+208);
    const size_t half=i/128,p=i%128,l=p%32,quarter=p/32;const uint8_t*lo=ql+half*64,*hi=qh+half*32;int v;
    if(quarter==0)v=(lo[l]&15)|(((hi[l]>>0)&3)<<4);else if(quarter==1)v=(lo[l+32]&15)|(((hi[l]>>2)&3)<<4);else if(quarter==2)v=(lo[l]>>4)|(((hi[l]>>4)&3)<<4);else v=(lo[l+32]>>4)|(((hi[l]>>6)&3)<<4);
    return d*sc[half*8+(l/16)+quarter*2]*(v-32);
}
}
float dequant(const Tensor&t,size_t e){if(e>=([&]{size_t n=1;for(auto d:t.shape)n*=d;return n;}()))throw std::runtime_error("dequant index");if(t.type==TensorType::F32)return reinterpret_cast<const float*>(t.data)[e];const size_t block=e/256,in=e%256,bs=t.type==TensorType::Q4_K?144:210;return t.type==TensorType::Q4_K?q4(t.data+block*bs,in):q6(t.data+block*bs,in);}
float dot_row(const Tensor&t,size_t row,const float*x){const size_t cols=t.shape.at(0);float sum=0;for(size_t i=0;i<cols;++i)sum+=dequant(t,row*cols+i)*x[i];return sum;}
}
