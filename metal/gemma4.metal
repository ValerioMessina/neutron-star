#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct Q4K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct Q6K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
struct PackedQ4Group { half2 dm; uchar qs[16]; };
struct MatArgs { uint cols; uint rows; uint batch; ulong weight_offset; };
struct PairMatArgs { uint cols; uint rows; uint batch; uint pad; ulong weight0_offset; ulong weight1_offset; };
struct QKVArgs { uint cols; uint q_rows; uint kv_rows; uint batch; ulong q_offset; ulong k_offset; ulong v_offset; };

inline float q4_value(device const Q4K &b,uint i){
    uint g=i/32,l=i%32;uchar sc,m;
    if(g<4){sc=b.scales[g]&63;m=b.scales[g+4]&63;}else{sc=(b.scales[g+4]&15)|((b.scales[g-4]>>6)<<4);m=(b.scales[g+4]>>4)|((b.scales[g]>>6)<<4);}
    uchar v=(g&1)?b.qs[(g/2)*32+l]>>4:b.qs[(g/2)*32+l]&15;
    return float(b.d)*float(sc)*float(v)-float(b.dmin)*float(m);
}
inline float q6_value(device const Q6K &b,uint i){
    uint h=i/128,p=i%128,l=p%32,q=p/32;uint v;
    if(q==0)v=(b.ql[h*64+l]&15)|(((b.qh[h*32+l]>>0)&3)<<4);
    else if(q==1)v=(b.ql[h*64+l+32]&15)|(((b.qh[h*32+l]>>2)&3)<<4);
    else if(q==2)v=(b.ql[h*64+l]>>4)|(((b.qh[h*32+l]>>4)&3)<<4);
    else v=(b.ql[h*64+l+32]>>4)|(((b.qh[h*32+l]>>6)&3)<<4);
    return float(b.d)*float(b.scales[h*8+(l/16)+q*2])*float(int(v)-32);
}

template<typename B,float value(device const B&,uint)>
inline void quant_mv(device const uchar*weights,device const float*x,device float*y,constant MatArgs&a,uint3 tg,uint tid,uint lane,uint simd,uint nsg,threadgroup float*partial){
    if(tg.x>=a.rows||tg.y>=a.batch)return;device const B*w=(device const B*)(weights+a.weight_offset);uint blocks=a.cols/256;float sum=0;
    for(uint i=tid;i<a.cols;i+=256)sum+=value(w[tg.x*blocks+i/256],i%256)*x[tg.y*a.cols+i];
    sum=simd_sum(sum);if(lane==0)partial[simd]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float total=0;for(uint i=0;i<nsg;++i)total+=partial[i];y[tg.y*a.rows+tg.x]=total;}
}
kernel void gemma_q4k_mv(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){quant_mv<Q4K,q4_value>(w,x,y,a,tg,tid,lane,simd,nsg,p);}
kernel void gemma_q6k_mv(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){quant_mv<Q6K,q6_value>(w,x,y,a,tg,tid,lane,simd,nsg,p);}

template<uint NR,typename B,float value(device const B&,uint)>
inline void quant_mvN(device const uchar*weights,device const float*x,device float*y,constant MatArgs&a,uint group,uint tid,uint nt,uint lane,uint simd,uint nsg,threadgroup float*partial){
    uint r0=group*NR;if(r0>=a.rows)return;device const B*w=(device const B*)(weights+a.weight_offset);uint blocks=a.cols/256;float sums[NR];for(uint r=0;r<NR;++r)sums[r]=0;
    for(uint i=tid;i<a.cols;i+=nt){float xv=x[i];for(uint r=0;r<NR;++r)if(r0+r<a.rows)sums[r]+=value(w[(r0+r)*blocks+i/256],i%256)*xv;}
    for(uint r=0;r<NR;++r){sums[r]=simd_sum(sums[r]);if(lane==0)partial[r*nsg+simd]=sums[r];}threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0)for(uint r=0;r<NR&&r0+r<a.rows;++r){float z=0;for(uint s=0;s<nsg;++s)z+=partial[r*nsg+s];y[r0+r]=z;}
}
kernel void gemma_q4k_mv4(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint nt[[threads_per_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){quant_mvN<4,Q4K,q4_value>(w,x,y,a,group,tid,nt,lane,simd,nsg,p);}
kernel void gemma_q6k_mv4(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint nt[[threads_per_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){quant_mvN<4,Q6K,q6_value>(w,x,y,a,group,tid,nt,lane,simd,nsg,p);}

template<typename B,float value(device const B&,uint)>
inline void quant_mv_blocks4(device const uchar*weights,device const float*x,device float*y,constant MatArgs&a,uint group,uint lane){
    uint r0=group*4;if(r0>=a.rows)return;device const B*w=(device const B*)(weights+a.weight_offset);uint blocks=a.cols/256;float sums[4]={0,0,0,0};
    for(uint ib=lane;ib<blocks;ib+=32)for(uint j=0;j<256;++j){float xv=x[ib*256+j];for(uint r=0;r<4;++r)if(r0+r<a.rows)sums[r]+=value(w[(r0+r)*blocks+ib],j)*xv;}
    for(uint r=0;r<4;++r){float z=simd_sum(sums[r]);if(lane==0&&r0+r<a.rows)y[r0+r]=z;}
}
kernel void gemma_q4k_mv_blocks4(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){quant_mv_blocks4<Q4K,q4_value>(w,x,y,a,group,lane);}
kernel void gemma_q6k_mv_blocks4(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]]){quant_mv_blocks4<Q6K,q6_value>(w,x,y,a,group,lane);}

kernel void gemma_q4k_mv_fast(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=4;uint first=(group*NSG+sg)*NR;if(first>=a.rows)return;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums[NR]={0,0,0,0};device const float*y4=y+ix*256+64*iq+8*ir;
    for(uint ib=ix;ib<nb;ib+=4){float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];}
        for(short row=0;row<NR&&first+row<a.rows;++row){device const Q4K&b=w[(first+row)*nb+ib];device const ushort*sc=(device const ushort*)b.scales+iq;device const ushort*q1=(device const ushort*)b.qs+16*iq+4*ir;device const ushort*q2=q1+32;ushort st[4];thread uchar*s=(thread uchar*)st;st[0]=sc[0]&0x3f3f;st[1]=sc[2]&0x3f3f;st[2]=((sc[4]>>0)&0x0f0f)|((sc[0]&0xc0c0)>>2);st[3]=((sc[4]>>4)&0x0f0f)|((sc[2]&0xc0c0)>>2);float4 a1=0,a2=0;
            for(short i=0;i<4;++i){a1[0]+=yl[2*i]*(q1[i]&0x000f);a1[1]+=yl[2*i+1]*(q1[i]&0x0f00);a1[2]+=yl[2*i+8]*(q1[i]&0x00f0);a1[3]+=yl[2*i+9]*(q1[i]&0xf000);a2[0]+=yh[2*i]*(q2[i]&0x000f);a2[1]+=yh[2*i+1]*(q2[i]&0x0f00);a2[2]+=yh[2*i+8]*(q2[i]&0x00f0);a2[3]+=yh[2*i+9]*(q2[i]&0xf000);}
            sums[row]+=float(b.d)*((a1[0]+a1[1]/256.0f)*s[0]+(a1[2]+a1[3]/256.0f)*s[1]/16.0f+(a2[0]+a2[1]/256.0f)*s[4]+(a2[2]+a2[3]/256.0f)*s[5]/16.0f)-float(b.dmin)*(sumy[0]*s[2]+sumy[1]*s[3]+sumy[2]*s[6]+sumy[3]*s[7]);}
        y4+=4*256;}
    for(short r=0;r<NR&&first+r<a.rows;++r){float z=simd_sum(sums[r]);if(lane==0)out[first+r]=z;}
}
kernel void gemma_q4k_mv_batch(device const uchar*w0[[buffer(0)]],device const float*yin[[buffer(1)]],device float*out0[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=4;uint first=(group.x*NSG+sg)*NR;if(first>=a.rows||group.y>=a.batch)return;device const float*y=yin+group.y*a.cols;device float*out=out0+group.y*a.rows;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums[NR]={0,0,0,0};device const float*y4=y+ix*256+64*iq+8*ir;
    for(uint ib=ix;ib<nb;ib+=4){float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];}
        for(short row=0;row<NR&&first+row<a.rows;++row){device const Q4K&b=w[(first+row)*nb+ib];device const ushort*sc=(device const ushort*)b.scales+iq;device const ushort*q1=(device const ushort*)b.qs+16*iq+4*ir;device const ushort*q2=q1+32;ushort st[4];thread uchar*s=(thread uchar*)st;st[0]=sc[0]&0x3f3f;st[1]=sc[2]&0x3f3f;st[2]=((sc[4]>>0)&0x0f0f)|((sc[0]&0xc0c0)>>2);st[3]=((sc[4]>>4)&0x0f0f)|((sc[2]&0xc0c0)>>2);float4 a1=0,a2=0;
            for(short i=0;i<4;++i){a1[0]+=yl[2*i]*(q1[i]&0x000f);a1[1]+=yl[2*i+1]*(q1[i]&0x0f00);a1[2]+=yl[2*i+8]*(q1[i]&0x00f0);a1[3]+=yl[2*i+9]*(q1[i]&0xf000);a2[0]+=yh[2*i]*(q2[i]&0x000f);a2[1]+=yh[2*i+1]*(q2[i]&0x0f00);a2[2]+=yh[2*i+8]*(q2[i]&0x00f0);a2[3]+=yh[2*i+9]*(q2[i]&0xf000);}
            sums[row]+=float(b.d)*((a1[0]+a1[1]/256.0f)*s[0]+(a1[2]+a1[3]/256.0f)*s[1]/16.0f+(a2[0]+a2[1]/256.0f)*s[4]+(a2[2]+a2[3]/256.0f)*s[5]/16.0f)-float(b.dmin)*(sumy[0]*s[2]+sumy[1]*s[3]+sumy[2]*s[6]+sumy[3]*s[7]);}y4+=4*256;}
    for(short r=0;r<NR&&first+r<a.rows;++r){float z=simd_sum(sums[r]);if(lane==0)out[first+r]=z;}
}

inline float q4k_mv_block_fast(device const Q4K&b,short iq,short ir,thread const float*yl,thread const float*yh,float4 sumy){device const ushort*sc=(device const ushort*)b.scales+iq;device const ushort*q1=(device const ushort*)b.qs+16*iq+4*ir;device const ushort*q2=q1+32;ushort st[4];thread uchar*s=(thread uchar*)st;st[0]=sc[0]&0x3f3f;st[1]=sc[2]&0x3f3f;st[2]=((sc[4]>>0)&0x0f0f)|((sc[0]&0xc0c0)>>2);st[3]=((sc[4]>>4)&0x0f0f)|((sc[2]&0xc0c0)>>2);float4 a1=0,a2=0;for(short i=0;i<4;++i){a1[0]+=yl[2*i]*(q1[i]&0x000f);a1[1]+=yl[2*i+1]*(q1[i]&0x0f00);a1[2]+=yl[2*i+8]*(q1[i]&0x00f0);a1[3]+=yl[2*i+9]*(q1[i]&0xf000);a2[0]+=yh[2*i]*(q2[i]&0x000f);a2[1]+=yh[2*i+1]*(q2[i]&0x0f00);a2[2]+=yh[2*i+8]*(q2[i]&0x00f0);a2[3]+=yh[2*i+9]*(q2[i]&0xf000);}return float(b.d)*((a1[0]+a1[1]/256.0f)*s[0]+(a1[2]+a1[3]/256.0f)*s[1]/16.0f+(a2[0]+a2[1]/256.0f)*s[4]+(a2[2]+a2[3]/256.0f)*s[5]/16.0f)-float(b.dmin)*(sumy[0]*s[2]+sumy[1]*s[3]+sumy[2]*s[6]+sumy[3]*s[7]);}

kernel void gemma_q4k_gate_up_geglu_mv(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=4;uint first=(group*NSG+sg)*NR;if(first>=a.rows)return;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums_g[NR]={0,0,0,0},sums_u[NR]={0,0,0,0};device const float*y4=y+ix*256+64*iq+8*ir;
    for(uint ib=ix;ib<nb;ib+=4){float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];}for(short row=0;row<NR&&first+row<a.rows;++row){sums_g[row]+=q4k_mv_block_fast(wg[(first+row)*nb+ib],iq,ir,yl,yh,sumy);sums_u[row]+=q4k_mv_block_fast(wu[(first+row)*nb+ib],iq,ir,yl,yh,sumy);}y4+=4*256;}
    for(short r=0;r<NR&&first+r<a.rows;++r){float g=simd_sum(sums_g[r]),u=simd_sum(sums_u[r]);if(lane==0){float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[first+r]=gelu*u;}}
}

kernel void gemma_q4k_gate_up_geglu_mv2(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=2;uint first=(group*NSG+sg)*NR;if(first>=a.rows)return;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums_g[NR]={0,0},sums_u[NR]={0,0};device const float*y4=y+ix*256+64*iq+8*ir;
    for(uint ib=ix;ib<nb;ib+=4){float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];}for(short row=0;row<NR&&first+row<a.rows;++row){sums_g[row]+=q4k_mv_block_fast(wg[(first+row)*nb+ib],iq,ir,yl,yh,sumy);sums_u[row]+=q4k_mv_block_fast(wu[(first+row)*nb+ib],iq,ir,yl,yh,sumy);}y4+=4*256;}
    for(short r=0;r<NR&&first+r<a.rows;++r){float g=simd_sum(sums_g[r]),u=simd_sum(sums_u[r]);if(lane==0){float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[first+r]=gelu*u;}}
}

kernel void gemma_q4k_mv_fast_metal(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=4;uint first=(group*NSG+sg)*NR;if(first>=a.rows)return;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums[NR]={0};device const float*y4=y+ix*256+64*iq+8*ir;
    for(uint ib=ix;ib<nb;ib+=4){float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];}for(short row=0;row<NR&&first+row<a.rows;++row){uint wr=first+row,at=((wr/64)*nb+ib)*64+wr%64;sums[row]+=q4k_mv_block_fast(w[at],iq,ir,yl,yh,sumy);}y4+=4*256;}
    for(short r=0;r<NR&&first+r<a.rows;++r){float z=simd_sum(sums[r]);if(lane==0)out[first+r]=z;}
}

kernel void gemma_q4k_gate_up_geglu_mv2_metal(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=2;uint first=(group*NSG+sg)*NR;if(first>=a.rows)return;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums_g[NR]={0},sums_u[NR]={0};device const float*y4=y+ix*256+64*iq+8*ir;
    for(uint ib=ix;ib<nb;ib+=4){float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];}for(short row=0;row<NR&&first+row<a.rows;++row){uint wr=first+row,at=((wr/64)*nb+ib)*64+wr%64;sums_g[row]+=q4k_mv_block_fast(wg[at],iq,ir,yl,yh,sumy);sums_u[row]+=q4k_mv_block_fast(wu[at],iq,ir,yl,yh,sumy);}y4+=4*256;}
    for(short r=0;r<NR&&first+r<a.rows;++r){float g=simd_sum(sums_g[r]),u=simd_sum(sums_u[r]);if(lane==0){float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[first+r]=gelu*u;}}
}

// Verification uses only 3-4 target rows.  Keep those rows in registers instead
// of padding them to the 32-token prefill tile, and emit the GEGLU intermediate
// directly as FP16 for the following W4A16 down projection.
kernel void gemma_q4k_gate_up_geglu_mm_f16(device const uchar*w0[[buffer(0)]],device const half*yin[[buffer(1)]],device half*out0[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=2,BT=4;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums_g[NR*BT]={0},sums_u[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=4)for(short t=0;t<BT&&t0+t<a.batch;++t){device const half*y4=yin+(t0+t)*a.cols+ib*256+64*iq+8*ir;float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=float(y4[i]);sumy[0]+=yl[i];yl[i+8]=float(y4[i+32]);sumy[1]+=yl[i+8];yh[i]=float(y4[i+128]);sumy[2]+=yh[i];yh[i+8]=float(y4[i+160]);sumy[3]+=yh[i+8];}for(short row=0;row<NR&&first+row<a.rows;++row){sums_g[row*BT+t]+=q4k_mv_block_fast(wg[(first+row)*nb+ib],iq,ir,yl,yh,sumy);sums_u[row*BT+t]+=q4k_mv_block_fast(wu[(first+row)*nb+ib],iq,ir,yl,yh,sumy);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float g=simd_sum(sums_g[r*BT+t]),u=simd_sum(sums_u[r*BT+t]);if(lane==0){float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out0[(t0+t)*a.rows+first+r]=half(gelu*u);}}
}

kernel void gemma_q4k_gate_up_geglu_mm_f16_metal(device const uchar*w0[[buffer(0)]],device const half*yin[[buffer(1)]],device half*out0[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=2,BT=4;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums_g[NR*BT]={0},sums_u[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=4)for(short t=0;t<BT&&t0+t<a.batch;++t){device const half*y4=yin+(t0+t)*a.cols+ib*256+64*iq+8*ir;float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=float(y4[i]);sumy[0]+=yl[i];yl[i+8]=float(y4[i+32]);sumy[1]+=yl[i+8];yh[i]=float(y4[i+128]);sumy[2]+=yh[i];yh[i+8]=float(y4[i+160]);sumy[3]+=yh[i+8];}for(short row=0;row<NR&&first+row<a.rows;++row){uint wr=first+row,at=((wr/64)*nb+ib)*64+wr%64;sums_g[row*BT+t]+=q4k_mv_block_fast(wg[at],iq,ir,yl,yh,sumy);sums_u[row*BT+t]+=q4k_mv_block_fast(wu[at],iq,ir,yl,yh,sumy);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float g=simd_sum(sums_g[r*BT+t]),u=simd_sum(sums_u[r*BT+t]);if(lane==0){float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out0[(t0+t)*a.rows+first+r]=half(gelu*u);}}
}

kernel void gemma_q6k_mv_fast(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=2,NR=2;uint first=(group*NSG+sg)*NR;if(first>=a.rows)return;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;short tid=lane/2,ix=lane%2,ip=tid/8,il=tid%8,l0=4*il,is=8*ip+l0/16;uint yo=128*ip+l0,ql_off=64*ip+l0,qh_off=32*ip+l0;float sums[NR]={0,0};
    for(uint ib=ix;ib<nb;ib+=2){float yl[16];device const float*yy=y+ib*256+yo;for(short l=0;l<4;++l){yl[4*l]=yy[l];yl[4*l+1]=yy[l+32];yl[4*l+2]=yy[l+64];yl[4*l+3]=yy[l+96];}
        for(short row=0;row<NR&&first+row<a.rows;++row){device const Q6K&b=w[(first+row)*nb+ib];device const uchar*q1=b.ql+ql_off,*q2=q1+32,*qh=b.qh+qh_off;device const char*sc=b.scales+is;float4 z=0;for(short l=0;l<4;++l){z[0]+=yl[4*l]*(int((q1[l]&15)|((qh[l]&3)<<4))-32);z[1]+=yl[4*l+1]*(int((q2[l]&15)|((qh[l]&12)<<2))-32);z[2]+=yl[4*l+2]*(int((q1[l]>>4)|((qh[l]&48)<<0))-32);z[3]+=yl[4*l+3]*(int((q2[l]>>4)|((qh[l]&192)>>2))-32);}sums[row]+=float(b.d)*(z[0]*sc[0]+z[1]*sc[2]+z[2]*sc[4]+z[3]*sc[6]);}}
    for(short r=0;r<NR&&first+r<a.rows;++r){float z=simd_sum(sums[r]);if(lane==0)out[first+r]=z;}
}
kernel void gemma_q6k_mv_fast_metal(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=2,NR=2;uint first=(group*NSG+sg)*NR;if(first>=a.rows)return;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;short tid=lane/2,ix=lane%2,ip=tid/8,il=tid%8,l0=4*il,is=8*ip+l0/16;uint yo=128*ip+l0,ql_off=64*ip+l0,qh_off=32*ip+l0;float sums[NR]={0};
    for(uint ib=ix;ib<nb;ib+=2){float yl[16];device const float*yy=y+ib*256+yo;for(short l=0;l<4;++l){yl[4*l]=yy[l];yl[4*l+1]=yy[l+32];yl[4*l+2]=yy[l+64];yl[4*l+3]=yy[l+96];}for(short row=0;row<NR&&first+row<a.rows;++row){uint wr=first+row,at=((wr/64)*nb+ib)*64+wr%64;device const Q6K&b=w[at];device const uchar*q1=b.ql+ql_off,*q2=q1+32,*qh=b.qh+qh_off;device const char*sc=b.scales+is;float4 z=0;for(short l=0;l<4;++l){z[0]+=yl[4*l]*(int((q1[l]&15)|((qh[l]&3)<<4))-32);z[1]+=yl[4*l+1]*(int((q2[l]&15)|((qh[l]&12)<<2))-32);z[2]+=yl[4*l+2]*(int((q1[l]>>4)|((qh[l]&48)<<0))-32);z[3]+=yl[4*l+3]*(int((q2[l]>>4)|((qh[l]&192)>>2))-32);}sums[row]+=float(b.d)*(z[0]*sc[0]+z[1]*sc[2]+z[2]*sc[4]+z[3]*sc[6]);}}
    for(short r=0;r<NR&&first+r<a.rows;++r){float z=simd_sum(sums[r]);if(lane==0)out[first+r]=z;}
}
kernel void gemma_q6k_mv_batch(device const uchar*w0[[buffer(0)]],device const float*yin[[buffer(1)]],device float*out0[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=2,NR=2;uint first=(group.x*NSG+sg)*NR;if(first>=a.rows||group.y>=a.batch)return;device const float*y=yin+group.y*a.cols;device float*out=out0+group.y*a.rows;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;short tid=lane/2,ix=lane%2,ip=tid/8,il=tid%8,l0=4*il,is=8*ip+l0/16;uint yo=128*ip+l0,ql_off=64*ip+l0,qh_off=32*ip+l0;float sums[NR]={0,0};
    for(uint ib=ix;ib<nb;ib+=2){float yl[16];device const float*yy=y+ib*256+yo;for(short l=0;l<4;++l){yl[4*l]=yy[l];yl[4*l+1]=yy[l+32];yl[4*l+2]=yy[l+64];yl[4*l+3]=yy[l+96];}for(short row=0;row<NR&&first+row<a.rows;++row){device const Q6K&b=w[(first+row)*nb+ib];device const uchar*q1=b.ql+ql_off,*q2=q1+32,*qh=b.qh+qh_off;device const char*sc=b.scales+is;float4 z=0;for(short l=0;l<4;++l){z[0]+=yl[4*l]*(int((q1[l]&15)|((qh[l]&3)<<4))-32);z[1]+=yl[4*l+1]*(int((q2[l]&15)|((qh[l]&12)<<2))-32);z[2]+=yl[4*l+2]*(int((q1[l]>>4)|((qh[l]&48)<<0))-32);z[3]+=yl[4*l+3]*(int((q2[l]>>4)|((qh[l]&192)>>2))-32);}sums[row]+=float(b.d)*(z[0]*sc[0]+z[1]*sc[2]+z[2]*sc[4]+z[3]*sc[6]);}}
    for(short r=0;r<NR&&first+r<a.rows;++r){float z=simd_sum(sums[r]);if(lane==0)out[first+r]=z;}
}

kernel void gemma_q4k_mm_fast(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=4,BT=4;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=4)for(short row=0;row<NR&&first+row<a.rows;++row){device const Q4K&b=w[(first+row)*nb+ib];device const ushort*sc=(device const ushort*)b.scales+iq;device const ushort*q1=(device const ushort*)b.qs+16*iq+4*ir;device const ushort*q2=q1+32;ushort st[4];thread uchar*s=(thread uchar*)st;st[0]=sc[0]&0x3f3f;st[1]=sc[2]&0x3f3f;st[2]=((sc[4]>>0)&0x0f0f)|((sc[0]&0xc0c0)>>2);st[3]=((sc[4]>>4)&0x0f0f)|((sc[2]&0xc0c0)>>2);
        for(short t=0;t<BT&&t0+t<a.batch;++t){device const float*y4=y+(t0+t)*a.cols+ib*256+64*iq+8*ir;float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];}float4 a1=0,a2=0;for(short i=0;i<4;++i){a1[0]+=yl[2*i]*(q1[i]&0x000f);a1[1]+=yl[2*i+1]*(q1[i]&0x0f00);a1[2]+=yl[2*i+8]*(q1[i]&0x00f0);a1[3]+=yl[2*i+9]*(q1[i]&0xf000);a2[0]+=yh[2*i]*(q2[i]&0x000f);a2[1]+=yh[2*i+1]*(q2[i]&0x0f00);a2[2]+=yh[2*i+8]*(q2[i]&0x00f0);a2[3]+=yh[2*i+9]*(q2[i]&0xf000);}sums[row*BT+t]+=float(b.d)*((a1[0]+a1[1]/256.0f)*s[0]+(a1[2]+a1[3]/256.0f)*s[1]/16.0f+(a2[0]+a2[1]/256.0f)*s[4]+(a2[2]+a2[3]/256.0f)*s[5]/16.0f)-float(b.dmin)*(sumy[0]*s[2]+sumy[1]*s[3]+sumy[2]*s[6]+sumy[3]*s[7]);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float z=simd_sum(sums[r*BT+t]);if(lane==0)out[(t0+t)*a.rows+first+r]=z;}
}

kernel void gemma_q6k_mm_fast(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=2,NR=2,BT=2;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;short tid=lane/2,ix=lane%2,ip=tid/8,il=tid%8,l0=4*il,is=8*ip+l0/16;uint yo=128*ip+l0,ql_off=64*ip+l0,qh_off=32*ip+l0;float sums[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=2)for(short row=0;row<NR&&first+row<a.rows;++row){device const Q6K&b=w[(first+row)*nb+ib];device const uchar*q1=b.ql+ql_off,*q2=q1+32,*qh=b.qh+qh_off;device const char*sc=b.scales+is;
        for(short t=0;t<BT&&t0+t<a.batch;++t){device const float*yy=y+(t0+t)*a.cols+ib*256+yo;float yl[16];for(short l=0;l<4;++l){yl[4*l]=yy[l];yl[4*l+1]=yy[l+32];yl[4*l+2]=yy[l+64];yl[4*l+3]=yy[l+96];}float4 z=0;for(short l=0;l<4;++l){z[0]+=yl[4*l]*(int((q1[l]&15)|((qh[l]&3)<<4))-32);z[1]+=yl[4*l+1]*(int((q2[l]&15)|((qh[l]&12)<<2))-32);z[2]+=yl[4*l+2]*(int((q1[l]>>4)|((qh[l]&48)<<0))-32);z[3]+=yl[4*l+3]*(int((q2[l]>>4)|((qh[l]&192)>>2))-32);}sums[row*BT+t]+=float(b.d)*(z[0]*sc[0]+z[1]*sc[2]+z[2]*sc[4]+z[3]*sc[6]);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float z=simd_sum(sums[r*BT+t]);if(lane==0)out[(t0+t)*a.rows+first+r]=z;}
}

kernel void gemma_q4k_mm_fast_metal(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=4,BT=4;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=4)for(short row=0;row<NR&&first+row<a.rows;++row){uint wr=first+row,at=((wr/64)*nb+ib)*64+wr%64;device const Q4K&b=w[at];for(short t=0;t<BT&&t0+t<a.batch;++t){device const float*y4=y+(t0+t)*a.cols+ib*256+64*iq+8*ir;float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];}sums[row*BT+t]+=q4k_mv_block_fast(b,iq,ir,yl,yh,sumy);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float z=simd_sum(sums[r*BT+t]);if(lane==0)out[(t0+t)*a.rows+first+r]=z;}
}

kernel void gemma_q6k_mm_fast_metal(device const uchar*w0[[buffer(0)]],device const float*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=2,NR=2,BT=2;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;short tid=lane/2,ix=lane%2,ip=tid/8,il=tid%8,l0=4*il,is=8*ip+l0/16;uint yo=128*ip+l0,ql_off=64*ip+l0,qh_off=32*ip+l0;float sums[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=2)for(short row=0;row<NR&&first+row<a.rows;++row){uint wr=first+row,at=((wr/64)*nb+ib)*64+wr%64;device const Q6K&b=w[at];device const uchar*q1=b.ql+ql_off,*q2=q1+32,*qh=b.qh+qh_off;device const char*sc=b.scales+is;for(short t=0;t<BT&&t0+t<a.batch;++t){device const float*yy=y+(t0+t)*a.cols+ib*256+yo;float yl[16];for(short l=0;l<4;++l){yl[4*l]=yy[l];yl[4*l+1]=yy[l+32];yl[4*l+2]=yy[l+64];yl[4*l+3]=yy[l+96];}float4 z=0;for(short l=0;l<4;++l){z[0]+=yl[4*l]*(int((q1[l]&15)|((qh[l]&3)<<4))-32);z[1]+=yl[4*l+1]*(int((q2[l]&15)|((qh[l]&12)<<2))-32);z[2]+=yl[4*l+2]*(int((q1[l]>>4)|((qh[l]&48)<<0))-32);z[3]+=yl[4*l+3]*(int((q2[l]>>4)|((qh[l]&192)>>2))-32);}sums[row*BT+t]+=float(b.d)*(z[0]*sc[0]+z[1]*sc[2]+z[2]*sc[4]+z[3]*sc[6]);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float z=simd_sum(sums[r*BT+t]);if(lane==0)out[(t0+t)*a.rows+first+r]=z;}
}

kernel void gemma_q4k_mm_fast_f16(device const uchar*w0[[buffer(0)]],device const half*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=4,BT=4;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=4)for(short row=0;row<NR&&first+row<a.rows;++row){device const Q4K&b=w[(first+row)*nb+ib];for(short t=0;t<BT&&t0+t<a.batch;++t){device const half*y4=y+(t0+t)*a.cols+ib*256+64*iq+8*ir;float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=float(y4[i]);sumy[0]+=yl[i];yl[i+8]=float(y4[i+32]);sumy[1]+=yl[i+8];yh[i]=float(y4[i+128]);sumy[2]+=yh[i];yh[i+8]=float(y4[i+160]);sumy[3]+=yh[i+8];}sums[row*BT+t]+=q4k_mv_block_fast(b,iq,ir,yl,yh,sumy);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float z=simd_sum(sums[r*BT+t]);if(lane==0)out[(t0+t)*a.rows+first+r]=z;}
}
kernel void gemma_q4k_mm_fast_f16_metal(device const uchar*w0[[buffer(0)]],device const half*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=4,NR=4,BT=4;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;float sums[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=4)for(short row=0;row<NR&&first+row<a.rows;++row){uint wr=first+row,at=((wr/64)*nb+ib)*64+wr%64;device const Q4K&b=w[at];for(short t=0;t<BT&&t0+t<a.batch;++t){device const half*y4=y+(t0+t)*a.cols+ib*256+64*iq+8*ir;float yl[16],yh[16];float4 sumy=0;for(short i=0;i<8;++i){yl[i]=float(y4[i]);sumy[0]+=yl[i];yl[i+8]=float(y4[i+32]);sumy[1]+=yl[i+8];yh[i]=float(y4[i+128]);sumy[2]+=yh[i];yh[i+8]=float(y4[i+160]);sumy[3]+=yh[i+8];}sums[row*BT+t]+=q4k_mv_block_fast(b,iq,ir,yl,yh,sumy);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float z=simd_sum(sums[r*BT+t]);if(lane==0)out[(t0+t)*a.rows+first+r]=z;}
}

kernel void gemma_q6k_mm_fast_f16(device const uchar*w0[[buffer(0)]],device const half*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=2,NR=2,BT=2;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;short tid=lane/2,ix=lane%2,ip=tid/8,il=tid%8,l0=4*il,is=8*ip+l0/16;uint yo=128*ip+l0,ql_off=64*ip+l0,qh_off=32*ip+l0;float sums[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=2)for(short row=0;row<NR&&first+row<a.rows;++row){device const Q6K&b=w[(first+row)*nb+ib];device const uchar*q1=b.ql+ql_off,*q2=q1+32,*qh=b.qh+qh_off;device const char*sc=b.scales+is;for(short t=0;t<BT&&t0+t<a.batch;++t){device const half*yy=y+(t0+t)*a.cols+ib*256+yo;float yl[16];for(short l=0;l<4;++l){yl[4*l]=float(yy[l]);yl[4*l+1]=float(yy[l+32]);yl[4*l+2]=float(yy[l+64]);yl[4*l+3]=float(yy[l+96]);}float4 z=0;for(short l=0;l<4;++l){z[0]+=yl[4*l]*(int((q1[l]&15)|((qh[l]&3)<<4))-32);z[1]+=yl[4*l+1]*(int((q2[l]&15)|((qh[l]&12)<<2))-32);z[2]+=yl[4*l+2]*(int((q1[l]>>4)|((qh[l]&48)<<0))-32);z[3]+=yl[4*l+3]*(int((q2[l]>>4)|((qh[l]&192)>>2))-32);}sums[row*BT+t]+=float(b.d)*(z[0]*sc[0]+z[1]*sc[2]+z[2]*sc[4]+z[3]*sc[6]);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float z=simd_sum(sums[r*BT+t]);if(lane==0)out[(t0+t)*a.rows+first+r]=z;}
}
kernel void gemma_q6k_mm_fast_f16_metal(device const uchar*w0[[buffer(0)]],device const half*y[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){
    constexpr ushort NSG=2,NR=2,BT=2;uint first=(group.x*NSG+sg)*NR,t0=group.y*BT;if(first>=a.rows||t0>=a.batch)return;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;short tid=lane/2,ix=lane%2,ip=tid/8,il=tid%8,l0=4*il,is=8*ip+l0/16;uint yo=128*ip+l0,ql_off=64*ip+l0,qh_off=32*ip+l0;float sums[NR*BT]={0};
    for(uint ib=ix;ib<nb;ib+=2)for(short row=0;row<NR&&first+row<a.rows;++row){uint wr=first+row,at=((wr/64)*nb+ib)*64+wr%64;device const Q6K&b=w[at];device const uchar*q1=b.ql+ql_off,*q2=q1+32,*qh=b.qh+qh_off;device const char*sc=b.scales+is;for(short t=0;t<BT&&t0+t<a.batch;++t){device const half*yy=y+(t0+t)*a.cols+ib*256+yo;float yl[16];for(short l=0;l<4;++l){yl[4*l]=float(yy[l]);yl[4*l+1]=float(yy[l+32]);yl[4*l+2]=float(yy[l+64]);yl[4*l+3]=float(yy[l+96]);}float4 z=0;for(short l=0;l<4;++l){z[0]+=yl[4*l]*(int((q1[l]&15)|((qh[l]&3)<<4))-32);z[1]+=yl[4*l+1]*(int((q2[l]&15)|((qh[l]&12)<<2))-32);z[2]+=yl[4*l+2]*(int((q1[l]>>4)|((qh[l]&48)<<0))-32);z[3]+=yl[4*l+3]*(int((q2[l]>>4)|((qh[l]&192)>>2))-32);}sums[row*BT+t]+=float(b.d)*(z[0]*sc[0]+z[1]*sc[2]+z[2]*sc[4]+z[3]*sc[6]);}}
    for(short r=0;r<NR&&first+r<a.rows;++r)for(short t=0;t<BT&&t0+t<a.batch;++t){float z=simd_sum(sums[r*BT+t]);if(lane==0)out[(t0+t)*a.rows+first+r]=z;}
}

template<uint TM,typename B,float value(device const B&,uint)>
inline void quant_mma(device const uchar*w0,device const float*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=32,KM=32,NT=TM/8;uint r0=group.x*RM,t0=group.y*TM;device const B*w=(device const B*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[NT];for(uint i=0;i<NT;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k0=0;k0<a.cols;k0+=KM){
        for(uint i=tid;i<RM*KM;i+=128){uint r=i/KM,k=i%KM;sa[i]=(r0+r<a.rows)?half(value(w[(r0+r)*nb+(k0+k)/256],(k0+k)%256)):half(0);}
        for(uint i=tid;i<TM*KM;i+=128){uint t=i/KM,k=i%KM;sb[i]=(t0+t<a.batch)?half(y[(t0+t)*a.cols+k0+k]):half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);for(uint kk=0;kk<KM;kk+=8){simdgroup_half8x8 ma;simdgroup_load(ma,sa+sg*8*KM+kk,KM,0,true);for(uint t=0;t<NT;++t){simdgroup_half8x8 mb;simdgroup_load(mb,sb+t*8*KM+kk,KM,0,false);simdgroup_multiply_accumulate(mc[t],mb,ma,mc[t]);}}threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint t=0;t<NT;++t)simdgroup_store(mc[t],out+(t0+t*8)*a.rows+r0+sg*8,a.rows,0,false);
}
kernel void gemma_q4k_mma(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){quant_mma<32,Q4K,q4_value>(w,x,y,a,g,tid,sg,tile);}
kernel void gemma_q6k_mma(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){quant_mma<32,Q6K,q6_value>(w,x,y,a,g,tid,sg,tile);}
kernel void gemma_q4k_mma64(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){quant_mma<64,Q4K,q4_value>(w,x,y,a,g,tid,sg,tile);}
kernel void gemma_q6k_mma64(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){quant_mma<64,Q6K,q6_value>(w,x,y,a,g,tid,sg,tile);}

template<typename B,float value(device const B&,uint)>
inline void quant_mma_rm64(device const uchar*w0,device const float*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM;device const B*w=(device const B*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k0=0;k0<a.cols;k0+=KM){
        for(uint i=tid;i<RM*KM;i+=128){uint r=i/KM,k=i%KM;sa[i]=(r0+r<a.rows)?half(value(w[(r0+r)*nb+(k0+k)/256],(k0+k)%256)):half(0);}
        for(uint i=tid;i<TM*KM;i+=128){uint t=i/KM,k=i%KM;sb[i]=(t0+t<a.batch)?half(y[(t0+t)*a.cols+k0+k]):half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);uint rb=(sg&1)*32,tb=(sg>>1)*16;for(uint kk=0;kk<KM;kk+=8){simdgroup_half8x8 ma[4],mb[2];for(uint r=0;r<4;++r)simdgroup_load(ma[r],sa+(rb+r*8)*KM+kk,KM,0,true);for(uint t=0;t<2;++t)simdgroup_load(mb[t],sb+(tb+t*8)*KM+kk,KM,0,false);for(uint t=0;t<2;++t)for(uint r=0;r<4;++r)simdgroup_multiply_accumulate(mc[t*4+r],mb[t],ma[r],mc[t*4+r]);}threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint rb=(sg&1)*32,tb=(sg>>1)*16;for(uint t=0;t<2;++t)for(uint r=0;r<4;++r)simdgroup_store(mc[t*4+r],out+(t0+tb+t*8)*a.rows+r0+rb+r*8,a.rows,0,false);
}
kernel void gemma_q4k_mma_rm64(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){quant_mma_rm64<Q4K,q4_value>(w,x,y,a,g,tid,sg,tile);}
kernel void gemma_q6k_mma_rm64(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){quant_mma_rm64<Q6K,q6_value>(w,x,y,a,g,tid,sg,tile);}

inline void q4_dequant_16(device const Q4K&b,uint g,uint l0,threadgroup half*dst){
    uchar sc,m;if(g<4){sc=b.scales[g]&63;m=b.scales[g+4]&63;}else{sc=(b.scales[g+4]&15)|((b.scales[g-4]>>6)<<4);m=(b.scales[g+4]>>4)|((b.scales[g]>>6)<<4);}float d=float(b.d)*float(sc),dm=float(b.dmin)*float(m);device const uchar*q=b.qs+(g/2)*32+l0;
    if(g&1)for(uint i=0;i<16;++i)dst[i]=half(d*float(q[i]>>4)-dm);else for(uint i=0;i<16;++i)dst[i]=half(d*float(q[i]&15)-dm);
}

template<typename Y>
inline void q4k_mma_rm64_fast(device const uchar*w0,device const Y*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k0=0;k0<a.cols;k0+=KM){uint r=tid/2,l0=(tid&1)*16;if(r0+r<a.rows){device const Q4K&b=w[(r0+r)*nb+k0/256];q4_dequant_16(b,(k0%256)/32,l0,sa+r*KM+l0);}else for(uint i=0;i<16;++i)sa[r*KM+l0+i]=half(0);
        for(uint i=tid;i<TM*KM;i+=128){uint t=i/KM,k=i%KM;sb[i]=(t0+t<a.batch)?half(y[(t0+t)*a.cols+k0+k]):half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);uint rb=(sg&1)*32,tb=(sg>>1)*16;for(uint kk=0;kk<KM;kk+=8){simdgroup_half8x8 ma[4],mb[2];for(uint rr=0;rr<4;++rr)simdgroup_load(ma[rr],sa+(rb+rr*8)*KM+kk,KM,0,true);for(uint t=0;t<2;++t)simdgroup_load(mb[t],sb+(tb+t*8)*KM+kk,KM,0,false);for(uint t=0;t<2;++t)for(uint rr=0;rr<4;++rr)simdgroup_multiply_accumulate(mc[t*4+rr],mb[t],ma[rr],mc[t*4+rr]);}threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint rb=(sg&1)*32,tb=(sg>>1)*16;for(uint t=0;t<2;++t)for(uint r=0;r<4;++r)simdgroup_store(mc[t*4+r],out+(t0+tb+t*8)*a.rows+r0+rb+r*8,a.rows,0,false);
}
kernel void gemma_q4k_mma_rm64_fast(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q4k_mma_rm64_fast<float>(w,x,y,a,g,tid,sg,tile);}
kernel void gemma_q4k_mma_rm64_fast_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q4k_mma_rm64_fast<half>(w,x,y,a,g,tid,sg,tile);}

inline void q4_dequant_16_thread(device const Q4K&b,uint g,uint l0,thread half*dst){
    uchar sc,m;if(g<4){sc=b.scales[g]&63;m=b.scales[g+4]&63;}else{sc=(b.scales[g+4]&15)|((b.scales[g-4]>>6)<<4);m=(b.scales[g+4]>>4)|((b.scales[g]>>6)<<4);}float d=float(b.d)*float(sc),dm=float(b.dmin)*float(m);uint4 packed=*(device const uint4*)(b.qs+(g/2)*32+l0);
    if(g&1)for(uint i=0;i<16;++i){uchar q=uchar(packed[i>>2]>>((i&3)*8));dst[i]=half(d*float(q>>4)-dm);}else for(uint i=0;i<16;++i){uchar q=uchar(packed[i>>2]>>((i&3)*8));dst[i]=half(d*float(q&15)-dm);}
}

inline void q4_dequant_32_pair_thread(device const Q4K&b,uint g,uint l0,thread half*dst){
    uint hi=l0>>4,gj=g+hi;uchar sc,m;if(gj<4){sc=b.scales[gj]&63;m=b.scales[gj+4]&63;}else{sc=(b.scales[gj+4]&15)|((b.scales[gj-4]>>6)<<4);m=(b.scales[gj+4]>>4)|((b.scales[gj]>>6)<<4);}float2 own=float2(float(b.d)*float(sc),float(b.dmin)*float(m)),peer=float2(simd_shuffle_xor(own.x,1),simd_shuffle_xor(own.y,1)),dm0=hi?peer:own,dm1=hi?own:peer;uint4 packed=*(device const uint4*)(b.qs+(g/2)*32+l0);
    FOR_UNROLL(uint i=0;i<16;++i){uchar q=uchar(packed[i>>2]>>((i&3)*8));dst[i]=half(dm0.x*float(q&15)-dm0.y);dst[16+i]=half(dm1.x*float(q>>4)-dm1.y);}
}

inline void load8_to_tile(device const half*src,threadgroup half*dst){*(threadgroup half2x4*)dst=*(device const half2x4*)src;}
inline void load8_to_tile(device const float*src,threadgroup half*dst){for(uint i=0;i<8;++i)dst[i]=half(src[i]);}
inline void q4_dequant_16_scaled(device const Q4K&b,half2 scale,uint g,uint l0,thread half*dst){float d=float(scale.x),dm=float(scale.y);uint4 packed=*(device const uint4*)(b.qs+(g/2)*32+l0);if(g&1)FOR_UNROLL(uint i=0;i<16;++i){uchar q=uchar(packed[i>>2]>>((i&3)*8));dst[i]=half(d*float(q>>4)-dm);}else FOR_UNROLL(uint i=0;i<16;++i){uchar q=uchar(packed[i>>2]>>((i&3)*8));dst[i]=half(d*float(q&15)-dm);}}
inline void q4_dequant_32_pair_scaled(device const Q4K&b,half2 scale0,half2 scale1,uint g,uint l0,thread half*dst){float2 d=float2(scale0.x,scale1.x),dm=float2(scale0.y,scale1.y);uint4 packed=*(device const uint4*)(b.qs+(g/2)*32+l0);FOR_UNROLL(uint i=0;i<16;++i){uchar q=uchar(packed[i>>2]>>((i&3)*8));dst[i]=half(d.x*float(q&15)-dm.x);dst[16+i]=half(d.y*float(q>>4)-dm.y);}}

kernel void gemma_q4k_expand_scales(device const uchar*w0[[buffer(0)]],device half2*out[[buffer(1)]],constant MatArgs&a[[buffer(2)]],uint i[[thread_position_in_grid]]){uint groups=a.cols*a.rows/32;if(i>=groups)return;uint block=i/8,g=i%8;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);device const Q4K&b=w[block];uchar sc,m;if(g<4){sc=b.scales[g]&63;m=b.scales[g+4]&63;}else{sc=(b.scales[g+4]&15)|((b.scales[g-4]>>6)<<4);m=(b.scales[g+4]>>4)|((b.scales[g]>>6)<<4);}out[i]=half2(float(b.d)*float(sc),float(b.dmin)*float(m));}
kernel void gemma_q4k_expand_scales_metal(device const uchar*w0[[buffer(0)]],device half2*out[[buffer(1)]],constant MatArgs&a[[buffer(2)]],uint i[[thread_position_in_grid]]){uint kg=a.cols/32,total=a.rows*kg;if(i>=total)return;uint row=i/kg,kgroup=i%kg,block=kgroup/8,g=kgroup%8,nb=a.cols/256;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);device const Q4K&b=w[((row/64)*nb+block)*64+row%64];uchar sc,m;if(g<4){sc=b.scales[g]&63;m=b.scales[g+4]&63;}else{sc=(b.scales[g+4]&15)|((b.scales[g-4]>>6)<<4);m=(b.scales[g+4]>>4)|((b.scales[g]>>6)<<4);}out[i]=half2(float(b.d)*float(sc),float(b.dmin)*float(m));}

// Reorder one FFN matrix into K-major 64-row slabs. The compact 20-byte group
// makes adjacent SIMD lanes fetch adjacent rows instead of jumping by an entire
// 3840-wide GGUF row on every K tile.
kernel void gemma_q4k_pack_rm64(device const uchar*w0[[buffer(0)]],device PackedQ4Group*out[[buffer(1)]],constant MatArgs&a[[buffer(2)]],uint i[[thread_position_in_grid]]){
    uint kg=a.cols/32,total=a.rows*kg;if(i>=total)return;uint row=i/kg,g=i%kg,block=g/8,sub=g%8;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);device const Q4K&b=w[row*(a.cols/256)+block];uchar sc,m;if(sub<4){sc=b.scales[sub]&63;m=b.scales[sub+4]&63;}else{sc=(b.scales[sub+4]&15)|((b.scales[sub-4]>>6)<<4);m=(b.scales[sub+4]>>4)|((b.scales[sub]>>6)<<4);}uint dst=((row/64)*kg+g)*64+row%64;out[dst].dm=half2(float(b.d)*float(sc),float(b.dmin)*float(m));device const uchar*q=b.qs+(sub/2)*32;for(uint j=0;j<16;++j){uchar q0=q[2*j],q1=q[2*j+1];if(sub&1){q0>>=4;q1>>=4;}else{q0&=15;q1&=15;}out[dst].qs[j]=q0|(q1<<4);}
}
inline void packed_q4_dequant_16(device const PackedQ4Group&b,uint l0,thread half*dst){float d=float(b.dm.x),dm=float(b.dm.y);for(uint i=0;i<16;++i){uint at=l0+i;uchar q=b.qs[at/2];q=(at&1)?q>>4:q&15;dst[i]=half(d*float(q)-dm);}}

// 64x32 swizzled tile: four simdgroups own 32x16 output quadrants.  The
// threadgroup layout is compatible with contiguous, non-transposed matrix loads.
template<typename Y>
inline void q4k_mma_swizzled(device const uchar*w0,device const Y*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);device const Q4K*wb=w+(r0+row)*nb;uint slice=il0;
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[16];q4_dequant_16_thread(*wb,slice>>1,(slice&1)*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;*(sa+64*ib+8*ly+lx)=tmp[i];}
        uint sx=tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const Y*src=y+(t0+trow)*a.cols+k0+iy;load8_to_tile(src,bdst);}else for(uint i=0;i<8;++i)bdst[i]=half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}slice+=2;if(slice>=16){slice=il0;++wb;}
    }
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}
kernel void gemma_q4k_mma_swizzled(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q4k_mma_swizzled<float>(w,x,y,a,g,tid,sg,tile);}
kernel void gemma_q4k_mma_swizzled_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q4k_mma_swizzled<half>(w,x,y,a,g,tid,sg,tile);}

// FFN down projection: eight simdgroups cover 128 output rows while sharing
// the same activation tile.  This halves the dominant reread of the 15360-wide
// GEGLU intermediate compared with the general 64-row projection.
inline void q4k_mma_swizzled_rm128_f16(device const uchar*w0,device const half*x,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=128,TM=16,KM=32;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);uint row=tid;
    for(uint k0=0;k0<a.cols;k0+=KM){uint block=k0/256,g=(k0%256)/32;half tmp[32];q4_dequant_16_thread(w[(r0+row)*nb+block],g,0,tmp);q4_dequant_16_thread(w[(r0+row)*nb+block],g,16,tmp+16);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=2*j+i/8,sy=row/8,lx=row%8,ly=i%8,ib=16*sx+sy;sa[64*ib+8*ly+lx]=tmp[16*j+i];}if(tid<64){uint trow=tid/4,iy=8*(tid&3),sx=tid%4,sy=trow/8,ly=trow%8,ib=2*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*sg;threadgroup const half*pb=sb;for(uint ik=0;ik<4;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=16*64;pb+=2*64;}}
    device float*C=out+r0+32*sg+t0*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}
kernel void gemma_q4k_mma_swizzled_rm128_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q4k_mma_swizzled_rm128_f16(w,x,y,a,g,tid,sg,tile);}

// A 64-token tile halves repeated weight dequantization at large prefill
// batches. Eight simdgroups retain the same 32x16 output ownership as the
// 32-token kernel, avoiding the register spill of a wider per-simdgroup tile.
template<typename Y>
inline void q4k_mma_swizzled_tm64(device const uchar*w0,device const Y*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=64,KM=32;uint r0=group.x*RM,t0=group.y*TM;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    const uint trow=tid/4,iy=8*(tid&3);for(uint k0=0;k0<a.cols;k0+=KM){half tmp[16];uint row=tid/2,il0=tid&1;if(tid<128)q4_dequant_16_thread(w[(r0+row)*nb+k0/256],(k0%256)/32,il0*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<128)for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[i];}
        uint sx=tid%4,sy=trow/8,ly=trow%8,ib=8*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const Y*src=y+(t0+trow)*a.cols+k0+iy;load8_to_tile(src,bdst);}else for(uint i=0;i<8;++i)bdst[i]=half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=8*64;}
    }
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}
kernel void gemma_q4k_mma_swizzled_tm64_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q4k_mma_swizzled_tm64<half>(w,x,y,a,g,tid,sg,tile);}

template<typename Y>
inline void q4k_mma_swizzled_k64(device const uchar*w0,device const Y*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];uint block=k0/256,g=(k0%256)/32;q4_dequant_32_pair_thread(w[(r0+row)*nb+block],g,il0*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[16*j+i];}
        for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const Y*src=y+(t0+trow)*a.cols+k0+32*j+iy;load8_to_tile(src,bdst);}else for(uint i=0;i<8;++i)bdst[i]=half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}
    }
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}
kernel void gemma_q4k_mma_swizzled_k64_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q4k_mma_swizzled_k64<half>(w,x,y,a,g,tid,sg,tile);}

kernel void gemma_q4k_mma_scaled_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],device const half2*scales[[buffer(4)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);device const Q4K*wb=w+(r0+row)*nb;device const half2*ss=scales+(r0+row)*nb*8;uint slice=il0;
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[16];q4_dequant_16_scaled(*wb,ss[slice>>1],slice>>1,(slice&1)*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[i];}uint sx=tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const half*src=x+(t0+trow)*a.cols+k0+iy;*(threadgroup half2x4*)bdst=*(device const half2x4*)src;}else for(uint i=0;i<8;++i)bdst[i]=half(0);threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}slice+=2;if(slice>=16){slice=il0;++wb;ss+=8;}}
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}

kernel void gemma_q4k_gate_up_geglu_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_float8x8 cg[8],cu[8];for(uint i=0;i<8;++i){cg[i]=make_filled_simdgroup_matrix<float,8>(0.0f);cu[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);device const Q4K*wgb=wg+(r0+row)*nb;device const Q4K*wub=wu+(r0+row)*nb;uint slice=il0;
    for(uint k0=0;k0<a.cols;k0+=KM){half tg[16],tu[16];q4_dequant_16_thread(*wgb,slice>>1,(slice&1)*16,tg);q4_dequant_16_thread(*wub,slice>>1,(slice&1)*16,tu);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;uint at=64*ib+8*ly+lx;sgate[at]=tg[i];sup[at]=tu[i];}uint sx=tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const half*src=x+(t0+trow)*a.cols+k0+iy;*(threadgroup half2x4*)bdst=*(device const half2x4*)src;}else for(uint i=0;i<8;++i)bdst[i]=half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pg=sgate+4*64*(sg%2);threadgroup const half*pu=sup+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 mg[4],mu[4],mb[2];for(uint i=0;i<4;++i){simdgroup_load(mg[i],pg+64*i,8,0,false);simdgroup_load(mu[i],pu+64*i,8,0,false);}for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i){simdgroup_multiply_accumulate(cg[i],mb[i/4],mg[i%4],cg[i]);simdgroup_multiply_accumulate(cu[i],mb[i/4],mu[i%4],cu[i]);}pg+=8*64;pu+=8*64;pb+=4*64;}slice+=2;if(slice>=16){slice=il0;++wgb;++wub;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*og=(threadgroup float*)tile;threadgroup float*ou=og+RM*TM;uint rb=32*(sg&1),tb=16*(sg>>1);for(uint i=0;i<8;++i){uint at=rb+tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(cg[i],og+at,RM,0,false);simdgroup_store(cu[i],ou+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float g=og[i],u=ou[i];float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[(t0+t)*a.rows+r0+r]=half(gelu*u);}}
}

kernel void gemma_q4k_gate_up_geglu_scaled_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],device const half2*scale_g[[buffer(4)]],device const half2*scale_u[[buffer(5)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_float8x8 cg[8],cu[8];for(uint i=0;i<8;++i){cg[i]=make_filled_simdgroup_matrix<float,8>(0.0f);cu[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);device const Q4K*wgb=wg+(r0+row)*nb;device const Q4K*wub=wu+(r0+row)*nb;device const half2*ssg=scale_g+(r0+row)*nb*8;device const half2*ssu=scale_u+(r0+row)*nb*8;uint slice=il0;
    for(uint k0=0;k0<a.cols;k0+=KM){half tg[16],tu[16];uint g=slice>>1,l0=(slice&1)*16;q4_dequant_16_scaled(*wgb,ssg[g],g,l0,tg);q4_dequant_16_scaled(*wub,ssu[g],g,l0,tu);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy,at=64*ib+8*ly+lx;sgate[at]=tg[i];sup[at]=tu[i];}uint sx=tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const half*src=x+(t0+trow)*a.cols+k0+iy;*(threadgroup half2x4*)bdst=*(device const half2x4*)src;}else for(uint i=0;i<8;++i)bdst[i]=half(0);threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pg=sgate+4*64*(sg%2);threadgroup const half*pu=sup+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 mg[4],mu[4],mb[2];for(uint i=0;i<4;++i){simdgroup_load(mg[i],pg+64*i,8,0,false);simdgroup_load(mu[i],pu+64*i,8,0,false);}for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i){simdgroup_multiply_accumulate(cg[i],mb[i/4],mg[i%4],cg[i]);simdgroup_multiply_accumulate(cu[i],mb[i/4],mu[i%4],cu[i]);}pg+=8*64;pu+=8*64;pb+=4*64;}slice+=2;if(slice>=16){slice=il0;++wgb;++wub;ssg+=8;ssu+=8;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*og=(threadgroup float*)tile;threadgroup float*ou=og+RM*TM;uint rb=32*(sg&1),tb=16*(sg>>1);for(uint i=0;i<8;++i){uint at=rb+tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(cg[i],og+at,RM,0,false);simdgroup_store(cu[i],ou+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float g=og[i],u=ou[i];float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[(t0+t)*a.rows+r0+r]=half(gelu*u);}}
}

kernel void gemma_q4k_gate_up_geglu_packed_f16(device const uchar*p0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM,kg=a.cols/32;device const PackedQ4Group*wg=(device const PackedQ4Group*)(p0+a.weight0_offset);device const PackedQ4Group*wu=(device const PackedQ4Group*)(p0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_float8x8 cg[8],cu[8];for(uint i=0;i<8;++i){cg[i]=make_filled_simdgroup_matrix<float,8>(0.0f);cu[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}uint row=tid/2,il0=tid&1;
    for(uint k0=0;k0<a.cols;k0+=KM){uint pi=(group.x*kg+k0/32)*64+row;half tg[16],tu[16];packed_q4_dequant_16(wg[pi],il0*16,tg);packed_q4_dequant_16(wu[pi],il0*16,tu);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy,at=64*ib+8*ly+lx;sgate[at]=tg[i];sup[at]=tu[i];}uint trow=tid/4,iy=8*(tid&3),sx=tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pg=sgate+4*64*(sg%2);threadgroup const half*pu=sup+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 mg[4],mu[4],mb[2];for(uint i=0;i<4;++i){simdgroup_load(mg[i],pg+64*i,8,0,false);simdgroup_load(mu[i],pu+64*i,8,0,false);}for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i){simdgroup_multiply_accumulate(cg[i],mb[i/4],mg[i%4],cg[i]);simdgroup_multiply_accumulate(cu[i],mb[i/4],mu[i%4],cu[i]);}pg+=8*64;pu+=8*64;pb+=4*64;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*og=(threadgroup float*)tile;threadgroup float*ou=og+RM*TM;uint rb=32*(sg&1),tb=16*(sg>>1);for(uint i=0;i<8;++i){uint at=rb+tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(cg[i],og+at,RM,0,false);simdgroup_store(cu[i],ou+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float g=og[i],u=ou[i];float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[(t0+t)*a.rows+r0+r]=half(gelu*u);}}
}

// 128-row gate/up tile. The two projections retain their accumulators together,
// and two 64-row output phases reuse the dequantization storage for GEGLU.
kernel void gemma_q4k_gate_up_geglu_scaled_rm128_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],device const half2*scale_g[[buffer(4)]],device const half2*scale_u[[buffer(5)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=128,TM=16,KM=32;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_float8x8 cg[8],cu[8];for(uint i=0;i<8;++i){cg[i]=make_filled_simdgroup_matrix<float,8>(0.0f);cu[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}uint row=tid,wr=r0+row;
    for(uint k0=0;k0<a.cols;k0+=KM){uint block=k0/256,g=(k0%256)/32;half tg[32],tu[32];half2 ssg=scale_g[(wr*nb+block)*8+g],ssu=scale_u[(wr*nb+block)*8+g];q4_dequant_16_scaled(wg[wr*nb+block],ssg,g,0,tg);q4_dequant_16_scaled(wg[wr*nb+block],ssg,g,16,tg+16);q4_dequant_16_scaled(wu[wr*nb+block],ssu,g,0,tu);q4_dequant_16_scaled(wu[wr*nb+block],ssu,g,16,tu+16);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=2*j+i/8,sy=row/8,lx=row%8,ly=i%8,ib=16*sx+sy,at=64*ib+8*ly+lx;sgate[at]=tg[16*j+i];sup[at]=tu[16*j+i];}if(tid<64){uint trow=tid/4,iy=8*(tid&3),sx=tid%4,sy=trow/8,ly=trow%8,ib=2*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pg=sgate+4*64*sg;threadgroup const half*pu=sup+4*64*sg;threadgroup const half*pb=sb;for(uint ik=0;ik<4;++ik){simdgroup_half8x8 mg[4],mu[4],mb[2];for(uint i=0;i<4;++i){simdgroup_load(mg[i],pg+64*i,8,0,false);simdgroup_load(mu[i],pu+64*i,8,0,false);}for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i){simdgroup_multiply_accumulate(cg[i],mb[i/4],mg[i%4],cg[i]);simdgroup_multiply_accumulate(cu[i],mb[i/4],mu[i%4],cu[i]);}pg+=16*64;pu+=16*64;pb+=2*64;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*og=(threadgroup float*)tile;threadgroup float*ou=og+RM*TM;uint rb=32*sg;for(uint i=0;i<8;++i){uint at=rb+8*(i%4)+8*RM*(i/4);simdgroup_store(cg[i],og+at,RM,0,false);simdgroup_store(cu[i],ou+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float g=og[i],u=ou[i];float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[(t0+t)*a.rows+r0+r]=half(gelu*u);}}
}

kernel void gemma_q4k_gate_up_geglu_scaled_k64_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],device const half2*scale_g[[buffer(4)]],device const half2*scale_u[[buffer(5)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_float8x8 cg[8],cu[8];for(uint i=0;i<8;++i){cg[i]=make_filled_simdgroup_matrix<float,8>(0.0f);cu[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3),wr=r0+row;
    for(uint k0=0;k0<a.cols;k0+=KM){half tg[32],tu[32];uint block=k0/256,g=(k0%256)/32;device const Q4K&bg=wg[wr*nb+block];device const Q4K&bu=wu[wr*nb+block];q4_dequant_32_pair_scaled(bg,scale_g[(wr*nb+block)*8+g],scale_g[(wr*nb+block)*8+g+1],g,il0*16,tg);q4_dequant_32_pair_scaled(bu,scale_u[(wr*nb+block)*8+g],scale_u[(wr*nb+block)*8+g+1],g,il0*16,tu);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy,at=64*ib+8*ly+lx;sgate[at]=tg[16*j+i];sup[at]=tu[16*j+i];}
        for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const half*src=x+(t0+trow)*a.cols+k0+32*j+iy;load8_to_tile(src,bdst);}else for(uint i=0;i<8;++i)bdst[i]=half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pg=sgate+4*64*(sg%2);threadgroup const half*pu=sup+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 mg[4],mu[4],mb[2];for(uint i=0;i<4;++i){simdgroup_load(mg[i],pg+64*i,8,0,false);simdgroup_load(mu[i],pu+64*i,8,0,false);}for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i){simdgroup_multiply_accumulate(cg[i],mb[i/4],mg[i%4],cg[i]);simdgroup_multiply_accumulate(cu[i],mb[i/4],mu[i%4],cu[i]);}pg+=8*64;pu+=8*64;pb+=4*64;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*og=(threadgroup float*)tile;threadgroup float*ou=og+RM*TM;uint rb=32*(sg&1),tb=16*(sg>>1);for(uint i=0;i<8;++i){uint at=rb+tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(cg[i],og+at,RM,0,false);simdgroup_store(cu[i],ou+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float g=og[i],u=ou[i];float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[(t0+t)*a.rows+r0+r]=half(gelu*u);}}
}

// Disk-persistent Metal FFN format: native Q4_K blocks are reordered as
// [64-row slab][K block][row].  No values are requantized and no scale sidecar
// is needed; adjacent SIMD lanes fetch adjacent quantized blocks.
kernel void gemma_q4k_gate_up_geglu_metal_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],device const half2*scale_g[[buffer(4)]],device const half2*scale_u[[buffer(5)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_float8x8 cg[8],cu[8];for(uint i=0;i<8;++i){cg[i]=make_filled_simdgroup_matrix<float,8>(0.0f);cu[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);
    for(uint k0=0;k0<a.cols;k0+=KM){half tg[32],tu[32];uint block=k0/256,g=(k0%256)/32,at=((r0/RM)*nb+block)*RM+row,si=((r0+row)*nb+block)*8+g;q4_dequant_32_pair_scaled(wg[at],scale_g[si],scale_g[si+1],g,il0*16,tg);q4_dequant_32_pair_scaled(wu[at],scale_u[si],scale_u[si+1],g,il0*16,tu);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy,p=64*ib+8*ly+lx;sgate[p]=tg[16*j+i];sup[p]=tu[16*j+i];}
        for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+32*j+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pg=sgate+4*64*(sg%2);threadgroup const half*pu=sup+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 mg[4],mu[4],mb[2];for(uint i=0;i<4;++i){simdgroup_load(mg[i],pg+64*i,8,0,false);simdgroup_load(mu[i],pu+64*i,8,0,false);}for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i){simdgroup_multiply_accumulate(cg[i],mb[i/4],mg[i%4],cg[i]);simdgroup_multiply_accumulate(cu[i],mb[i/4],mu[i%4],cu[i]);}pg+=8*64;pu+=8*64;pb+=4*64;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*og=(threadgroup float*)tile;threadgroup float*ou=og+RM*TM;uint rb=32*(sg&1),tb=16*(sg>>1);for(uint i=0;i<8;++i){uint at=rb+tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(cg[i],og+at,RM,0,false);simdgroup_store(cu[i],ou+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float g=og[i],u=ou[i];float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[(t0+t)*a.rows+r0+r]=half(gelu*u);}}
}

// Eight SIMD groups split gate and up ownership. Each group retains only one
// FP32 accumulator bank, while both projections share the same activation tile
// and are joined in threadgroup memory before the FP16 GEGLU epilogue.
kernel void gemma_q4k_gate_up_geglu_metal_split_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);uint proj=tid>>7,ltid=tid&127,row=ltid/2,il0=ltid&1,local_sg=sg&3;
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];uint block=k0/256,g=(k0%256)/32,at=((r0/RM)*nb+block)*RM+row;device const Q4K*w=proj?wu:wg;q4_dequant_32_pair_thread(w[at],g,il0*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup half*sw=proj?sup:sgate;for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sw[64*ib+8*ly+lx]=tmp[16*j+i];}if(tid<128){uint trow=ltid/4,iy=8*(ltid&3);for(uint j=0;j<2;++j){uint sx=4*j+ltid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+32*j+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=(sg<4?sgate:sup)+4*64*(local_sg&1);threadgroup const half*pb=sb+2*64*(local_sg>>1);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*og=(threadgroup float*)tile;threadgroup float*ou=og+RM*TM;uint rb=32*(local_sg&1),tb=16*(local_sg>>1);threadgroup float*dst=sg<4?og:ou;for(uint i=0;i<8;++i){uint at=rb+tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(mc[i],dst+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=256){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float g=og[i],u=ou[i];float gelu=0.5f*g*(1.0f+precise::tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)));out[(t0+t)*a.rows+r0+r]=half(gelu*u);}}
}

// Two SIMD groups compute gate and two compute up for the same 32x32 tile.
// This preserves the current 128-thread/K64 geometry while fusing the global
// FP32 gate/up stores, reloads and GEGLU dispatch into one threadgroup.
kernel void gemma_q4k_gate_up_geglu_metal_rm32_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=32,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);uint proj=tid>>6,ltid=tid&63,row=ltid/2,il0=ltid&1,local_sg=sg&1,wr=r0+row;
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];uint block=k0/256,g=(k0%256)/32,at=((wr/64)*nb+block)*64+wr%64;device const Q4K*w=proj?wu:wg;q4_dequant_32_pair_thread(w[at],g,il0*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup half*sw=proj?sup:sgate;for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=4*sx+sy;sw[64*ib+8*ly+lx]=tmp[16*j+i];}uint trow=tid/4,iy=8*(tid&3);for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+32*j+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=(sg<2?sgate:sup);threadgroup const half*pb=sb+2*64*local_sg;for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=4*64;pb+=4*64;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*og=(threadgroup float*)tile;threadgroup float*ou=og+RM*TM;threadgroup float*dst=sg<2?og:ou;uint tb=16*local_sg;for(uint i=0;i<8;++i){uint at=tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(mc[i],dst+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float gv=og[i],uv=ou[i];float gelu=0.5f*gv*(1.0f+precise::tanh(0.7978845608028654f*gv*(1.0f+0.044715f*gv*gv)));out[(t0+t)*a.rows+r0+r]=half(gelu*uv);}}
}

kernel void gemma_q4k_gate_up_geglu_metal_rm32_f16acc_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device half*out[[buffer(2)]],constant PairMatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=32,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*wg=(device const Q4K*)(w0+a.weight0_offset);device const Q4K*wu=(device const Q4K*)(w0+a.weight1_offset);threadgroup half*sgate=tile;threadgroup half*sup=tile+RM*KM;threadgroup half*sb=tile+2*RM*KM;simdgroup_half8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<half,8>(half(0));uint proj=tid>>6,ltid=tid&63,row=ltid/2,il0=ltid&1,local_sg=sg&1,wr=r0+row;
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];uint block=k0/256,g=(k0%256)/32,at=((wr/64)*nb+block)*64+wr%64;device const Q4K*w=proj?wu:wg;q4_dequant_32_pair_thread(w[at],g,il0*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup half*sw=proj?sup:sgate;for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=4*sx+sy;sw[64*ib+8*ly+lx]=tmp[16*j+i];}uint trow=tid/4,iy=8*(tid&3);for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+32*j+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=(sg<2?sgate:sup);threadgroup const half*pb=sb+2*64*local_sg;for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=4*64;pb+=4*64;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup half*og=tile;threadgroup half*ou=og+RM*TM;threadgroup half*dst=sg<2?og:ou;uint tb=16*local_sg;for(uint i=0;i<8;++i){uint at=tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(mc[i],dst+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch){float gv=float(og[i]),uv=float(ou[i]);float gelu=0.5f*gv*(1.0f+precise::tanh(0.7978845608028654f*gv*(1.0f+0.044715f*gv*gv)));out[(t0+t)*a.rows+r0+r]=half(gelu*uv);}}
}

kernel void gemma_q4k_mma_metal_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];uint block=k0/256,g=(k0%256)/32,at=((r0/RM)*nb+block)*RM+row;q4_dequant_32_pair_thread(w[at],g,il0*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[16*j+i];}for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+32*j+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}}
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}

kernel void gemma_q4k_mma_metal_f16acc_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_half8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<half,8>(half(0));const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];uint block=k0/256,g=(k0%256)/32,at=((r0/RM)*nb+block)*RM+row;q4_dequant_32_pair_thread(w[at],g,il0*16,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[16*j+i];}for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+32*j+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup half*oc=tile;uint rb=32*(sg&1),tb=16*(sg>>1);for(uint i=0;i<8;++i){uint at=rb+tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(mc[i],oc+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch)out[(t0+t)*a.rows+r0+r]=float(oc[i]);}
}

inline void q6_dequant_16(device const Q6K&b,uint q,uint l0,threadgroup half*dst){
    uint h=q/4;q%=4;device const uchar*ql=b.ql+h*64,*qh=b.qh+h*32;float d=float(b.d)*float(b.scales[h*8+l0/16+q*2]);for(uint i=0;i<16;++i){uint l=l0+i,v;if(q==0)v=(ql[l]&15)|(((qh[l]>>0)&3)<<4);else if(q==1)v=(ql[l+32]&15)|(((qh[l]>>2)&3)<<4);else if(q==2)v=(ql[l]>>4)|(((qh[l]>>4)&3)<<4);else v=(ql[l+32]>>4)|(((qh[l]>>6)&3)<<4);dst[i]=half(d*float(int(v)-32));}
}

template<typename Y>
inline void q6k_mma_rm64_fast(device const uchar*w0,device const Y*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k0=0;k0<a.cols;k0+=KM){uint r=tid/2,l0=(tid&1)*16;if(r0+r<a.rows){device const Q6K&b=w[(r0+r)*nb+k0/256];q6_dequant_16(b,(k0%256)/32,l0,sa+r*KM+l0);}else for(uint i=0;i<16;++i)sa[r*KM+l0+i]=half(0);
        for(uint i=tid;i<TM*KM;i+=128){uint t=i/KM,k=i%KM;sb[i]=(t0+t<a.batch)?half(y[(t0+t)*a.cols+k0+k]):half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);uint rb=(sg&1)*32,tb=(sg>>1)*16;for(uint kk=0;kk<KM;kk+=8){simdgroup_half8x8 ma[4],mb[2];for(uint rr=0;rr<4;++rr)simdgroup_load(ma[rr],sa+(rb+rr*8)*KM+kk,KM,0,true);for(uint t=0;t<2;++t)simdgroup_load(mb[t],sb+(tb+t*8)*KM+kk,KM,0,false);for(uint t=0;t<2;++t)for(uint rr=0;rr<4;++rr)simdgroup_multiply_accumulate(mc[t*4+rr],mb[t],ma[rr],mc[t*4+rr]);}threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint rb=(sg&1)*32,tb=(sg>>1)*16;for(uint t=0;t<2;++t)for(uint r=0;r<4;++r)simdgroup_store(mc[t*4+r],out+(t0+tb+t*8)*a.rows+r0+rb+r*8,a.rows,0,false);
}
kernel void gemma_q6k_mma_rm64_fast(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q6k_mma_rm64_fast<float>(w,x,y,a,g,tid,sg,tile);}
kernel void gemma_q6k_mma_rm64_fast_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q6k_mma_rm64_fast<half>(w,x,y,a,g,tid,sg,tile);}

// Decode four Q6_K values per packed uint operation.  Keeping the result in
// registers lets the leading barrier of the next K tile guard shared-memory reuse.
inline void q6_dequant_16_thread(device const Q6K&b,uint il,thread half*dst){
    device const ushort*ql=(device const ushort*)b.ql;device const ushort*qh=(device const ushort*)b.qh;ql+=32*(il/8)+16*((il/2)&1)+8*(il&1);qh+=16*(il/8)+8*(il&1);float sc=float(b.scales[(il%2)+2*(il/2)]),d=float(b.d);il=(il/2)&3;
    uint km1=il>1?(il>2?0xC0C0C0C0:0x30303030):(il>0?0x0C0C0C0C:0x03030303),km2=il>1?0xF0F0F0F0:0x0F0F0F0F;uint shr_h=il>2?2:0,shl_h=il>1?0:(il>0?2:4),shr_l=il>1?4:0;float ml=d*sc*32.0f,dl0=d*sc,dl1=dl0/256.0f,dl2=dl1/256.0f,dl3=dl2/256.0f;
    for(uint i=0;i<4;++i){uint low=(uint(ql[2*i])|(uint(ql[2*i+1])<<16))&km2,high=(uint(qh[2*i])|(uint(qh[2*i+1])<<16))&km1,q=((high<<shl_h)>>shr_h)|(low>>shr_l);dst[4*i]=half(dl0*float(q&0xFF)-ml);dst[4*i+1]=half(dl1*float(q&0xFF00)-ml);dst[4*i+2]=half(dl2*float(q&0xFF0000)-ml);dst[4*i+3]=half(dl3*float(q&0xFF000000)-ml);}
}

template<typename Y>
inline void q6k_mma_swizzled(device const uchar*w0,device const Y*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=32,KM=32;uint r0=group.x*RM,t0=group.y*TM;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[16];q6_dequant_16_thread(w[(r0+row)*nb+k0/256],2*((k0%256)/32)+il0,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;*(sa+64*ib+8*ly+lx)=tmp[i];}
        uint sx=tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const Y*src=y+(t0+trow)*a.cols+k0+iy;for(uint i=0;i<8;++i)bdst[i]=half(src[i]);}else for(uint i=0;i<8;++i)bdst[i]=half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}
    }
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}
kernel void gemma_q6k_mma_swizzled(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q6k_mma_swizzled<float>(w,x,y,a,g,tid,sg,tile);}
kernel void gemma_q6k_mma_swizzled_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q6k_mma_swizzled<half>(w,x,y,a,g,tid,sg,tile);}

inline void q6k_mma_swizzled_rm128_f16(device const uchar*w0,device const half*x,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=128,TM=16,KM=32;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);uint row=tid;
    for(uint k0=0;k0<a.cols;k0+=KM){uint block=k0/256,g=(k0%256)/32;half tmp[32];q6_dequant_16_thread(w[(r0+row)*nb+block],2*g,tmp);q6_dequant_16_thread(w[(r0+row)*nb+block],2*g+1,tmp+16);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=2*j+i/8,sy=row/8,lx=row%8,ly=i%8,ib=16*sx+sy;sa[64*ib+8*ly+lx]=tmp[16*j+i];}if(tid<64){uint trow=tid/4,iy=8*(tid&3),sx=tid%4,sy=trow/8,ly=trow%8,ib=2*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*sg;threadgroup const half*pb=sb;for(uint ik=0;ik<4;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=16*64;pb+=2*64;}}
    device float*C=out+r0+32*sg+t0*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}
kernel void gemma_q6k_mma_swizzled_rm128_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q6k_mma_swizzled_rm128_f16(w,x,y,a,g,tid,sg,tile);}

// Q, K and V share each 16x32 activation tile. K/V are only evaluated for the
// leading kv_rows, while Q continues through its larger head matrix. Post-
// projection Q normalization/RoPE and K/V normalization/RoPE/cache storage stay
// fused in their existing two dispatches because head RMSNorm needs all row tiles.
template<bool V_Q6,bool TIED_V>
inline void qkv_mma_swizzled_f16(device const uchar*w0,device const half*x,device float*qout,device float*kout,device float*vout,constant QKVArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=16,KM=32;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;bool have_kv=r0<a.kv_rows;device const Q4K*wq=(device const Q4K*)(w0+a.q_offset);device const Q4K*wk=(device const Q4K*)(w0+a.k_offset);device const Q4K*wv4=(device const Q4K*)(w0+a.v_offset);device const Q6K*wv6=(device const Q6K*)(w0+a.v_offset);threadgroup half*tq=tile;threadgroup half*tk=tq+RM*KM;threadgroup half*tv=tk+RM*KM;threadgroup half*tx=tv+RM*KM;simdgroup_float8x8 cq[4],ck[4],cv[4];for(uint i=0;i<4;++i){cq[i]=make_filled_simdgroup_matrix<float,8>(0.0f);ck[i]=make_filled_simdgroup_matrix<float,8>(0.0f);cv[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}uint row=tid/2,il0=tid&1;
    for(uint k0=0;k0<a.cols;k0+=KM){uint block=k0/256,g=(k0%256)/32;half vq[16],vk[16],vv[16];q4_dequant_16_thread(wq[(r0+row)*nb+block],g,il0*16,vq);if(have_kv){q4_dequant_16_thread(wk[(r0+row)*nb+block],g,il0*16,vk);if(!TIED_V){if(V_Q6)q6_dequant_16_thread(wv6[(r0+row)*nb+block],2*g+il0,vv);else q4_dequant_16_thread(wv4[(r0+row)*nb+block],g,il0*16,vv);}}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy,at=64*ib+8*ly+lx;tq[at]=vq[i];if(have_kv){tk[at]=vk[i];if(!TIED_V)tv[at]=vv[i];}}if(tid<64){uint trow=tid/4,iy=8*(tid&3),sx=tid%4,sy=trow/8,ly=trow%8,ib=2*sx+sy;threadgroup half*dst=tx+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pq=tq+4*64*(sg&1);threadgroup const half*pk=tk+4*64*(sg&1);threadgroup const half*pv=tv+4*64*(sg&1);threadgroup const half*px=tx+64*(sg>>1);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 mq[4],mk[4],mv[4],mx;for(uint i=0;i<4;++i)simdgroup_load(mq[i],pq+64*i,8,0,false);if(have_kv)for(uint i=0;i<4;++i){simdgroup_load(mk[i],pk+64*i,8,0,false);if(!TIED_V)simdgroup_load(mv[i],pv+64*i,8,0,false);}simdgroup_load(mx,px,8,0,false);for(uint i=0;i<4;++i){simdgroup_multiply_accumulate(cq[i],mx,mq[i],cq[i]);if(have_kv){simdgroup_multiply_accumulate(ck[i],mx,mk[i],ck[i]);if(!TIED_V)simdgroup_multiply_accumulate(cv[i],mx,mv[i],cv[i]);}}pq+=8*64;pk+=8*64;pv+=8*64;px+=2*64;}}
    uint rb=32*(sg&1),tb=8*(sg>>1);device float*Q=qout+(t0+tb)*a.q_rows+r0+rb;for(uint i=0;i<4;++i)simdgroup_store(cq[i],Q+8*i,a.q_rows,0,false);if(have_kv){device float*K=kout+(t0+tb)*a.kv_rows+r0+rb;device float*V=vout+(t0+tb)*a.kv_rows+r0+rb;for(uint i=0;i<4;++i){simdgroup_store(ck[i],K+8*i,a.kv_rows,0,false);if(TIED_V)simdgroup_store(ck[i],V+8*i,a.kv_rows,0,false);else simdgroup_store(cv[i],V+8*i,a.kv_rows,0,false);}}
}
kernel void gemma_qkv_q4_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*q[[buffer(2)]],device float*k[[buffer(3)]],device float*v[[buffer(4)]],constant QKVArgs&a[[buffer(5)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){qkv_mma_swizzled_f16<false,false>(w,x,q,k,v,a,g,tid,sg,tile);}
kernel void gemma_qkv_q6v_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*q[[buffer(2)]],device float*k[[buffer(3)]],device float*v[[buffer(4)]],constant QKVArgs&a[[buffer(5)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){qkv_mma_swizzled_f16<true,false>(w,x,q,k,v,a,g,tid,sg,tile);}
kernel void gemma_qkv_tied_v_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*q[[buffer(2)]],device float*k[[buffer(3)]],device float*v[[buffer(4)]],constant QKVArgs&a[[buffer(5)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){qkv_mma_swizzled_f16<false,true>(w,x,q,k,v,a,g,tid,sg,tile);}

template<typename Y>
inline void q6k_mma_swizzled_tm64(device const uchar*w0,device const Y*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=64,KM=32;uint r0=group.x*RM,t0=group.y*TM;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    const uint trow=tid/4,iy=8*(tid&3);for(uint k0=0;k0<a.cols;k0+=KM){half tmp[16];uint row=tid/2,il0=tid&1;if(tid<128)q6_dequant_16_thread(w[(r0+row)*nb+k0/256],2*((k0%256)/32)+il0,tmp);threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<128)for(uint i=0;i<16;++i){uint sx=2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[i];}
        uint sx=tid%4,sy=trow/8,ly=trow%8,ib=8*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const Y*src=y+(t0+trow)*a.cols+k0+iy;load8_to_tile(src,bdst);}else for(uint i=0;i<8;++i)bdst[i]=half(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<4;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=8*64;}
    }
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}
kernel void gemma_q6k_mma_swizzled_tm64_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q6k_mma_swizzled_tm64<half>(w,x,y,a,g,tid,sg,tile);}

template<typename Y>
inline void q6k_mma_swizzled_k64(device const uchar*w0,device const Y*y,device float*out,constant MatArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint nb=a.cols/256;threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];for(uint j=0;j<2;++j)q6_dequant_16_thread(w[(r0+row)*nb+(k0+32*j)/256],2*(((k0+32*j)%256)/32)+il0,tmp+16*j);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[16*j+i];}
        for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*bdst=sb+64*ib+8*ly;if(t0+trow<a.batch){device const Y*src=y+(t0+trow)*a.cols+k0+32*j+iy;load8_to_tile(src,bdst);}else for(uint i=0;i<8;++i)bdst[i]=half(0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}
    }
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}
kernel void gemma_q6k_mma_swizzled_k64_f16(device const uchar*w[[buffer(0)]],device const half*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 g[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){q6k_mma_swizzled_k64<half>(w,x,y,a,g,tid,sg,tile);}

kernel void gemma_q6k_mma_metal_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_float8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];for(uint j=0;j<2;++j){uint kk=k0+32*j,block=kk/256,g=(kk%256)/32,at=((r0/RM)*nb+block)*RM+row;q6_dequant_16_thread(w[at],2*g+il0,tmp+16*j);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[16*j+i];}for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+32*j+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}}
    device float*C=out+(r0+32*(sg&1))+(t0+16*(sg>>1))*a.rows;for(uint i=0;i<8;++i)simdgroup_store(mc[i],C+8*(i%4)+8*a.rows*(i/4),a.rows,0,false);
}

kernel void gemma_q6k_mma_metal_f16acc_f16(device const uchar*w0[[buffer(0)]],device const half*x[[buffer(1)]],device float*out[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint RM=64,TM=32,KM=64;uint r0=group.x*RM,t0=group.y*TM,nb=a.cols/256;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);threadgroup half*sa=tile;threadgroup half*sb=tile+RM*KM;simdgroup_half8x8 mc[8];for(uint i=0;i<8;++i)mc[i]=make_filled_simdgroup_matrix<half,8>(half(0));const uint row=tid/2,il0=tid&1,trow=tid/4,iy=8*(tid&3);
    for(uint k0=0;k0<a.cols;k0+=KM){half tmp[32];for(uint j=0;j<2;++j){uint kk=k0+32*j,block=kk/256,g=(kk%256)/32,at=((r0/RM)*nb+block)*RM+row;q6_dequant_16_thread(w[at],2*g+il0,tmp+16*j);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint j=0;j<2;++j)for(uint i=0;i<16;++i){uint sx=4*j+2*il0+i/8,sy=row/8,lx=row%8,ly=i%8,ib=8*sx+sy;sa[64*ib+8*ly+lx]=tmp[16*j+i];}for(uint j=0;j<2;++j){uint sx=4*j+tid%4,sy=trow/8,ly=trow%8,ib=4*sx+sy;threadgroup half*dst=sb+64*ib+8*ly;if(t0+trow<a.batch)load8_to_tile(x+(t0+trow)*a.cols+k0+32*j+iy,dst);else for(uint i=0;i<8;++i)dst[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*pa=sa+4*64*(sg%2);threadgroup const half*pb=sb+2*64*(sg/2);for(uint ik=0;ik<8;++ik){simdgroup_half8x8 ma[4],mb[2];for(uint i=0;i<4;++i)simdgroup_load(ma[i],pa+64*i,8,0,false);for(uint i=0;i<2;++i)simdgroup_load(mb[i],pb+64*i,8,0,false);for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);pa+=8*64;pb+=4*64;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup half*oc=tile;uint rb=32*(sg&1),tb=16*(sg>>1);for(uint i=0;i<8;++i){uint at=rb+tb*RM+8*(i%4)+8*RM*(i/4);simdgroup_store(mc[i],oc+at,RM,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<RM*TM;i+=128){uint r=i%RM,t=i/RM;if(r0+r<a.rows&&t0+t<a.batch)out[(t0+t)*a.rows+r0+r]=float(oc[i]);}
}

template<typename B,float value(device const B&,uint)>
inline void quant_mm4(device const uchar*weights,device const float*x,device float*y,constant MatArgs&a,uint2 tg,uint tid,uint lane,uint simd,uint nsg,threadgroup float*partial){
    uint r0=tg.x*4,t0=tg.y*4;if(r0>=a.rows||t0>=a.batch)return;device const B*w=(device const B*)(weights+a.weight_offset);uint blocks=a.cols/256;float sums[16];for(uint j=0;j<16;++j)sums[j]=0;
    for(uint i=tid;i<a.cols;i+=256){float xv[4];for(uint t=0;t<4;++t)xv[t]=(t0+t<a.batch)?x[(t0+t)*a.cols+i]:0;for(uint r=0;r<4;++r){float q=(r0+r<a.rows)?value(w[(r0+r)*blocks+i/256],i%256):0;for(uint t=0;t<4;++t)sums[r*4+t]+=q*xv[t];}}
    for(uint j=0;j<16;++j){sums[j]=simd_sum(sums[j]);if(lane==0)partial[j*nsg+simd]=sums[j];}threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)for(uint r=0;r<4&&r0+r<a.rows;++r)for(uint t=0;t<4&&t0+t<a.batch;++t){float z=0;for(uint s=0;s<nsg;++s)z+=partial[(r*4+t)*nsg+s];y[(t0+t)*a.rows+r0+r]=z;}
}
kernel void gemma_q4k_mm4(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 tg[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){quant_mm4<Q4K,q4_value>(w,x,y,a,tg,tid,lane,simd,nsg,p);}
kernel void gemma_q6k_mm4(device const uchar*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant MatArgs&a[[buffer(3)]],uint2 tg[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){quant_mm4<Q6K,q6_value>(w,x,y,a,tg,tid,lane,simd,nsg,p);}

struct EmbArgs{uint width;uint batch;ulong weight_offset;float scale;};
kernel void gemma_q6k_embedding(device const uchar*w0[[buffer(0)]],device const int*ids[[buffer(1)]],device float*out[[buffer(2)]],constant EmbArgs&a[[buffer(3)]],uint2 gid[[thread_position_in_grid]]){
    if(gid.x>=a.width||gid.y>=a.batch)return;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);uint blocks=a.width/256;out[gid.y*a.width+gid.x]=a.scale*q6_value(w[uint(ids[gid.y])*blocks+gid.x/256],gid.x%256);
}
kernel void gemma_q4k_dequant_matrix(device const uchar*w0[[buffer(0)]],device float*out[[buffer(1)]],constant MatArgs&a[[buffer(2)]],uint i[[thread_position_in_grid]]){uint n=a.cols*a.rows;if(i>=n)return;device const Q4K*w=(device const Q4K*)(w0+a.weight_offset);out[i]=q4_value(w[i/256],i%256);}
kernel void gemma_q6k_dequant_matrix(device const uchar*w0[[buffer(0)]],device float*out[[buffer(1)]],constant MatArgs&a[[buffer(2)]],uint i[[thread_position_in_grid]]){uint n=a.cols*a.rows;if(i>=n)return;device const Q6K*w=(device const Q6K*)(w0+a.weight_offset);out[i]=q6_value(w[i/256],i%256);}

struct NormArgs{uint width;uint rows;float eps;};
kernel void gemma_rmsnorm(device const float*x[[buffer(0)]],device const float*weight[[buffer(1)]],device float*y[[buffer(2)]],constant NormArgs&a[[buffer(3)]],uint row[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){
    float s=0;for(uint i=tid;i<a.width;i+=256){float v=x[row*a.width+i];s+=v*v;}s=simd_sum(s);if(lane==0)p[simd]=s;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float z=0;for(uint i=0;i<nsg;++i)z+=p[i];p[0]=rsqrt(z/float(a.width)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<a.width;i+=256)y[row*a.width+i]=x[row*a.width+i]*p[0]*weight[i];
}
kernel void gemma_rmsnorm_f16(device const float*x[[buffer(0)]],device const float*weight[[buffer(1)]],device half*y[[buffer(2)]],constant NormArgs&a[[buffer(3)]],uint row[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){
    float s=0;for(uint i=tid;i<a.width;i+=256){float v=x[row*a.width+i];s+=v*v;}s=simd_sum(s);if(lane==0)p[simd]=s;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float z=0;for(uint i=0;i<nsg;++i)z+=p[i];p[0]=rsqrt(z/float(a.width)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<a.width;i+=256)y[row*a.width+i]=half(x[row*a.width+i]*p[0]*weight[i]);
}
kernel void gemma_rmsnorm_add(device const float*x[[buffer(0)]],device const float*weight[[buffer(1)]],device const float*residual[[buffer(2)]],device float*y[[buffer(3)]],constant NormArgs&a[[buffer(4)]],uint row[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){
    float s=0;for(uint i=tid;i<a.width;i+=256){float v=x[row*a.width+i];s+=v*v;}s=simd_sum(s);if(lane==0)p[simd]=s;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float z=0;for(uint i=0;i<nsg;++i)z+=p[i];p[0]=rsqrt(z/float(a.width)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<a.width;i+=256){uint at=row*a.width+i;y[at]=residual[at]+x[at]*p[0]*weight[i];}
}
kernel void gemma_rmsnorm_add_scale(device const float*x[[buffer(0)]],device const float*weight[[buffer(1)]],device const float*residual[[buffer(2)]],device float*y[[buffer(3)]],constant NormArgs&a[[buffer(4)]],constant float&scale[[buffer(5)]],uint row[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){
    float s=0;for(uint i=tid;i<a.width;i+=256){float v=x[row*a.width+i];s+=v*v;}s=simd_sum(s);if(lane==0)p[simd]=s;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float z=0;for(uint i=0;i<nsg;++i)z+=p[i];p[0]=rsqrt(z/float(a.width)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<a.width;i+=256){uint at=row*a.width+i;y[at]=(residual[at]+x[at]*p[0]*weight[i])*scale;}
}

template<bool SCALE>
inline void rmsnorm_add_then_f16(device const float*x,device const float*weight,device const float*residual,device float*y,device const float*next_weight,device half*next,constant NormArgs&a,float scale,uint row,uint tid,uint lane,uint simd,uint nsg,threadgroup float*p){
    float s=0;for(uint i=tid;i<a.width;i+=256){float v=x[row*a.width+i];s+=v*v;}s=simd_sum(s);if(lane==0)p[simd]=s;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float z=0;for(uint i=0;i<nsg;++i)z+=p[i];p[0]=rsqrt(z/float(a.width)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);float norm=p[0],sy=0;
    for(uint i=tid;i<a.width;i+=256){uint at=row*a.width+i;float v=residual[at]+x[at]*norm*weight[i];if(SCALE)v*=scale;y[at]=v;sy+=v*v;}sy=simd_sum(sy);if(lane==0)p[simd]=sy;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float z=0;for(uint i=0;i<nsg;++i)z+=p[i];p[0]=rsqrt(z/float(a.width)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);norm=p[0];for(uint i=tid;i<a.width;i+=256){uint at=row*a.width+i;next[at]=half(y[at]*norm*next_weight[i]);}
}
kernel void gemma_rmsnorm_add_then_f16(device const float*x[[buffer(0)]],device const float*weight[[buffer(1)]],device const float*residual[[buffer(2)]],device float*y[[buffer(3)]],device const float*next_weight[[buffer(4)]],device half*next[[buffer(5)]],constant NormArgs&a[[buffer(6)]],uint row[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){rmsnorm_add_then_f16<false>(x,weight,residual,y,next_weight,next,a,0.0f,row,tid,lane,simd,nsg,p);}
kernel void gemma_rmsnorm_add_scale_then_f16(device const float*x[[buffer(0)]],device const float*weight[[buffer(1)]],device const float*residual[[buffer(2)]],device float*y[[buffer(3)]],device const float*next_weight[[buffer(4)]],device half*next[[buffer(5)]],constant NormArgs&a[[buffer(6)]],constant float&scale[[buffer(7)]],uint row[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){rmsnorm_add_then_f16<true>(x,weight,residual,y,next_weight,next,a,scale,row,tid,lane,simd,nsg,p);}
kernel void gemma_geglu(device const float*g[[buffer(0)]],device const float*u[[buffer(1)]],device float*y[[buffer(2)]],uint i[[thread_position_in_grid]]){float x=g[i];float gelu=0.5f*x*(1.0f+precise::tanh(0.7978845608028654f*x*(1.0f+0.044715f*x*x)));y[i]=gelu*u[i];}
kernel void gemma_geglu_f16(device const float*g[[buffer(0)]],device const float*u[[buffer(1)]],device half*y[[buffer(2)]],uint i[[thread_position_in_grid]]){float x=g[i];float gelu=0.5f*x*(1.0f+precise::tanh(0.7978845608028654f*x*(1.0f+0.044715f*x*x)));y[i]=half(gelu*u[i]);}
kernel void gemma_add(device const float*a[[buffer(0)]],device const float*b[[buffer(1)]],device float*y[[buffer(2)]],uint i[[thread_position_in_grid]]){y[i]=a[i]+b[i];}
kernel void gemma_copy(device const float*a[[buffer(0)]],device float*y[[buffer(1)]],uint i[[thread_position_in_grid]]){y[i]=a[i];}
kernel void gemma_mtp_concat(device const float*embedding[[buffer(0)]],device const float*hidden[[buffer(1)]],device float*out[[buffer(2)]],uint i[[thread_position_in_grid]]){out[i]=i<3840?embedding[i]:hidden[i-3840];}
kernel void gemma_f32_to_f16(device const float*x[[buffer(0)]],device half*y[[buffer(1)]],uint i[[thread_position_in_grid]]){y[i]=half(x[i]);}
kernel void gemma_f16_to_f32(device const half*x[[buffer(0)]],device float*y[[buffer(1)]],uint i[[thread_position_in_grid]]){y[i]=float(x[i]);}

struct HeadNormArgs{uint dim;uint heads;uint batch;float eps;};
kernel void gemma_head_rmsnorm(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant HeadNormArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*p[[threadgroup(0)]]){
    uint h=group%a.heads,t=group/a.heads,off=(t*a.heads+h)*a.dim;float s=0;for(uint i=tid;i<a.dim;i+=256){float v=x[off+i];s+=v*v;}s=simd_sum(s);if(lane==0)p[simd]=s;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float z=0;for(uint i=0;i<nsg;++i)z+=p[i];p[0]=rsqrt(z/float(a.dim)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<a.dim;i+=256)y[off+i]=x[off+i]*p[0]*(w?w[i]:1.0f);
}

struct NormRopeArgs{uint dim;uint heads;uint batch;uint pos0;float eps;float base;uint use_factors;uint capacity;};
kernel void gemma_q_norm_rope(device float*x[[buffer(0)]],device const float*w[[buffer(1)]],device const float*factors[[buffer(2)]],constant NormRopeArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*scratch[[threadgroup(0)]]){
    uint h=group%a.heads,t=group/a.heads,off=(t*a.heads+h)*a.dim;float sum=0;for(uint d=tid;d<a.dim;d+=256){float z=x[off+d];sum+=z*z;}sum=simd_sum(sum);if(lane==0)scratch[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){sum=0;for(uint i=0;i<nsg;++i)sum+=scratch[i];scratch[0]=rsqrt(sum/float(a.dim)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);float norm=scratch[0];
    for(uint pair=tid;pair<a.dim/2;pair+=256){float theta=float(a.pos0+t)*pow(a.base,-2.0f*float(pair)/float(a.dim));if(a.use_factors)theta/=factors[pair];float cs=cos(theta),sn=sin(theta),v0=x[off+pair]*norm*w[pair],v1=x[off+pair+a.dim/2]*norm*w[pair+a.dim/2];x[off+pair]=v0*cs-v1*sn;x[off+pair+a.dim/2]=v0*sn+v1*cs;}
}

kernel void gemma_kv_norm_rope_store(device const float*k[[buffer(0)]],device const float*v[[buffer(1)]],device const float*kw[[buffer(2)]],device const float*factors[[buffer(3)]],device half*kc[[buffer(4)]],device half*vc[[buffer(5)]],constant NormRopeArgs&a[[buffer(6)]],uint group[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float2*scratch[[threadgroup(0)]]){
    uint h=group%a.heads,t=group/a.heads,src=(t*a.heads+h)*a.dim,slot=(a.pos0+t)%a.capacity,dst=(h*a.capacity+slot)*a.dim;float sk=0,sv=0;for(uint d=tid;d<a.dim;d+=256){float x=k[src+d],y=v[src+d];sk+=x*x;sv+=y*y;}sk=simd_sum(sk);sv=simd_sum(sv);if(lane==0)scratch[sg]=float2(sk,sv);threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float2 sum=0;for(uint i=0;i<nsg;++i)sum+=scratch[i];scratch[0]=rsqrt(sum/float(a.dim)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);float2 norm=scratch[0];
    for(uint pair=tid;pair<a.dim/2;pair+=256){float theta=float(a.pos0+t)*pow(a.base,-2.0f*float(pair)/float(a.dim));if(a.use_factors)theta/=factors[pair];float cs=cos(theta),sn=sin(theta),k0=k[src+pair]*norm.x*kw[pair],k1=k[src+pair+a.dim/2]*norm.x*kw[pair+a.dim/2];kc[dst+pair]=half(k0*cs-k1*sn);kc[dst+pair+a.dim/2]=half(k0*sn+k1*cs);vc[dst+pair]=half(v[src+pair]*norm.y);vc[dst+pair+a.dim/2]=half(v[src+pair+a.dim/2]*norm.y);}
}

kernel void gemma_k_norm_rope_store_tied_v(device const float*k[[buffer(0)]],device const float*kw[[buffer(1)]],device const float*factors[[buffer(2)]],device half*kc[[buffer(3)]],device half*vc[[buffer(4)]],constant NormRopeArgs&a[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*scratch[[threadgroup(0)]]){
    uint h=group%a.heads,t=group/a.heads,src=(t*a.heads+h)*a.dim,slot=(a.pos0+t)%a.capacity,dst=(h*a.capacity+slot)*a.dim;float sum=0;for(uint d=tid;d<a.dim;d+=256){float z=k[src+d];sum+=z*z;}sum=simd_sum(sum);if(lane==0)scratch[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){sum=0;for(uint i=0;i<nsg;++i)sum+=scratch[i];scratch[0]=rsqrt(sum/float(a.dim)+a.eps);}threadgroup_barrier(mem_flags::mem_threadgroup);float norm=scratch[0];
    for(uint pair=tid;pair<a.dim/2;pair+=256){float theta=float(a.pos0+t)*pow(a.base,-2.0f*float(pair)/float(a.dim));if(a.use_factors)theta/=factors[pair];float cs=cos(theta),sn=sin(theta),raw0=k[src+pair],raw1=k[src+pair+a.dim/2],k0=raw0*norm*kw[pair],k1=raw1*norm*kw[pair+a.dim/2];kc[dst+pair]=half(k0*cs-k1*sn);kc[dst+pair+a.dim/2]=half(k0*sn+k1*cs);vc[dst+pair]=half(raw0*norm);vc[dst+pair+a.dim/2]=half(raw1*norm);}
}

struct RopeArgs{uint dim;uint heads;uint batch;uint pos0;float base;uint use_factors;};
kernel void gemma_rope_neox(device float*x[[buffer(0)]],device const float*factors[[buffer(1)]],constant RopeArgs&a[[buffer(2)]],uint3 gid[[thread_position_in_grid]]){
    uint pair=gid.x,h=gid.y,t=gid.z;if(pair>=a.dim/2||h>=a.heads||t>=a.batch)return;uint off=(t*a.heads+h)*a.dim;float theta=float(a.pos0+t)*pow(a.base,-2.0f*float(pair)/float(a.dim));if(a.use_factors)theta/=factors[pair];float cs=cos(theta),sn=sin(theta),v0=x[off+pair],v1=x[off+pair+a.dim/2];x[off+pair]=v0*cs-v1*sn;x[off+pair+a.dim/2]=v0*sn+v1*cs;
}

struct KVArgs{uint dim;uint kv_heads;uint batch;uint pos0;uint capacity;};
kernel void gemma_store_kv(device const float*k[[buffer(0)]],device const float*v[[buffer(1)]],device half*kc[[buffer(2)]],device half*vc[[buffer(3)]],constant KVArgs&a[[buffer(4)]],uint3 gid[[thread_position_in_grid]]){
    uint d=gid.x,h=gid.y,t=gid.z;if(d>=a.dim||h>=a.kv_heads||t>=a.batch)return;uint src=(t*a.kv_heads+h)*a.dim+d,slot=(a.pos0+t)%a.capacity,dst=(h*a.capacity+slot)*a.dim+d;kc[dst]=half(k[src]);vc[dst]=half(v[src]);
}

struct AttnArgs{uint dim;uint heads;uint kv_heads;uint batch;uint pos0;uint capacity;uint window;};
kernel void gemma_attention(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*out[[buffer(3)]],constant AttnArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*scores[[threadgroup(0)]]){
    uint h=group%a.heads,t=group/a.heads,pos=a.pos0+t,kh=h/(a.heads/a.kv_heads);uint start=a.window?((pos+1>a.window)?pos+1-a.window:0):0,span=pos-start+1,qoff=(t*a.heads+h)*a.dim;
    uint sub=lane>>1,sl=lane&1;for(uint base=simd*16;base<span;base+=nsg*16){uint j=base+sub;float s=0;if(j<span){uint p=start+j,slot=p%a.capacity,ko=(kh*a.capacity+slot)*a.dim;for(uint d=sl;d<a.dim;d+=2)s+=q[qoff+d]*float(kc[ko+d]);}s+=simd_shuffle_xor(s,1);if(sl==0&&j<span)scores[j]=s;}threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float*scratch=scores+span;float m=-INFINITY;for(uint j=tid;j<span;j+=256)m=max(m,scores[j]);m=simd_max(m);if(lane==0)scratch[simd]=m;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){m=-INFINITY;for(uint i=0;i<nsg;++i)m=max(m,scratch[i]);scratch[0]=m;}threadgroup_barrier(mem_flags::mem_threadgroup);m=scratch[0];
    float z=0;for(uint j=tid;j<span;j+=256){float v=exp(scores[j]-m);scores[j]=v;z+=v;}z=simd_sum(z);if(lane==0)scratch[simd]=z;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){z=0;for(uint i=0;i<nsg;++i)z+=scratch[i];scratch[0]=1.0f/z;}threadgroup_barrier(mem_flags::mem_threadgroup);float invz=scratch[0];
    for(uint d=tid;d<a.dim;d+=256){float v=0;for(uint j=0;j<span;++j){uint slot=(start+j)%a.capacity;v+=scores[j]*float(vc[(kh*a.capacity+slot)*a.dim+d]);}out[qoff+d]=v*invz;}
}

// Gemma 4 Assistant queries the target cache at the next position but does not
// append K/V of its own. Consequently the visible span ends at pos-1.
kernel void gemma_mtp_attention(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*out[[buffer(3)]],constant AttnArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint simd[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*scores[[threadgroup(0)]]){
    uint h=group%a.heads,t=group/a.heads,pos=a.pos0,kh=h/(a.heads/a.kv_heads);uint start=a.window&&pos>a.window?pos-a.window:0,span=pos-start,qoff=(t*a.heads+h)*a.dim;
    uint sub=lane>>1,sl=lane&1;for(uint base=simd*16;base<span;base+=nsg*16){uint j=base+sub;float s=0;if(j<span){uint slot=(start+j)%a.capacity,ko=(kh*a.capacity+slot)*a.dim;for(uint d=sl;d<a.dim;d+=2)s+=q[qoff+d]*float(kc[ko+d]);}s+=simd_shuffle_xor(s,1);if(sl==0&&j<span)scores[j]=s;}threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float*scratch=scores+span;float m=-INFINITY;for(uint j=tid;j<span;j+=256)m=max(m,scores[j]);m=simd_max(m);if(lane==0)scratch[simd]=m;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){m=-INFINITY;for(uint i=0;i<nsg;++i)m=max(m,scratch[i]);scratch[0]=m;}threadgroup_barrier(mem_flags::mem_threadgroup);m=scratch[0];
    float z=0;for(uint j=tid;j<span;j+=256){float v=exp(scores[j]-m);scores[j]=v;z+=v;}z=simd_sum(z);if(lane==0)scratch[simd]=z;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){z=0;for(uint i=0;i<nsg;++i)z+=scratch[i];scratch[0]=1.0f/z;}threadgroup_barrier(mem_flags::mem_threadgroup);float invz=scratch[0];
    for(uint d=tid;d<a.dim;d+=256){float v=0;for(uint j=0;j<span;++j){uint slot=(start+j)%a.capacity;v+=scores[j]*float(vc[(kh*a.capacity+slot)*a.dim+d]);}out[qoff+d]=v*invz;}
}

struct FlashArgs{
    uint dim;uint heads;uint kv_heads;uint batch;uint pos0;uint capacity;uint window;uint span;uint causal;
    uint sparse_sink_blocks;uint sparse_recent_block;uint sparse_stride;uint sparse_selected_blocks;
};
kernel void gemma_flash_qk(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device float*scores[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=8,KT=32,DK=32;uint kb=group.x*KT,y=group.y,h=y%a.heads,t0=(y/a.heads)*QT,kh=h/(a.heads/a.kv_heads),base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;threadgroup half*sq=tile;threadgroup half*sk=tile+QT*DK;simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint d0=0;d0<a.dim;d0+=DK){for(uint i=tid;i<QT*DK;i+=128){uint t=i/DK,d=i%DK;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d0+d]):half(0);}for(uint i=tid;i<KT*DK;i+=128){uint k=i/DK,d=i%DK,j=kb+k;if(j<a.span){uint slot=(base+j)%a.capacity;sk[i]=kc[(kh*a.capacity+slot)*a.dim+d0+d];}else sk[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint dk=0;dk<DK;dk+=8){simdgroup_half8x8 mq,mk;simdgroup_load(mq,sq+dk,DK,0,false);simdgroup_load(mk,sk+(sg*8)*DK+dk,DK,0,true);simdgroup_multiply_accumulate(acc,mq,mk,acc);}threadgroup_barrier(mem_flags::mem_threadgroup);}
    threadgroup float*tmp=(threadgroup float*)tile;simdgroup_store(acc,tmp+sg*8,KT,0,false);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<QT*KT;i+=128){uint t=i/KT,k=i%KT,j=kb+k;if(t0+t<a.batch&&j<a.span){uint pos=a.pos0+t0+t,p=base+j,limit=a.causal?pos:a.pos0+a.batch-1;bool valid=(!a.causal||p<=pos)&&(!a.window||p+a.window>limit);scores[((t0+t)*a.heads+h)*a.span+j]=valid?tmp[i]:-INFINITY;}}
}
kernel void gemma_flash_softmax(device float*scores[[buffer(0)]],constant FlashArgs&a[[buffer(1)]],uint row[[threadgroup_position_in_grid]],uint tid[[thread_index_in_threadgroup]],uint lane[[thread_index_in_simdgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint nsg[[simdgroups_per_threadgroup]],threadgroup float*scratch[[threadgroup(0)]]){
    device float*s=scores+row*a.span;float m=-INFINITY;for(uint j=tid;j<a.span;j+=256)m=max(m,s[j]);m=simd_max(m);if(lane==0)scratch[sg]=m;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){m=-INFINITY;for(uint i=0;i<nsg;++i)m=max(m,scratch[i]);scratch[0]=m;}threadgroup_barrier(mem_flags::mem_threadgroup);m=scratch[0];float z=0;for(uint j=tid;j<a.span;j+=256){float v=exp(s[j]-m);s[j]=v;z+=v;}z=simd_sum(z);if(lane==0)scratch[sg]=z;threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){z=0;for(uint i=0;i<nsg;++i)z+=scratch[i];scratch[0]=1.0f/z;}threadgroup_barrier(mem_flags::mem_threadgroup);float inv=scratch[0];for(uint j=tid;j<a.span;j+=256)s[j]*=inv;
}
kernel void gemma_flash_pv(device const float*scores[[buffer(0)]],device const half*vc[[buffer(1)]],device float*out[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=8,RT=64,KT=32;uint d0=group.x*RT,y=group.y,h=y%a.heads,t0=(y/a.heads)*QT,kh=h/(a.heads/a.kv_heads),base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;threadgroup half*sv=tile;threadgroup half*sp=tile+RT*KT;simdgroup_float8x8 acc[4];for(uint i=0;i<4;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k0=0;k0<a.span;k0+=KT){for(uint i=tid;i<RT*KT;i+=64){uint r=i/KT,k=i%KT,j=k0+k;if(j<a.span&&d0+r<a.dim){uint slot=(base+j)%a.capacity;sv[i]=vc[(kh*a.capacity+slot)*a.dim+d0+r];}else sv[i]=half(0);}for(uint i=tid;i<QT*KT;i+=64){uint t=i/KT,k=i%KT,j=k0+k;sp[i]=(t0+t<a.batch&&j<a.span)?half(scores[((t0+t)*a.heads+h)*a.span+j]):half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);uint rb=sg*32;for(uint kk=0;kk<KT;kk+=8){simdgroup_half8x8 mp,mv[4];simdgroup_load(mp,sp+kk,KT,0,false);for(uint r=0;r<4;++r)simdgroup_load(mv[r],sv+(rb+r*8)*KT+kk,KT,0,true);for(uint r=0;r<4;++r)simdgroup_multiply_accumulate(acc[r],mp,mv[r],acc[r]);}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(t0<a.batch){device float*dst=out+(t0*a.heads+h)*a.dim+d0+sg*32;for(uint r=0;r<4;++r)simdgroup_store(acc[r],dst+r*8,a.heads*a.dim,0,false);}
}

// Each threadgroup owns one (8 queries x one head) score slab. It reuses Q
// across all key tiles, applies the causal mask, and normalizes directly into a
// compact FP16 score buffer consumed by the matrix PV kernel below.
template<bool TRIM,bool CACHE_Q>
inline void flash_qk_softmax_f16_impl(device const float*q,device const half*kc,device half*scores,constant FlashArgs&a,uint group,ushort tid,ushort lane,ushort sg,threadgroup half*tile){
    constexpr uint QT=8,KT=32,DK=32;uint y=group,h=y%a.heads,t0=(y/a.heads)*QT,kh=h/(a.heads/a.kv_heads),base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;threadgroup half*sq=tile;threadgroup half*sk=tile+(CACHE_Q?QT*a.dim:QT*DK);if(CACHE_Q){for(uint i=tid;i<QT*a.dim;i+=128){uint t=i/a.dim,d=i%a.dim;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d]):half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);}
    uint last=min(t0+QT,a.batch)-1,tile_span=TRIM&&a.causal?min(a.span,a.pos0+last+1-base):a.span;for(uint kb=0;kb<tile_span;kb+=KT){
        simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);
        for(uint d0=0;d0<a.dim;d0+=DK){if(!CACHE_Q)for(uint i=tid;i<QT*DK;i+=128){uint t=i/DK,d=i%DK;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d0+d]):half(0);}for(uint i=tid;i<KT*DK;i+=128){uint k=i/DK,d=i%DK,j=kb+k;if(j<tile_span){uint slot=(base+j)%a.capacity;sk[i]=kc[(kh*a.capacity+slot)*a.dim+d0+d];}else sk[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint dk=0;dk<DK;dk+=8){simdgroup_half8x8 mq,mk;simdgroup_load(mq,sq+(CACHE_Q?d0:0)+dk,CACHE_Q?a.dim:DK,0,false);simdgroup_load(mk,sk+(sg*8)*DK+dk,DK,0,true);simdgroup_multiply_accumulate(acc,mq,mk,acc);}threadgroup_barrier(mem_flags::mem_threadgroup);}
        threadgroup float*tmp=CACHE_Q?(threadgroup float*)(sk+KT*DK):(threadgroup float*)tile;simdgroup_store(acc,tmp+sg*8,KT,0,false);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<QT*KT;i+=128){uint t=i/KT,k=i%KT,j=kb+k;if(t0+t<a.batch&&j<tile_span){uint pos=a.pos0+t0+t,p=base+j,limit=a.causal?pos:a.pos0+a.batch-1;bool valid=(!a.causal||p<=pos)&&(!a.window||p+a.window>limit);scores[((t0+t)*a.heads+h)*a.span+j]=valid?half(tmp[i]):half(-INFINITY);}}threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint pair=0;pair<2;++pair){uint t=2*sg+pair;if(t0+t>=a.batch)continue;uint row_span=TRIM&&a.causal?min(a.span,a.pos0+t0+t+1-base):a.span;device half*s=scores+((t0+t)*a.heads+h)*a.span;float m=-INFINITY;for(uint j=lane;j<row_span;j+=32)m=max(m,float(s[j]));m=simd_max(m);float z=0;for(uint j=lane;j<row_span;j+=32)z+=exp(float(s[j])-m);z=simd_sum(z);float inv=1.0f/z;for(uint j=lane;j<row_span;j+=32)s[j]=half(exp(float(s[j])-m)*inv);}
}
kernel void gemma_flash_qk_softmax_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device half*scores[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_qk_softmax_f16_impl<false,false>(q,kc,scores,a,group,tid,lane,sg,tile);}
kernel void gemma_flash_qk_softmax_f16_causal(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device half*scores[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_qk_softmax_f16_impl<true,false>(q,kc,scores,a,group,tid,lane,sg,tile);}
kernel void gemma_flash_qk_softmax_f16_causal_cached(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device half*scores[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_qk_softmax_f16_impl<true,true>(q,kc,scores,a,group,tid,lane,sg,tile);}

template<bool TRIM>
inline void flash_pv_f16_impl(device const half*scores,device const half*vc,device float*out,constant FlashArgs&a,uint2 group,ushort tid,ushort sg,threadgroup half*tile){
    constexpr uint QT=8,RT=64,KT=32;uint d0=group.x*RT,y=group.y,h=y%a.heads,t0=(y/a.heads)*QT,kh=h/(a.heads/a.kv_heads),base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;threadgroup half*sv=tile;threadgroup half*sp=tile+RT*KT;simdgroup_float8x8 acc[4];for(uint i=0;i<4;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    uint last=min(t0+QT,a.batch)-1,tile_span=TRIM&&a.causal?min(a.span,a.pos0+last+1-base):a.span;for(uint k0=0;k0<tile_span;k0+=KT){for(uint i=tid;i<RT*KT;i+=64){uint r=i/KT,k=i%KT,j=k0+k;if(j<tile_span&&d0+r<a.dim){uint slot=(base+j)%a.capacity;sv[i]=vc[(kh*a.capacity+slot)*a.dim+d0+r];}else sv[i]=half(0);}for(uint i=tid;i<QT*KT;i+=64){uint t=i/KT,k=i%KT,j=k0+k,row_span=TRIM&&a.causal?min(a.span,a.pos0+t0+t+1-base):a.span;sp[i]=(t0+t<a.batch&&j<tile_span&&(!TRIM||!a.causal||j<row_span))?scores[((t0+t)*a.heads+h)*a.span+j]:half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);uint rb=sg*32;for(uint kk=0;kk<KT;kk+=8){simdgroup_half8x8 mp,mv[4];simdgroup_load(mp,sp+kk,KT,0,false);for(uint r=0;r<4;++r)simdgroup_load(mv[r],sv+(rb+r*8)*KT+kk,KT,0,true);for(uint r=0;r<4;++r)simdgroup_multiply_accumulate(acc[r],mp,mv[r],acc[r]);}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(t0<a.batch){device float*dst=out+(t0*a.heads+h)*a.dim+d0+sg*32;for(uint r=0;r<4;++r)simdgroup_store(acc[r],dst+r*8,a.heads*a.dim,0,false);}
}
kernel void gemma_flash_pv_f16(device const half*scores[[buffer(0)]],device const half*vc[[buffer(1)]],device float*out[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_pv_f16_impl<false>(scores,vc,out,a,group,tid,sg,tile);}
kernel void gemma_flash_pv_f16_causal(device const half*scores[[buffer(0)]],device const half*vc[[buffer(1)]],device float*out[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_pv_f16_impl<true>(scores,vc,out,a,group,tid,sg,tile);}

kernel void gemma_flash_qk_softmax_f16_causal_q16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device half*scores[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=16,KT=32,DK=32;uint y=group,h=y%a.heads,t0=(y/a.heads)*QT,kh=h/(a.heads/a.kv_heads),base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;threadgroup half*sq=tile;threadgroup half*sk=sq+QT*a.dim;for(uint i=tid;i<QT*a.dim;i+=128){uint t=i/a.dim,d=i%a.dim;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d]):half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);uint last=min(t0+QT,a.batch)-1,tile_span=min(a.span,a.pos0+last+1-base);
    for(uint kb=0;kb<tile_span;kb+=KT){simdgroup_float8x8 acc[2];for(uint i=0;i<2;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);for(uint d0=0;d0<a.dim;d0+=DK){for(uint i=tid;i<KT*DK;i+=128){uint k=i/DK,d=i%DK,j=kb+k;if(j<tile_span){uint slot=(base+j)%a.capacity;sk[i]=kc[(kh*a.capacity+slot)*a.dim+d0+d];}else sk[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint dk=0;dk<DK;dk+=8){simdgroup_half8x8 mk;simdgroup_load(mk,sk+(sg*8)*DK+dk,DK,0,true);for(uint qb=0;qb<2;++qb){simdgroup_half8x8 mq;simdgroup_load(mq,sq+qb*8*a.dim+d0+dk,a.dim,0,false);simdgroup_multiply_accumulate(acc[qb],mq,mk,acc[qb]);}}threadgroup_barrier(mem_flags::mem_threadgroup);}threadgroup float*tmp=(threadgroup float*)(sk+KT*DK);for(uint qb=0;qb<2;++qb)simdgroup_store(acc[qb],tmp+qb*8*KT+sg*8,KT,0,false);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<QT*KT;i+=128){uint t=i/KT,k=i%KT,j=kb+k;if(t0+t<a.batch&&j<tile_span){uint pos=a.pos0+t0+t,p=base+j;bool valid=p<=pos&&(!a.window||p+a.window>pos);scores[((t0+t)*a.heads+h)*a.span+j]=valid?half(tmp[i]):half(-INFINITY);}}threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(uint pair=0;pair<4;++pair){uint t=4*sg+pair;if(t0+t>=a.batch)continue;uint row_span=min(a.span,a.pos0+t0+t+1-base);device half*s=scores+((t0+t)*a.heads+h)*a.span;float m=-INFINITY;for(uint j=lane;j<row_span;j+=32)m=max(m,float(s[j]));m=simd_max(m);float z=0;for(uint j=lane;j<row_span;j+=32)z+=exp(float(s[j])-m);z=simd_sum(z);float inv=1.0f/z;for(uint j=lane;j<row_span;j+=32)s[j]=half(exp(float(s[j])-m)*inv);}
}

kernel void gemma_flash_pv_f16_causal_q16(device const half*scores[[buffer(0)]],device const half*vc[[buffer(1)]],device float*out[[buffer(2)]],constant FlashArgs&a[[buffer(3)]],uint2 group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=16,RT=64,KT=32;uint d0=group.x*RT,y=group.y,h=y%a.heads,t0=(y/a.heads)*QT,kh=h/(a.heads/a.kv_heads),base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;threadgroup half*sv=tile;threadgroup half*sp=tile+RT*KT;simdgroup_float8x8 acc[8];for(uint i=0;i<8;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);uint last=min(t0+QT,a.batch)-1,tile_span=min(a.span,a.pos0+last+1-base);
    for(uint k0=0;k0<tile_span;k0+=KT){for(uint i=tid;i<RT*KT;i+=64){uint r=i/KT,k=i%KT,j=k0+k;if(j<tile_span&&d0+r<a.dim){uint slot=(base+j)%a.capacity;sv[i]=vc[(kh*a.capacity+slot)*a.dim+d0+r];}else sv[i]=half(0);}for(uint i=tid;i<QT*KT;i+=64){uint t=i/KT,k=i%KT,j=k0+k,row_span=min(a.span,a.pos0+t0+t+1-base);sp[i]=(t0+t<a.batch&&j<tile_span&&j<row_span)?scores[((t0+t)*a.heads+h)*a.span+j]:half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);uint rb=sg*32;for(uint kk=0;kk<KT;kk+=8){simdgroup_half8x8 mv[4];for(uint r=0;r<4;++r)simdgroup_load(mv[r],sv+(rb+r*8)*KT+kk,KT,0,true);for(uint qb=0;qb<2;++qb){simdgroup_half8x8 mp;simdgroup_load(mp,sp+qb*8*KT+kk,KT,0,false);for(uint r=0;r<4;++r)simdgroup_multiply_accumulate(acc[qb*4+r],mp,mv[r],acc[qb*4+r]);}}threadgroup_barrier(mem_flags::mem_threadgroup);}
    for(uint qb=0;qb<2;++qb)if(t0+qb*8<a.batch){device float*dst=out+((t0+qb*8)*a.heads+h)*a.dim+d0+sg*32;for(uint r=0;r<4;++r)simdgroup_store(acc[qb*4+r],dst+r*8,a.heads*a.dim,0,false);}
}

// One-pass tiled attention for Gemma's eight-query prefill tiles. QK remains a
// simdgroup matrix multiply, while the causal mask, online softmax and PV are
// kept inside the same threadgroup. No O(batch * heads * context) score tensor
// is materialized in device memory.
template<bool TRIM,bool CACHE_Q>
inline void flash_online_impl(device const float*q,device const half*kc,device const half*vc,device float*out,constant FlashArgs&a,uint group,ushort tid,ushort lane,ushort sg,threadgroup half*tile){
    constexpr uint QT=8,KT=32,DK=32;
    uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads);
    uint base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;
    threadgroup half*sq=tile;
    threadgroup half*sk=sq+(CACHE_Q?QT*a.dim:QT*DK);
    threadgroup float*scores=(threadgroup float*)(sk+KT*DK);
    threadgroup float*state=scores+QT*KT;
    threadgroup float*running_m=state;
    threadgroup float*running_l=state+QT;
    threadgroup float*alpha=state+2*QT;
    float ov[QT*2];
    for(uint i=0;i<QT*2;++i)ov[i]=0.0f;
    if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}
    if(CACHE_Q)for(uint i=tid;i<QT*a.dim;i+=256){uint t=i/a.dim,d=i%a.dim;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d]):half(0);}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint last=min(t0+QT,a.batch)-1,tile_span=TRIM&&a.causal?min(a.span,a.pos0+last+1-base):a.span;for(uint kb=0;kb<tile_span;kb+=KT){
        simdgroup_float8x8 qk=make_filled_simdgroup_matrix<float,8>(0.0f);
        for(uint d0=0;d0<a.dim;d0+=DK){
            if(!CACHE_Q)for(uint i=tid;i<QT*DK;i+=256){uint t=i/DK,d=i%DK;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d0+d]):half(0);}
            for(uint i=tid;i<KT*DK;i+=256){uint k=i/DK,d=i%DK,j=kb+k;if(j<tile_span){uint slot=(base+j)%a.capacity;sk[i]=kc[(kh*a.capacity+slot)*a.dim+d0+d];}else sk[i]=half(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(sg<4)for(uint dk=0;dk<DK;dk+=8){simdgroup_half8x8 mq,mk;simdgroup_load(mq,sq+(CACHE_Q?d0:0)+dk,CACHE_Q?a.dim:DK,0,false);simdgroup_load(mk,sk+(sg*8)*DK+dk,DK,0,true);simdgroup_multiply_accumulate(qk,mq,mk,qk);}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if(sg<4)simdgroup_store(qk,scores+sg*8,KT,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint t=sg,j=kb+lane,pos=a.pos0+t0+t,p=base+j;
        uint limit=a.causal?pos:a.pos0+a.batch-1;
        bool valid=t0+t<a.batch&&j<a.span&&(!a.causal||p<=pos)&&(!a.window||p+a.window>limit);
        float s=valid?scores[t*KT+lane]:-INFINITY;
        float tile_m=simd_max(s),new_m=max(running_m[t],tile_m);
        float scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f;
        float e=valid?exp(s-new_m):0.0f;
        scores[t*KT+lane]=e;
        float tile_l=simd_sum(e);
        if(lane==0){alpha[t]=scale;running_m[t]=new_m;running_l[t]=scale*running_l[t]+tile_l;}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for(uint r=0;r<2;++r){
            uint d=tid+r*256;if(d>=a.dim)continue;
            for(uint t=0;t<QT;++t)ov[t*2+r]*=alpha[t];
            for(uint k=0;k<KT&&kb+k<tile_span;++k){uint slot=(base+kb+k)%a.capacity;float v=float(vc[(kh*a.capacity+slot)*a.dim+d]);for(uint t=0;t<QT;++t)ov[t*2+r]+=scores[t*KT+k]*v;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint r=0;r<2;++r){uint d=tid+r*256;if(d<a.dim)for(uint t=0;t<QT&&t0+t<a.batch;++t)out[((t0+t)*a.heads+h)*a.dim+d]=ov[t*2+r]/running_l[t];}
}
kernel void gemma_flash_online(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_online_impl<false,false>(q,kc,vc,out,a,group,tid,lane,sg,tile);}
kernel void gemma_flash_online_causal(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_online_impl<true,false>(q,kc,vc,out,a,group,tid,lane,sg,tile);}
kernel void gemma_flash_online_causal_cached(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_online_impl<true,true>(q,kc,vc,out,a,group,tid,lane,sg,tile);}

// FlashAttention-4-inspired Metal schedule. A 64-key tile keeps all eight
// SIMD groups busy during QK, then applies the causal mask, online softmax and
// PV in the same threadgroup. Values are reused across the whole query tile and
// the result is written directly as FP16, avoiding the separate F32->F16 pass.
template<uint QT,uint DIM>
inline void flash4_online_f16_impl(device const float*q,device const half*kc,device const half*vc,device half*out,constant FlashArgs&a,uint group,ushort tid,ushort lane,ushort sg,threadgroup half*tile){
    constexpr uint KT=64,DK=32,QB=QT/8,DR=DIM/256;
    uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads),base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;
    threadgroup half*sq=tile;threadgroup half*sk=sq+QT*DIM;threadgroup float*scores=(threadgroup float*)(sk+KT*DK);threadgroup float*state=scores+QT*KT;threadgroup float*running_m=state;threadgroup float*running_l=state+QT;threadgroup float*alpha=state+2*QT;
    float ov[QT*DR];for(uint i=0;i<QT*DR;++i)ov[i]=0.0f;if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}for(uint i=tid;i<QT*DIM;i+=256){uint t=i/DIM,d=i%DIM;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*DIM+d]):half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=min(t0+QT,a.batch)-1,tile_span=a.causal?min(a.span,a.pos0+last+1-base):a.span;for(uint kb=0;kb<tile_span;kb+=KT){
        simdgroup_float8x8 qk[QB];for(uint i=0;i<QB;++i)qk[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
        for(uint d0=0;d0<DIM;d0+=DK){for(uint i=tid;i<KT*DK;i+=256){uint k=i/DK,d=i%DK,j=kb+k;if(j<tile_span){uint slot=(base+j)%a.capacity;sk[i]=kc[(kh*a.capacity+slot)*DIM+d0+d];}else sk[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint dk=0;dk<DK;dk+=8){simdgroup_half8x8 mk;simdgroup_load(mk,sk+(sg*8)*DK+dk,DK,0,true);for(uint qb=0;qb<QB;++qb){simdgroup_half8x8 mq;simdgroup_load(mq,sq+qb*8*DIM+d0+dk,DIM,0,false);simdgroup_multiply_accumulate(qk[qb],mq,mk,qk[qb]);}}threadgroup_barrier(mem_flags::mem_threadgroup);}
        for(uint qb=0;qb<QB;++qb)simdgroup_store(qk[qb],scores+qb*8*KT+sg*8,KT,0,false);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint qi=0;qi<QB;++qi){uint t=QB*sg+qi,j0=kb+lane,j1=j0+32,pos=a.pos0+t0+t,p0=base+j0,p1=base+j1,limit=a.causal?pos:a.pos0+a.batch-1;bool row=t0+t<a.batch,v0=row&&j0<tile_span&&(!a.causal||p0<=pos)&&(!a.window||p0+a.window>limit),v1=row&&j1<tile_span&&(!a.causal||p1<=pos)&&(!a.window||p1+a.window>limit);float s0=v0?scores[t*KT+lane]:-INFINITY,s1=v1?scores[t*KT+32+lane]:-INFINITY,tile_m=simd_max(max(s0,s1)),new_m=max(running_m[t],tile_m),scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f,e0=v0?exp(s0-new_m):0.0f,e1=v1?exp(s1-new_m):0.0f;scores[t*KT+lane]=e0;scores[t*KT+32+lane]=e1;float tile_l=simd_sum(e0+e1);if(lane==0){alpha[t]=scale;running_m[t]=new_m;running_l[t]=scale*running_l[t]+tile_l;}}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint r=0;r<DR;++r){uint d=tid+r*256;if(d>=DIM)continue;for(uint t=0;t<QT;++t)ov[t*DR+r]*=alpha[t];for(uint k=0;k<KT&&kb+k<tile_span;++k){uint slot=(base+kb+k)%a.capacity;float v=float(vc[(kh*a.capacity+slot)*DIM+d]);for(uint t=0;t<QT;++t)ov[t*DR+r]+=scores[t*KT+k]*v;}}threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint r=0;r<DR;++r){uint d=tid+r*256;if(d<DIM)for(uint t=0;t<QT&&t0+t<a.batch;++t)out[((t0+t)*a.heads+h)*DIM+d]=half(ov[t*DR+r]/running_l[t]);}
}
kernel void gemma_flash4_online_causal_q16_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash4_online_f16_impl<16,512>(q,kc,vc,out,a,group,tid,lane,sg,tile);}
kernel void gemma_flash4_online_causal_swa_q32_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash4_online_f16_impl<32,256>(q,kc,vc,out,a,group,tid,lane,sg,tile);}

// llama.cpp-style global attention schedule for long prefills. Eight SIMD
// groups cooperatively form the 8x64 score tile, then split the 512 output
// channels and use simdgroup matrix multiply for P*V.
kernel void gemma_flash_llama_causal_q8_f16_shared(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=8,KT=64,DIM=512;
    uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads);
    threadgroup half*sq=tile;
    threadgroup float*scores=(threadgroup float*)(sq+QT*DIM);
    threadgroup float*ov=scores+QT*KT;
    threadgroup float*state=ov+QT*DIM;
    threadgroup float*running_m=state;
    threadgroup float*running_l=state+QT;
    threadgroup float*alpha=state+2*QT;
    if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}
    for(uint i=tid;i<QT*DIM;i+=256){
        uint t=i/DIM,d=i%DIM;
        sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*DIM+d]):half(0);
        ov[i]=0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=min(t0+QT,a.batch)-1,tile_span=a.causal?min(a.span,a.pos0+last+1):a.span;
    for(uint kb=0;kb<tile_span;kb+=KT){
        simdgroup_float8x8 qk=make_filled_simdgroup_matrix<float,8>(0.0f);
        device const half*pk=kc+(kh*a.capacity+kb+sg*8)*DIM;
        for(uint d0=0;d0<DIM;d0+=16){
            simdgroup_half8x8 mq[2],mk[2];
            simdgroup_load(mq[0],sq+d0,DIM,0,false);
            simdgroup_load(mq[1],sq+d0+8,DIM,0,false);
            simdgroup_load(mk[0],pk+d0,DIM,0,true);
            simdgroup_load(mk[1],pk+d0+8,DIM,0,true);
            simdgroup_multiply_accumulate(qk,mq[0],mk[0],qk);
            simdgroup_multiply_accumulate(qk,mq[1],mk[1],qk);
        }
        simdgroup_store(qk,scores+sg*8,KT,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint t=sg,pos=a.pos0+t0+t,j0=kb+lane,j1=j0+32;
        bool row=t0+t<a.batch,v0=row&&j0<tile_span&&(!a.causal||j0<=pos),v1=row&&j1<tile_span&&(!a.causal||j1<=pos);
        float s0=v0?scores[t*KT+lane]:-INFINITY,s1=v1?scores[t*KT+32+lane]:-INFINITY;
        float tile_m=simd_max(max(s0,s1)),new_m=max(running_m[t],tile_m);
        float scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f;
        float e0=v0?exp(s0-new_m):0.0f,e1=v1?exp(s1-new_m):0.0f;
        scores[t*KT+lane]=e0;scores[t*KT+32+lane]=e1;
        float tile_l=simd_sum(e0+e1);
        if(lane==0){alpha[t]=scale;running_m[t]=new_m;running_l[t]=scale*running_l[t]+tile_l;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<QT*DIM;i+=256)ov[i]*=alpha[i/DIM];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc[8];
        for(uint col=0;col<8;++col)simdgroup_load(acc[col],ov+sg*64+col*8,DIM,0,false);
        for(uint k=0;k<KT;k+=8){
            simdgroup_float8x8 ps;
            simdgroup_load(ps,scores+k,KT,0,false);
            device const half*pv=vc+(kh*a.capacity+kb+k)*DIM+sg*64;
            for(uint col=0;col<8;++col){
                simdgroup_half8x8 mv;
                simdgroup_load(mv,pv+col*8,DIM,0,false);
                simdgroup_multiply_accumulate(acc[col],ps,mv,acc[col]);
            }
        }
        for(uint col=0;col<8;++col)simdgroup_store(acc[col],ov+sg*64+col*8,DIM,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=tid;i<QT*DIM;i+=256){
        uint t=i/DIM,d=i%DIM;
        if(t0+t<a.batch)out[((t0+t)*a.heads+h)*DIM+d]=half(ov[i]/running_l[t]);
    }
}

// Keep the 8x512 output accumulator in SIMD registers across cache tiles.
// The same kernel handles long prefills and a single-token long decode; inactive
// query rows are masked without materializing a heads-by-context score buffer.
// A diagonal 8x8 matrix applies online-softmax rescaling without spilling the
// accumulator to threadgroup memory. The query/score storage and the final
// FP32-to-FP16 conversion scratch overlap, cutting the allocation from 26 KiB
// to 16 KiB and allowing another resident threadgroup on M3.
kernel void gemma_flash_llama_causal_q8_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=8,KT=128,DIM=512;
    uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads);
    threadgroup half*sq=tile;
    threadgroup float*scores=(threadgroup float*)(sq+QT*DIM);
    threadgroup float*diag=scores+QT*KT;
    threadgroup float*state=diag+QT*QT;
    threadgroup float*running_m=state;
    threadgroup float*running_l=state+QT;
    threadgroup float*alpha=state+2*QT;
    if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}
    if(tid<QT*QT)diag[tid]=0.0f;
    for(uint i=tid;i<QT*DIM;i+=256){
        uint t=i/DIM,d=i%DIM;
        sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*DIM+d]):half(0);
    }
    simdgroup_float8x8 acc[8];
    for(uint col=0;col<8;++col)acc[col]=make_filled_simdgroup_matrix<float,8>(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=min(t0+QT,a.batch)-1,logical_span=a.causal?min(a.span,a.pos0+last+1):a.span;
    uint logical_blocks=(logical_span+KT-1)/KT;
    uint selected_blocks=a.sparse_selected_blocks?a.sparse_selected_blocks:logical_blocks;
    uint sampled_blocks=a.sparse_selected_blocks
        ? (a.sparse_recent_block-a.sparse_sink_blocks+a.sparse_stride-1)/a.sparse_stride
        : 0;
    for(uint virtual_block=0;virtual_block<selected_blocks;++virtual_block){
        uint physical_block=virtual_block;
        if(a.sparse_selected_blocks){
            if(virtual_block<a.sparse_sink_blocks)physical_block=virtual_block;
            else if(virtual_block<a.sparse_sink_blocks+sampled_blocks)
                physical_block=a.sparse_sink_blocks+
                    (virtual_block-a.sparse_sink_blocks)*a.sparse_stride;
            else physical_block=a.sparse_recent_block+
                    (virtual_block-a.sparse_sink_blocks-sampled_blocks);
        }
        uint kb=physical_block*KT;
        simdgroup_float8x8 qk[2];
        qk[0]=make_filled_simdgroup_matrix<float,8>(0.0f);
        qk[1]=make_filled_simdgroup_matrix<float,8>(0.0f);
        device const half*pk=kc+(kh*a.capacity+kb+sg*16)*DIM;
        for(uint d0=0;d0<DIM;d0+=16){
            simdgroup_half8x8 mq[2],mk[4];
            simdgroup_load(mq[0],sq+d0,DIM,0,false);
            simdgroup_load(mq[1],sq+d0+8,DIM,0,false);
            simdgroup_load(mk[0],pk+d0,DIM,0,true);
            simdgroup_load(mk[1],pk+d0+8,DIM,0,true);
            simdgroup_load(mk[2],pk+8*DIM+d0,DIM,0,true);
            simdgroup_load(mk[3],pk+8*DIM+d0+8,DIM,0,true);
            simdgroup_multiply_accumulate(qk[0],mq[0],mk[0],qk[0]);
            simdgroup_multiply_accumulate(qk[0],mq[1],mk[1],qk[0]);
            simdgroup_multiply_accumulate(qk[1],mq[0],mk[2],qk[1]);
            simdgroup_multiply_accumulate(qk[1],mq[1],mk[3],qk[1]);
        }
        simdgroup_store(qk[0],scores+sg*16,KT,0,false);
        simdgroup_store(qk[1],scores+sg*16+8,KT,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint t=sg,pos=a.pos0+t0+t,j0=kb+lane,j1=j0+32,j2=j0+64,j3=j0+96;
        bool row=t0+t<a.batch,v0=row&&j0<a.span&&(!a.causal||j0<=pos),v1=row&&j1<a.span&&(!a.causal||j1<=pos),
             v2=row&&j2<a.span&&(!a.causal||j2<=pos),v3=row&&j3<a.span&&(!a.causal||j3<=pos);
        float s0=v0?scores[t*KT+lane]:-INFINITY,s1=v1?scores[t*KT+32+lane]:-INFINITY,
              s2=v2?scores[t*KT+64+lane]:-INFINITY,s3=v3?scores[t*KT+96+lane]:-INFINITY;
        float tile_m=simd_max(max(max(s0,s1),max(s2,s3))),new_m=max(running_m[t],tile_m);
        float scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f;
        float e0=v0?exp(s0-new_m):0.0f,e1=v1?exp(s1-new_m):0.0f,
              e2=v2?exp(s2-new_m):0.0f,e3=v3?exp(s3-new_m):0.0f;
        scores[t*KT+lane]=e0;scores[t*KT+32+lane]=e1;
        scores[t*KT+64+lane]=e2;scores[t*KT+96+lane]=e3;
        float tile_l=simd_sum(e0+e1+e2+e3);
        if(lane==0){alpha[t]=scale;running_m[t]=new_m;running_l[t]=scale*running_l[t]+tile_l;}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(virtual_block){
            if(tid<QT)diag[tid*QT+tid]=alpha[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 scale_m,zero=make_filled_simdgroup_matrix<float,8>(0.0f);
            simdgroup_load(scale_m,diag,QT,0,false);
            for(uint col=0;col<8;++col){
                simdgroup_float8x8 scaled;
                simdgroup_multiply_accumulate(scaled,scale_m,acc[col],zero);
                acc[col]=scaled;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for(uint k=0;k<KT;k+=8){
            simdgroup_float8x8 ps;
            simdgroup_load(ps,scores+k,KT,0,false);
            device const half*pv=vc+(kh*a.capacity+kb+k)*DIM+sg*64;
            for(uint col=0;col<8;++col){
                simdgroup_half8x8 mv;
                simdgroup_load(mv,pv+col*8,DIM,0,false);
                simdgroup_multiply_accumulate(acc[col],ps,mv,acc[col]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid<QT)diag[tid*QT+tid]=(t0+tid<a.batch&&running_l[tid]>0.0f)?1.0f/running_l[tid]:0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 norm_m,zero=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_load(norm_m,diag,QT,0,false);
    threadgroup float*scratch=(threadgroup float*)tile;
    for(uint col=0;col<8;++col){
        simdgroup_float8x8 normalized;
        simdgroup_multiply_accumulate(normalized,norm_m,acc[col],zero);
        simdgroup_store(normalized,scratch+sg*64+col*8,DIM,0,false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<QT*DIM;i+=256){
        uint t=i/DIM,d=i%DIM;
        if(t0+t<a.batch)out[((t0+t)*a.heads+h)*DIM+d]=half(scratch[i]);
    }
}

kernel void gemma_flash_online_causal_q16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){    
    constexpr uint QT=16,KT=32,DK=32;uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads),base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;threadgroup half*sq=tile;threadgroup half*sk=sq+QT*a.dim;threadgroup float*scores=(threadgroup float*)(sk+KT*DK);threadgroup float*state=scores+QT*KT;threadgroup float*running_m=state;threadgroup float*running_l=state+QT;threadgroup float*alpha=state+2*QT;float ov[QT*2];for(uint i=0;i<QT*2;++i)ov[i]=0.0f;if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}for(uint i=tid;i<QT*a.dim;i+=256){uint t=i/a.dim,d=i%a.dim;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d]):half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=min(t0+QT,a.batch)-1,tile_span=a.causal?min(a.span,a.pos0+last+1-base):a.span;for(uint kb=0;kb<tile_span;kb+=KT){simdgroup_float8x8 qk[2];for(uint i=0;i<2;++i)qk[i]=make_filled_simdgroup_matrix<float,8>(0.0f);for(uint d0=0;d0<a.dim;d0+=DK){for(uint i=tid;i<KT*DK;i+=256){uint k=i/DK,d=i%DK,j=kb+k;if(j<tile_span){uint slot=(base+j)%a.capacity;sk[i]=kc[(kh*a.capacity+slot)*a.dim+d0+d];}else sk[i]=half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);if(sg<4)for(uint dk=0;dk<DK;dk+=8){simdgroup_half8x8 mk;simdgroup_load(mk,sk+(sg*8)*DK+dk,DK,0,true);for(uint qb=0;qb<2;++qb){simdgroup_half8x8 mq;simdgroup_load(mq,sq+qb*8*a.dim+d0+dk,a.dim,0,false);simdgroup_multiply_accumulate(qk[qb],mq,mk,qk[qb]);}}threadgroup_barrier(mem_flags::mem_threadgroup);}if(sg<4)for(uint qb=0;qb<2;++qb)simdgroup_store(qk[qb],scores+qb*8*KT+sg*8,KT,0,false);threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint pair=0;pair<2;++pair){uint t=2*sg+pair,j=kb+lane,pos=a.pos0+t0+t,p=base+j,limit=a.causal?pos:a.pos0+a.batch-1;bool valid=t0+t<a.batch&&j<tile_span&&(!a.causal||p<=pos)&&(!a.window||p+a.window>limit);float s=valid?scores[t*KT+lane]:-INFINITY,tile_m=simd_max(s),new_m=max(running_m[t],tile_m),scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f,e=valid?exp(s-new_m):0.0f;scores[t*KT+lane]=e;float tile_l=simd_sum(e);if(lane==0){alpha[t]=scale;running_m[t]=new_m;running_l[t]=scale*running_l[t]+tile_l;}}threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint r=0;r<2;++r){uint d=tid+r*256;if(d>=a.dim)continue;for(uint t=0;t<QT;++t)ov[t*2+r]*=alpha[t];for(uint k=0;k<KT&&kb+k<tile_span;++k){uint slot=(base+kb+k)%a.capacity;float v=float(vc[(kh*a.capacity+slot)*a.dim+d]);for(uint t=0;t<QT;++t)ov[t*2+r]+=scores[t*KT+k]*v;}}threadgroup_barrier(mem_flags::mem_threadgroup);
    }for(uint r=0;r<2;++r){uint d=tid+r*256;if(d<a.dim)for(uint t=0;t<QT&&t0+t<a.batch;++t)out[((t0+t)*a.heads+h)*a.dim+d]=ov[t*2+r]/running_l[t];}
}

// Gemma 4 sliding-window heads are 256-wide. One thread therefore owns one
// output channel, allowing a 32-query tile without increasing its PV register
// footprint over the generic Q16/D512 kernel.
template<uint QT,bool CACHE_Q,typename OUT> inline void flash_online_causal_swa_impl(device const float*q,device const half*kc,device const half*vc,device OUT*out,constant FlashArgs&a,uint group,ushort tid,ushort lane,ushort sg,threadgroup half*tile){
    constexpr uint KT=32,DK=32,QB=QT/8,ACTIVE=QT/4;uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads),global_base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0,base=a.window&&a.pos0+t0>=a.window?a.pos0+t0-a.window+1:0,local_span=a.span-(base-global_base);threadgroup half*sq=tile;threadgroup half*sk=sq+(CACHE_Q?QT*a.dim:QT*DK);threadgroup float*scores=(threadgroup float*)(sk+KT*DK);threadgroup float*state=scores+QT*KT;threadgroup float*running_m=state;threadgroup float*running_l=state+QT;threadgroup float*alpha=state+2*QT;float ov[QT];for(uint i=0;i<QT;++i)ov[i]=0.0f;if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}if(CACHE_Q)for(uint i=tid;i<QT*a.dim;i+=256){uint t=i/a.dim,d=i%a.dim;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d]):half(0);}threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=min(t0+QT,a.batch)-1,tile_span=a.causal?min(local_span,a.pos0+last+1-base):local_span;for(uint kb=0;kb<tile_span;kb+=KT){uint block_slot=(base+kb)%a.capacity;bool wraps_block=block_slot+KT>a.capacity;simdgroup_float8x8 qk[QB];for(uint i=0;i<QB;++i)qk[i]=make_filled_simdgroup_matrix<float,8>(0.0f);for(uint d0=0;d0<a.dim;d0+=DK){if(!CACHE_Q)for(uint i=tid;i<QT*DK;i+=256){uint t=i/DK,d=i%DK;sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*a.dim+d0+d]):half(0);}if(wraps_block)for(uint i=tid;i<KT*DK;i+=256){uint k=i/DK,d=i%DK,j=kb+k,slot=block_slot+k;if(slot>=a.capacity)slot-=a.capacity;sk[i]=j<tile_span?kc[(kh*a.capacity+slot)*a.dim+d0+d]:half(0);}if(!CACHE_Q||wraps_block)threadgroup_barrier(mem_flags::mem_threadgroup);if(sg<4)for(uint dk=0;dk<DK;dk+=8){simdgroup_half8x8 mk;if(wraps_block)simdgroup_load(mk,sk+(sg*8)*DK+dk,DK,0,true);else simdgroup_load(mk,kc+(kh*a.capacity+block_slot+sg*8)*a.dim+d0+dk,a.dim,0,true);for(uint qb=0;qb<QB;++qb){simdgroup_half8x8 mq;simdgroup_load(mq,sq+qb*8*(CACHE_Q?a.dim:DK)+(CACHE_Q?d0:0)+dk,CACHE_Q?a.dim:DK,0,false);simdgroup_multiply_accumulate(qk[qb],mq,mk,qk[qb]);}}if(!CACHE_Q||wraps_block)threadgroup_barrier(mem_flags::mem_threadgroup);}if(sg<4)for(uint qb=0;qb<QB;++qb)simdgroup_store(qk[qb],scores+qb*8*KT+sg*8,KT,0,false);threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg<ACTIVE)for(uint quad=0;quad<4;++quad){uint t=4*sg+quad,j=kb+lane,pos=a.pos0+t0+t,p=base+j,limit=a.causal?pos:a.pos0+a.batch-1;bool valid=t0+t<a.batch&&j<tile_span&&(!a.causal||p<=pos)&&p+a.window>limit;float s=valid?scores[t*KT+lane]:-INFINITY,tile_m=simd_max(s),new_m=max(running_m[t],tile_m),scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f,e=valid?exp(s-new_m):0.0f;scores[t*KT+lane]=e;float tile_l=simd_sum(e);if(lane==0){alpha[t]=scale;running_m[t]=new_m;running_l[t]=scale*running_l[t]+tile_l;}}threadgroup_barrier(mem_flags::mem_threadgroup);
        uint d=tid;for(uint t=0;t<QT;++t)ov[t]*=alpha[t];for(uint k=0;k<KT&&kb+k<tile_span;++k){uint slot=block_slot+k;if(slot>=a.capacity)slot-=a.capacity;float v=float(vc[(kh*a.capacity+slot)*a.dim+d]);for(uint t=0;t<QT;++t)ov[t]+=scores[t*KT+k]*v;}threadgroup_barrier(mem_flags::mem_threadgroup);
    }uint d=tid;for(uint t=0;t<QT&&t0+t<a.batch;++t)out[((t0+t)*a.heads+h)*a.dim+d]=OUT(ov[t]/running_l[t]);
}

kernel void gemma_flash_online_causal_swa_q32(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_online_causal_swa_impl<32,true,float>(q,kc,vc,out,a,group,tid,lane,sg,tile);}
kernel void gemma_flash_online_causal_swa_q32_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_online_causal_swa_impl<32,true,half>(q,kc,vc,out,a,group,tid,lane,sg,tile);}
kernel void gemma_flash_online_causal_swa_q32_reload_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_online_causal_swa_impl<32,false,half>(q,kc,vc,out,a,group,tid,lane,sg,tile);}
kernel void gemma_flash_online_causal_swa_q16_compact_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){flash_online_causal_swa_impl<16,true,half>(q,kc,vc,out,a,group,tid,lane,sg,tile);}

// Local-base SWA variant with a 64-key tile.  Unlike the older generic K64
// diagnostic it trims keys per 32-query tile before QK/PV, preserving the
// bounded sliding-window work of the production K32 kernel while halving the
// number of online-softmax synchronization rounds.
kernel void gemma_flash_online_causal_swa_q32_k64_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=32,KT=64,DK=32,QB=4,DIM=256;
    uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads);
    uint global_base=a.pos0>=a.window?a.pos0-a.window+1:0;
    uint base=a.pos0+t0>=a.window?a.pos0+t0-a.window+1:0;
    uint local_span=a.span-(base-global_base);
    threadgroup half*sq=tile;
    threadgroup half*sk=sq+QT*DIM;
    threadgroup float*scores=(threadgroup float*)(sk+KT*DK);
    threadgroup float*state=scores+QT*KT;
    threadgroup float*running_m=state;
    threadgroup float*running_l=state+QT;
    threadgroup float*alpha=state+2*QT;
    float ov[QT];
    for(uint i=0;i<QT;++i)ov[i]=0.0f;
    if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}
    for(uint i=tid;i<QT*DIM;i+=256){
        uint t=i/DIM,d=i%DIM;
        sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*DIM+d]):half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=min(t0+QT,a.batch)-1;
    uint tile_span=min(local_span,a.pos0+last+1-base);
    uint logical_blocks=(tile_span+KT-1)/KT;
    uint sink_blocks=min(logical_blocks,a.sparse_sink_blocks);
    uint recent_count=min(logical_blocks,a.sparse_recent_block);
    uint recent_start=max(sink_blocks,logical_blocks-recent_count);
    uint sampled_blocks=(recent_start-sink_blocks+a.sparse_stride-1)/
                        max(1u,a.sparse_stride);
    uint selected_blocks=sink_blocks+sampled_blocks+
                         (logical_blocks-recent_start);
    bool sparse=a.sparse_selected_blocks&&a.sparse_stride>1&&
                selected_blocks<logical_blocks;
    uint executed_blocks=sparse?selected_blocks:logical_blocks;
    for(uint virtual_block=0;virtual_block<executed_blocks;++virtual_block){
        uint physical_block=virtual_block;
        if(sparse){
            if(virtual_block<sink_blocks)physical_block=virtual_block;
            else if(virtual_block<sink_blocks+sampled_blocks)
                physical_block=sink_blocks+
                    (virtual_block-sink_blocks)*a.sparse_stride;
            else physical_block=recent_start+
                    (virtual_block-sink_blocks-sampled_blocks);
        }
        uint kb=physical_block*KT;
        uint block_slot=(base+kb)%a.capacity;
        bool wraps_block=block_slot+KT>a.capacity;
        simdgroup_float8x8 qk[QB];
        for(uint i=0;i<QB;++i)qk[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
        for(uint d0=0;d0<DIM;d0+=DK){
            if(wraps_block)for(uint i=tid;i<KT*DK;i+=256){
                uint k=i/DK,d=i%DK,j=kb+k,slot=block_slot+k;
                if(slot>=a.capacity)slot-=a.capacity;
                sk[i]=j<tile_span?kc[(kh*a.capacity+slot)*DIM+d0+d]:half(0);
            }
            if(wraps_block)threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint dk=0;dk<DK;dk+=8){
                simdgroup_half8x8 mk,mq[QB];
                if(wraps_block)simdgroup_load(mk,sk+sg*8*DK+dk,DK,0,true);
                else simdgroup_load(mk,kc+(kh*a.capacity+block_slot+sg*8)*DIM+d0+dk,DIM,0,true);
                for(uint qb=0;qb<QB;++qb){
                    simdgroup_load(mq[qb],sq+qb*8*DIM+d0+dk,DIM,0,false);
                    simdgroup_multiply_accumulate(qk[qb],mq[qb],mk,qk[qb]);
                }
            }
            if(wraps_block)threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint qb=0;qb<QB;++qb)
            simdgroup_store(qk[qb],scores+qb*8*KT+sg*8,KT,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint quad=0;quad<4;++quad){
            uint t=4*sg+quad,j0=kb+lane,j1=j0+32,pos=a.pos0+t0+t;
            uint p0=base+j0,p1=base+j1;
            bool row=t0+t<a.batch;
            bool v0=row&&j0<tile_span&&p0<=pos&&p0+a.window>pos;
            bool v1=row&&j1<tile_span&&p1<=pos&&p1+a.window>pos;
            float s0=v0?scores[t*KT+lane]:-INFINITY;
            float s1=v1?scores[t*KT+32+lane]:-INFINITY;
            float tile_m=simd_max(max(s0,s1)),new_m=max(running_m[t],tile_m);
            float scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f;
            float e0=v0?exp(s0-new_m):0.0f,e1=v1?exp(s1-new_m):0.0f;
            scores[t*KT+lane]=e0;
            scores[t*KT+32+lane]=e1;
            float tile_l=simd_sum(e0+e1);
            if(lane==0){alpha[t]=scale;running_m[t]=new_m;running_l[t]=scale*running_l[t]+tile_l;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint d=tid;
        for(uint t=0;t<QT;++t)ov[t]*=alpha[t];
        for(uint k=0;k<KT&&kb+k<tile_span;++k){
            uint slot=block_slot+k;
            if(slot>=a.capacity)slot-=a.capacity;
            float v=float(vc[(kh*a.capacity+slot)*DIM+d]);
            for(uint t=0;t<QT;++t)ov[t]+=scores[t*KT+k]*v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint d=tid;
    for(uint t=0;t<QT&&t0+t<a.batch;++t)
        out[((t0+t)*a.heads+h)*DIM+d]=half(ov[t]/running_l[t]);
}

// llama.cpp-style SWA path specialized for Gemma's D=256 heads. Four SIMD
// groups share an 8x64 score tile and use matrix instructions for both QK and
// PV. The local base trims keys that cannot belong to any query in this tile.
kernel void gemma_flash_llama_swa_q8_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=8,KT=64,DK=32,DIM=256;
    uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads);
    uint global_base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;
    uint base=a.window&&a.pos0+t0>=a.window?a.pos0+t0-a.window+1:0;
    uint local_span=a.span-(base-global_base);
    threadgroup half*sq=tile;
    threadgroup half*sk=sq+QT*DIM;
    threadgroup float*scores=(threadgroup float*)(sk+KT*DK);
    threadgroup float*so=scores+QT*KT;
    threadgroup float*state=so+QT*DIM;
    threadgroup float*running_m=state;
    threadgroup float*running_l=state+QT;
    threadgroup float*alpha=state+2*QT;
    if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}
    for(uint i=tid;i<QT*DIM;i+=128){
        uint t=i/DIM,d=i%DIM;
        sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*DIM+d]):half(0);
        so[i]=0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=min(t0+QT,a.batch)-1;
    uint tile_span=a.causal?min(local_span,a.pos0+last+1-base):local_span;
    for(uint kb=0;kb<tile_span;kb+=KT){
        uint block_slot=(base+kb)%a.capacity;
        bool wraps_block=block_slot+KT>a.capacity;
        simdgroup_float8x8 qk[2];
        qk[0]=make_filled_simdgroup_matrix<float,8>(0.0f);
        qk[1]=make_filled_simdgroup_matrix<float,8>(0.0f);
        for(uint d0=0;d0<DIM;d0+=DK){
            if(wraps_block){
                for(uint i=tid;i<KT*DK;i+=128){
                    uint k=i/DK,d=i%DK,j=kb+k;
                    sk[i]=j<tile_span?kc[(kh*a.capacity+(base+j)%a.capacity)*DIM+d0+d]:half(0);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for(uint dk=0;dk<DK;dk+=8){
                simdgroup_half8x8 mq,mk0,mk1;
                simdgroup_load(mq,sq+d0+dk,DIM,0,false);
                if(wraps_block){
                    simdgroup_load(mk0,sk+(sg*16)*DK+dk,DK,0,true);
                    simdgroup_load(mk1,sk+(sg*16+8)*DK+dk,DK,0,true);
                }else{
                    device const half*pk=kc+(kh*a.capacity+block_slot+sg*16)*DIM+d0+dk;
                    simdgroup_load(mk0,pk,DIM,0,true);
                    simdgroup_load(mk1,pk+8*DIM,DIM,0,true);
                }
                simdgroup_multiply_accumulate(qk[0],mq,mk0,qk[0]);
                simdgroup_multiply_accumulate(qk[1],mq,mk1,qk[1]);
            }
            if(wraps_block)threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(qk[0],scores+sg*16,KT,0,false);
        simdgroup_store(qk[1],scores+sg*16+8,KT,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for(uint pair=0;pair<2;++pair){
            uint t=2*sg+pair,j0=kb+lane,j1=j0+32,pos=a.pos0+t0+t,p0=base+j0,p1=base+j1;
            bool row=t0+t<a.batch;
            bool v0=row&&j0<tile_span&&p0<=pos&&p0+a.window>pos;
            bool v1=row&&j1<tile_span&&p1<=pos&&p1+a.window>pos;
            float s0=v0?scores[t*KT+lane]:-INFINITY;
            float s1=v1?scores[t*KT+32+lane]:-INFINITY;
            float tile_m=simd_max(max(s0,s1));
            float new_m=max(running_m[t],tile_m);
            float scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f;
            float e0=v0?exp(s0-new_m):0.0f,e1=v1?exp(s1-new_m):0.0f;
            scores[t*KT+lane]=e0;
            scores[t*KT+32+lane]=e1;
            float tile_l=simd_sum(e0+e1);
            if(lane==0){
                alpha[t]=scale;
                running_m[t]=new_m;
                running_l[t]=scale*running_l[t]+tile_l;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<QT*DIM;i+=128)so[i]*=alpha[i/DIM];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc[8];
        for(uint col=0;col<8;++col)simdgroup_load(acc[col],so+sg*64+col*8,DIM,0,false);
        for(uint k=0;k<KT;k+=8){
            simdgroup_float8x8 ps;
            simdgroup_load(ps,scores+k,KT,0,false);
            uint slot=(base+kb+k)%a.capacity;
            bool wraps=slot+7>=a.capacity;
            if(wraps){
                for(uint i=tid;i<8*DIM;i+=128){
                    uint key=i/DIM,d=i%DIM;
                    sk[i]=vc[(kh*a.capacity+(slot+key)%a.capacity)*DIM+d];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            device const half*pv=vc+(kh*a.capacity+slot)*DIM+sg*64;
            for(uint col=0;col<8;++col){
                simdgroup_half8x8 mv;
                if(wraps)simdgroup_load(mv,sk+sg*64+col*8,DIM,0,false);
                else simdgroup_load(mv,pv+col*8,DIM,0,false);
                simdgroup_multiply_accumulate(acc[col],ps,mv,acc[col]);
            }
            if(wraps)threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint col=0;col<8;++col)simdgroup_store(acc[col],so+sg*64+col*8,DIM,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=tid;i<QT*DIM;i+=128){
        uint t=i/DIM,d=i%DIM;
        if(t0+t<a.batch)out[((t0+t)*a.heads+h)*DIM+d]=half(so[i]/running_l[t]);
    }
}

// Q16 register-resident variant: P*V never spills between key tiles. The
// threadgroup scratch is reused for the final FP32 normalization/conversion.
kernel void gemma_flash_llama_swa_q16_register_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device half*out[[buffer(3)]],constant FlashArgs&a[[buffer(4)]],uint group[[threadgroup_position_in_grid]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],threadgroup half*tile[[threadgroup(0)]]){
    constexpr uint QT=16,KT=64,DK=32,DIM=256,QB=2;
    uint h=group%a.heads,t0=(group/a.heads)*QT,kh=h/(a.heads/a.kv_heads);
    uint global_base=a.window&&a.pos0>=a.window?a.pos0-a.window+1:0;
    uint base=a.window&&a.pos0+t0>=a.window?a.pos0+t0-a.window+1:0;
    uint local_span=a.span-(base-global_base);
    threadgroup half*sq=tile;
    threadgroup half*sk=sq+QT*DIM;
    threadgroup float*scores=(threadgroup float*)(sk+KT*DK);
    threadgroup float*diag=scores+QT*KT;
    threadgroup float*state=diag+QB*64;
    threadgroup float*running_m=state;
    threadgroup float*running_l=state+QT;
    threadgroup float*alpha=state+2*QT;
    if(tid<QT){running_m[tid]=-INFINITY;running_l[tid]=0.0f;}
    for(uint i=tid;i<QB*64;i+=128)diag[i]=0.0f;
    for(uint i=tid;i<QT*DIM;i+=128){
        uint t=i/DIM,d=i%DIM;
        sq[i]=t0+t<a.batch?half(q[((t0+t)*a.heads+h)*DIM+d]):half(0);
    }
    simdgroup_float8x8 acc[QB*8];
    for(uint i=0;i<QB*8;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=min(t0+QT,a.batch)-1;
    uint tile_span=a.causal?min(local_span,a.pos0+last+1-base):local_span;
    uint block_index=0;
    for(uint kb=0;kb<tile_span;kb+=KT,++block_index){
        uint block_slot=(base+kb)%a.capacity;
        bool wraps_block=block_slot+KT>a.capacity;
        simdgroup_float8x8 qk[QB*2];
        for(uint i=0;i<QB*2;++i)qk[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
        for(uint d0=0;d0<DIM;d0+=DK){
            if(wraps_block){
                for(uint i=tid;i<KT*DK;i+=128){
                    uint k=i/DK,d=i%DK,j=kb+k;
                    sk[i]=j<tile_span?kc[(kh*a.capacity+(base+j)%a.capacity)*DIM+d0+d]:half(0);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for(uint dk=0;dk<DK;dk+=8){
                simdgroup_half8x8 mq[QB],mk[2];
                for(uint qb=0;qb<QB;++qb)simdgroup_load(mq[qb],sq+qb*8*DIM+d0+dk,DIM,0,false);
                if(wraps_block){
                    simdgroup_load(mk[0],sk+(sg*16)*DK+dk,DK,0,true);
                    simdgroup_load(mk[1],sk+(sg*16+8)*DK+dk,DK,0,true);
                }else{
                    device const half*pk=kc+(kh*a.capacity+block_slot+sg*16)*DIM+d0+dk;
                    simdgroup_load(mk[0],pk,DIM,0,true);
                    simdgroup_load(mk[1],pk+8*DIM,DIM,0,true);
                }
                for(uint qb=0;qb<QB;++qb){
                    simdgroup_multiply_accumulate(qk[qb*2],mq[qb],mk[0],qk[qb*2]);
                    simdgroup_multiply_accumulate(qk[qb*2+1],mq[qb],mk[1],qk[qb*2+1]);
                }
            }
            if(wraps_block)threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint qb=0;qb<QB;++qb){
            simdgroup_store(qk[qb*2],scores+qb*8*KT+sg*16,KT,0,false);
            simdgroup_store(qk[qb*2+1],scores+qb*8*KT+sg*16+8,KT,0,false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint quad=0;quad<4;++quad){
            uint t=4*sg+quad,j0=kb+lane,j1=j0+32,pos=a.pos0+t0+t,p0=base+j0,p1=base+j1;
            bool row=t0+t<a.batch;
            bool v0=row&&j0<tile_span&&p0<=pos&&p0+a.window>pos;
            bool v1=row&&j1<tile_span&&p1<=pos&&p1+a.window>pos;
            float s0=v0?scores[t*KT+lane]:-INFINITY,s1=v1?scores[t*KT+32+lane]:-INFINITY;
            float tile_m=simd_max(max(s0,s1)),new_m=max(running_m[t],tile_m);
            float scale=isfinite(running_m[t])?exp(running_m[t]-new_m):0.0f;
            float e0=v0?exp(s0-new_m):0.0f,e1=v1?exp(s1-new_m):0.0f;
            scores[t*KT+lane]=e0;scores[t*KT+32+lane]=e1;
            float tile_l=simd_sum(e0+e1);
            if(lane==0){alpha[t]=scale;running_m[t]=new_m;running_l[t]=scale*running_l[t]+tile_l;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(block_index){
            if(tid<QT){uint qb=tid/8,r=tid%8;diag[qb*64+r*8+r]=alpha[tid];}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint qb=0;qb<QB;++qb){
                simdgroup_float8x8 scale_m,zero=make_filled_simdgroup_matrix<float,8>(0.0f);
                simdgroup_load(scale_m,diag+qb*64,8,0,false);
                for(uint col=0;col<8;++col){
                    simdgroup_float8x8 scaled;
                    simdgroup_multiply_accumulate(scaled,scale_m,acc[qb*8+col],zero);
                    acc[qb*8+col]=scaled;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint k=0;k<KT;k+=8){
            simdgroup_float8x8 ps[QB];
            for(uint qb=0;qb<QB;++qb)simdgroup_load(ps[qb],scores+qb*8*KT+k,KT,0,false);
            uint slot=(base+kb+k)%a.capacity;
            bool wraps=slot+7>=a.capacity;
            if(wraps){
                for(uint i=tid;i<8*DIM;i+=128){
                    uint key=i/DIM,d=i%DIM;
                    sk[i]=vc[(kh*a.capacity+(slot+key)%a.capacity)*DIM+d];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            device const half*pv=vc+(kh*a.capacity+slot)*DIM+sg*64;
            for(uint col=0;col<8;++col){
                simdgroup_half8x8 mv;
                if(wraps)simdgroup_load(mv,sk+sg*64+col*8,DIM,0,false);
                else simdgroup_load(mv,pv+col*8,DIM,0,false);
                for(uint qb=0;qb<QB;++qb)simdgroup_multiply_accumulate(acc[qb*8+col],ps[qb],mv,acc[qb*8+col]);
            }
            if(wraps)threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid<QT){uint qb=tid/8,r=tid%8;diag[qb*64+r*8+r]=(t0+tid<a.batch&&running_l[tid]>0.0f)?1.0f/running_l[tid]:0.0f;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float*scratch=(threadgroup float*)tile;
    for(uint qb=0;qb<QB;++qb){
        simdgroup_float8x8 norm_m,zero=make_filled_simdgroup_matrix<float,8>(0.0f);
        simdgroup_load(norm_m,diag+qb*64,8,0,false);
        for(uint col=0;col<8;++col){
            simdgroup_float8x8 normalized;
            simdgroup_multiply_accumulate(normalized,norm_m,acc[qb*8+col],zero);
            simdgroup_store(normalized,scratch+qb*8*DIM+sg*64+col*8,DIM,0,false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<QT*DIM;i+=128){
        uint t=i/DIM,d=i%DIM;
        if(t0+t<a.batch)out[((t0+t)*a.heads+h)*DIM+d]=half(scratch[i]);
    }
}

kernel void gemma_scale_scalar(device float*x[[buffer(0)]],constant float&s[[buffer(1)]],uint i[[thread_position_in_grid]]){x[i]*=s;}
kernel void gemma_softcap(device float*x[[buffer(0)]],constant float&cap[[buffer(1)]],uint i[[thread_position_in_grid]]){x[i]=cap*precise::tanh(x[i]/cap);}
