#include "neutron/native/tokenizer.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace neutron::native {
namespace {
size_t utf8_len(unsigned char c) { if ((c&0x80)==0)return 1; if((c&0xE0)==0xC0)return 2; if((c&0xF0)==0xE0)return 3; if((c&0xF8)==0xF0)return 4; return 1; }
std::string rank_key(const std::string & a,const std::string & b) { return a+'\0'+b; }
void replace_spaces(std::string & s) { size_t p=0; while((p=s.find(' ',p))!=std::string::npos){s.replace(p,1,"\xE2\x96\x81");p+=3;} }
void unescape_spaces(std::string & s) { size_t p=0; while((p=s.find("\xE2\x96\x81",p))!=std::string::npos){s.replace(p,3," ");++p;} }
} // namespace

Gemma4Tokenizer::Gemma4Tokenizer(const GGUF & g) {
    tokens_=g.meta<std::vector<std::string>>("tokenizer.ggml.tokens");
    const auto & raw_types=g.meta<std::vector<int32_t>>("tokenizer.ggml.token_type"); types_=raw_types;
    const auto & merges=g.meta<std::vector<std::string>>("tokenizer.ggml.merges");
    ids_.reserve(tokens_.size()*2); for(size_t i=0;i<tokens_.size();++i) ids_.emplace(tokens_[i],static_cast<int32_t>(i));
    ranks_.reserve(merges.size()*2);
    for(size_t i=0;i<merges.size();++i){const auto p=merges[i].find(' ',1);if(p!=std::string::npos)ranks_.emplace(rank_key(merges[i].substr(0,p),merges[i].substr(p+1)),i);}
    bos_=static_cast<int32_t>(g.meta_u64("tokenizer.ggml.bos_token_id")); eos_=static_cast<int32_t>(g.meta_u64("tokenizer.ggml.eos_token_id"));
    for(size_t i=0;i<tokens_.size();++i){const auto &s=tokens_[i];const bool protocol=s.size()>=4&&s.front()=='<'&&s.back()=='>'&&(s[1]=='|'||s[s.size()-2]=='|');if((i<types_.size()&&types_[i]==3)||protocol||s=="<turn|>")specials_.push_back({s,static_cast<int32_t>(i)});}
    std::sort(specials_.begin(),specials_.end(),[](const auto&a,const auto&b){return a.first.size()>b.first.size();});
}

int32_t Gemma4Tokenizer::token_id(const std::string & p) const { auto i=ids_.find(p);return i==ids_.end()?-1:i->second; }
bool Gemma4Tokenizer::is_eog(int32_t t) const { return t==eos_||t==token_id("<turn|>")||t==token_id("<|tool_response>"); }

std::vector<int32_t> Gemma4Tokenizer::encode(const std::string & text,bool add_bos) const {
    std::vector<int32_t> out; if(add_bos)out.push_back(bos_);
    size_t plain=0,pos=0;
    while(pos<text.size()){
        bool matched=false;
        if(text[pos]=='<')for(const auto &[s,id]:specials_)if(text.compare(pos,s.size(),s)==0){if(pos>plain)encode_plain(text.substr(plain,pos-plain),out);out.push_back(id);pos+=s.size();plain=pos;matched=true;break;}
        if(!matched)++pos;
    }
    if(plain<text.size())encode_plain(text.substr(plain),out);
    return out;
}

void Gemma4Tokenizer::encode_plain(std::string text,std::vector<int32_t>&out) const {
    replace_spaces(text);
    size_t p=0;
    while(p<text.size()){
        const bool nl=text[p]=='\n';size_t e=p+1;while(e<text.size()&&(text[e]=='\n')==nl)++e;
        encode_run(text.substr(p,e-p),out);p=e;
    }
}

void Gemma4Tokenizer::encode_run(const std::string & run,std::vector<int32_t>&out) const {
    if(run.empty())return;
    if(run.find_first_not_of('\n')==std::string::npos){auto i=ids_.find(run);if(i!=ids_.end()){out.push_back(i->second);return;}}
    struct Sym{std::string text;int prev=-1,next=-1;bool live=true;};
    std::vector<Sym>s;
    for(size_t p=0;p<run.size();){size_t n=std::min(utf8_len(static_cast<unsigned char>(run[p])),run.size()-p);s.push_back({run.substr(p,n),static_cast<int>(s.size())-1,-1,true});if(s.size()>1)s[s.size()-2].next=s.size()-1;p+=n;}
    struct Pair{int rank,left,right;bool operator<(const Pair&o)const{return rank>o.rank||(rank==o.rank&&left>o.left);}};
    std::priority_queue<Pair>q;
    auto add=[&](int l,int r){if(l<0||r<0||!s[l].live||!s[r].live)return;auto i=ranks_.find(rank_key(s[l].text,s[r].text));if(i!=ranks_.end())q.push({i->second,l,r});};
    for(size_t i=1;i<s.size();++i)add(i-1,i);
    while(!q.empty()){
        auto [rank,l,r]=q.top();q.pop();(void)rank;
        if(!s[l].live||!s[r].live||s[l].next!=r)continue;
        auto check=ranks_.find(rank_key(s[l].text,s[r].text));if(check==ranks_.end()||check->second!=rank)continue;
        s[l].text+=s[r].text;s[r].live=false;s[l].next=s[r].next;if(s[r].next>=0)s[s[r].next].prev=l;add(s[l].prev,l);add(l,s[l].next);
    }
    for(const auto &x:s)if(x.live){auto i=ids_.find(x.text);if(i!=ids_.end())out.push_back(i->second);else for(unsigned char c:x.text){char h[]="<0x00>";const char*hex="0123456789ABCDEF";h[3]=hex[c>>4];h[4]=hex[c&15];auto b=ids_.find(h);if(b!=ids_.end())out.push_back(b->second);else throw std::runtime_error("Gemma4 tokenizer byte fallback missing");}}
}

std::string Gemma4Tokenizer::decode(int32_t t,bool special) const {
    if(t<0||static_cast<size_t>(t)>=tokens_.size())throw std::runtime_error("token id outside vocabulary");
    if(!special&&t<static_cast<int32_t>(types_.size())&&types_[t]==3)return {};
    std::string p=tokens_[t];
    if(p.size()==6&&p.rfind("<0x",0)==0&&p[5]=='>'){auto v=[](char c){return c>='A'?c-'A'+10:c-'0';};return std::string(1,char((v(p[3])<<4)|v(p[4])));}
    unescape_spaces(p);return p;
}
std::string Gemma4Tokenizer::decode(const std::vector<int32_t>&ts,bool sp) const {std::string o;for(auto t:ts)o+=decode(t,sp);return o;}

} // namespace neutron::native
