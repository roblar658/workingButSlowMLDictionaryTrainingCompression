"""
C++ Solver Generator Module
===========================
Emits standalone, self-contained, high-performance C++20 solver (solve.cpp).
"""

from compressor.codegen.layout import compute_solver_layout


class CppSolverGenerator:
    """Emits standalone C++20 dictionary decompressor source code."""

    @staticmethod
    def generate(
        u_str: str,
        x_end: int,
        w_bytes_len: int,
        n_fsm: int = 80,
        f1_k: int = 4,
        m3b: int = 18,
        n_tot_ctx: int = 143,
        sig_pad: int = 2048,
        sig_cut: int = 850,
        denom: int = 96,
        n_ch: int = 53,
        n_masks: int = 200,
        n_up_inf: int = 4,
        n_base: int = 35364,
        total_words: int = 102400,
        total_out_bytes: int = 1000000,
        max_bchars: int = 48,
        max_wbytes: int = 64,
        blob_len: int = 76000,
        meta_raw_len: int = 14000,
        use_h2_hash: bool = False,
        cb_shift: int = 6,
        h_stride: int = 64,
        ac0: int = 48,
        aci: int = 32,
        ex_cap: int = 128,
    ) -> str:
        """Generate standalone C++20 decompressor source code."""
        lay = compute_solver_layout(
            u_str=u_str,
            w_bytes_len=w_bytes_len,
            n_fsm=n_fsm,
            m3b=m3b,
            n_tot_ctx=n_tot_ctx,
            n_ch=n_ch,
            n_masks=n_masks,
            n_up_inf=n_up_inf,
            total_words=total_words,
            total_out_bytes=total_out_bytes,
            max_bchars=max_bchars,
            max_wbytes=max_wbytes,
            blob_len=blob_len,
            meta_raw_len=meta_raw_len,
            use_h2_hash=use_h2_hash,
            ex_cap=ex_cap,
        )
        stride = lay["stride"]
        m3 = lay["m3"]
        mask3 = lay["mask3"]
        eow = lay["eow"]
        bos = lay["bos"]
        s_stride = lay["s_stride"]
        h0 = lay["h0"]
        n_ch_safe = lay["n_ch_safe"]
        n_masks_safe = lay["n_masks_safe"]
        ex_rows = lay["ex_rows"]
        m2_size = lay["m2_size"]
        b2_expr = (
            f"{{V h=((c2<<{cb_shift})^c1^0x7777u)*0x9E3779B1u;b2=(h^(h>>16))&262143u;}}"
            if use_h2_hash
            else f"b2=(c2*{s_stride}+c1)*{h0};"
        )
        w_buf = lay["w_buf"]
        o_buf = lay["o_buf"]
        nw_cap = lay["nw_cap"]
        len_bits = lay["len_bits"]
        len_mask = lay["len_mask"]
        id_bits = lay["id_bits"]
        id_mask_hex = lay["id_mask_hex"]
        cmp_start = lay["cmp_start"]
        ol_type = "V" if lay["is_ol_32"] else "J"
        mb_buf = lay["mb_buf"]
        an_start = lay["an_start"]
        b_buf = lay["b_buf"]
        cp_buf = lay["cp_buf"]
        wv_buf = lay["wv_buf"]
        db_buf = lay["db_buf"]
        pc_buf = lay["pc_buf"]
        wb_buf = lay["wb_buf"]

        cpp_code = (
f"""#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
using u8=uint8_t;using i8=int8_t;using i16=int16_t;using H=uint16_t;using V=uint32_t;using J=uint64_t;using D=char;
#define I int
#define _u const u8*
#define _c const D*
#define F(i,n) for(I i=0;i<(n);i++)
#define Z memcpy
#define X(a,b) ((a)<(b)?(a):(b))
#define _R return
#define _(x) if(x)
#define _W while
#define U2(p) ((p)[0]|((p)[1]<<8))
#define U4(p) (((V)(p)[0]<<24)|((V)(p)[1]<<16)|((V)(p)[2]<<8)|(p)[3])
#define DR(v,c) V v=RP[c][xp&255];xp=((v>>8)&4095)*(xp>>8)+(v&255);_W(xp<65536)xp=(xp<<8)|pb[po++];
#define ST(nd,lu,ru) k1=o1+nd;k2=b2+nd;k3=b3+nd;s1=F1[k1];s2=F2[k2];s3=F3[k3];rv=(R>>12)*S[T1[nd][s1]+T2[nd][s2]+T3[nd][s3]];{{V b=(L>=rv);L-=b?rv:0;R=b?R-rv:rv;u=b?(ru):(lu);_u n=nx+(b<<8);F1[k1]=n[s1];F2[k2]=n[s2];F3[k3]=n[s3];}}_W(R<1ULL<<24)R<<=8,L=(L<<8)|Qd[di++];
#define EC(t) pc[5+cln++]=t;Z(Wb+wl,CS+CO[t],CL[t]);wl+=CL[t];pbo[cln]=wl;
_c U=R"qQ({u_str})qQ",*A[46],*CS;u8 F1[{s_stride * f1_k * h_stride}],F2[{m2_size}],F3[{m3 + h0}],nx[512],MB[{mb_buf}],B[{b_buf}],C0[{s_stride * s_stride}],CL[{n_ch_safe}],UP[{s_stride}];D W[{w_buf}],O[{o_buf}];H S[{2 * sig_pad + 1}],WV[{wv_buf}],DB[{db_buf}],CO[{n_ch_safe}],IS[{s_stride * s_stride}],IL[{s_stride * s_stride}],EX[{ex_rows}][{s_stride}];i16 T1[{h0}][{stride}],T2[{h0}][{stride}],T3[{h0}][{stride}];{ol_type} OL[{nw_cap}];V RT[2048],RP[{max(1, n_tot_ctx)}][256],CP[{cp_buf}],CT[4096],C2[4096],lr[{h0}],rr[{h0}],vr[{h0 + 1}],mo[{n_masks_safe}],ao[37],AL[46],wp,nw;I ex;J PA[{nw_cap}],PX[{nw_cap}];_u T;
void ae(H*r,I i,I k,H v){{_(r[i]<512){{H d=r[i];I q=ex++;F(j,{s_stride})EX[q][j]=d;r[i]=512+q;}}EX[r[i]-512][k]=v;}}
void ae2(H*r,I i,I k,I m,H v){{_(r[i]<512){{H d=r[i];I q=ex++;F(j,{s_stride})EX[q][j]=d;r[i]=512+q;}}ae(EX[r[i]-512],k,m,v);}}
I rd(_u b,I mb,H*o){{I nu=U2(b),nl=U2(b+2),p=4,ps=0,mk=(1<<mb)-1,tn=0;F(i,nu){{I d=b[p++];_(d==255)d+=b[p++];ps+=d;I f=b[p++];_(f>=240)f=((f-240)<<8)|b[p++];V h=((V)ps<<(mb+12))|((V)f<<mb);F(j,f)RT[tn++]=h+j;}}V x=U4(b+p);p+=4;F(i,nl){{V v=RT[x&mk];x=((v>>mb)&4095)*(x>>mb)+(v&mk);_W(x<65536)x=(x<<8)|b[p++];o[i]=v>>(mb+12);}}_R nl;}}
void dz(_u b,u8*o){{I l1=U2(b),l2=U2(b+2),nt=rd(b+6,11,WV);_u e=b+6+l1;rd(e,8,DB);_u l_=e+l2,*ex=l_+U2(b+4);I di=0,li=0,bp=0,n=0;F(i,nt){{H s=WV[i];_(s<256)o[n++]=(u8)s;else{{I l=s==287?34+l_[li++]:3+s-256,d=DB[di++],q;_(d<2)q=d+1;else{{I g=d-1;q=1+(1<<g);F(k,g)q+=((ex[bp>>3]>>(bp&7))&1)<<k,bp++;}}F(k,l)o[n]=o[n-q],n++;}}}}}}
void ct(_u ln,I q,V*L,V*R,V*V0){{V nt=1,li=0,c=0,pl=0;for(I l=1;l<33;l++)F(s,q)_(ln[s]==l){{c<<=(l-pl);pl=l;V u=0;for(I b=l-1;b>0;b--){{V*t=(c>>b)&1?&R[u]:&L[u];_(!*t)*t=nt++;u=*t;}}(c&1?R:L)[u]=q-1+li;V0[li++]=s;c++;}}}}
void G(_c s,I l){{V o=wp;_(s!=W+o)Z(W+o,s,l);wp+=l;OL[nw]=(({ol_type})o<<{len_bits})|({ol_type})l;J k=0;Z(&k,W+o,8);PA[nw]=(__builtin_bswap64(k)&~{id_mask_hex}ULL)|nw;nw++;}}
I VV(D c){{_R c=='a'||c=='e'||c=='i'||c=='o'||c=='u';}}I Y(_c w,I l){{_R l>1&&w[l-1]=='y'&&!VV(w[l-2]);}}I M(D*d,_c w,I l,I k,I i){{I m=l-k;Z(d,w,m);Z(d+m,A[i],AL[i]);_R m+AL[i];}}I Q(D*d,I i,_c w,I l){{I m=AL[i];Z(d,A[i],m);Z(d+m,w,l);_R m+l;}}
I SF(I k,D*d,_c w,I l,I u0){{D e=w[l-1],f=l>1?w[l-2]:0;I y=Y(w,l),z=e=='e';
#define S_(x,i) _R M(d,w,l,x,i)
_(!k){{_(!u0){{_(e=='s'||e=='x'||e=='z'||(e=='h'&&(f=='c'||f=='s')))S_(0,39);_(y)S_(1,40);}}S_(0,41);}}_(k==1||k==3||k==4){{I j=k>>1;S_(z?0:y,z?33+j:y?36+j:j==0?7:j==1?10:11);}}_(k==2)S_(z&&!(l>1&&(f=='e'||f=='y'||f=='o')),8);_(k==5){{I t=l>2&&z&&f=='l'&&!VV(w[l-3]);S_(y||t,y?42:t?12:l>1&&f=='i'&&e=='c'?14:17);}}_(k==6)S_(z&&!(l>1&&(f=='e'||f=='c'||f=='g')),43);_(k==7)S_(z||e=='y',e=='y'?44:13);S_(z||e=='y',45);}}
void AP(_c w,I l,I o,I u0){{D*d=W+wp;_(o>=37){{G(d,Q(d,o-19,w,l));_R;}}_(o==0){{Z(d,w,l);d[l]='\\'';d[l+1]='s';G(d,l+2);_R;}}_(o<4){{G(d,SF(o-1,d,w,l,u0));_R;}}D a[{wb_buf}],b[{wb_buf}];d=a;_c s=w;I sl=l,dl=0,c=0;_u p=T+ao[o];_W(!(c&32)){{c=*p++;I u=c&31;_(u<9)dl=SF(u,d,s,sl,u0);else _(u<14){{I i=*p++,y=Y(s,sl),e=s[sl-1]=='e',k=u==10?e:u==11?y:u==12?(y||e):u>12;dl=M(d,s,sl,k,u==11&&!y?i+1:i);}}else _(u==14){{Z(d,s,sl);d[sl]=s[sl-1];dl=Q(d+sl+1,*p++,s,0)+sl+1;}}else _(u==15)dl=Q(d,*p++,s,sl);else _(u==16)G(d,dl);else _(u<19){{G(d,dl);D u2[{wb_buf}];G(u2,M(u2,d,dl,0,0));_(u>17)G(u2,SF(0,u2,d,dl,u0));}}else s=a,sl=dl,d=b;}}}}
I gt(J a,J b){{_((a>>{id_bits})!=(b>>{id_bits}))_R(a>>{id_bits})>(b>>{id_bits});{ol_type} oa=OL[(V)a&{id_mask_hex}u],ob=OL[(V)b&{id_mask_hex}u];I la=oa&{len_mask},lb=ob&{len_mask},m=X(la,lb);_c sa=W+(oa>>{len_bits}),*sb=W+(ob>>{len_bits});for(I i={cmp_start};i<m;i++)_(sa[i]!=sb[i])_R(u8)sa[i]>(u8)sb[i];_R la>lb;}}
I main(){{
#ifdef _WIN32
_setmode(_fileno(stdout),_O_BINARY);
#endif
V C[128]={{}};I ck=0,cn=0,an={an_start};F(i,128)_(i&&i!=10&&i!=13)C[i]=ck++<<14;for(_u s=(_u)U;*s;)CP[cn++]=*s<128?*s++:(*s<224?(s+=2,((s[-2]&31)<<6)|(s[-1]&63)):(s+=3,((s[-3]&15)<<12)|((s[-2]&63)<<6)|(s[-1]&63)));J R=2354958,O2=2048000,O3=2293632,P4=1ULL<<40,U0=P4/R,x={x_end}ULL;for(I i=cn;i--;){{V k=CP[i];I sh=k<128?22:k<2048?15:8;_W(x>=(U0<<sh))B[--an]=x&255,x>>=8;x=k<128?(x>>14)*R+C[k]+(x&16383):k<2048?(x>>7)*R+O2+((k-128-(k>1564))<<7)+(x&127):x*R+O3+k-(k<8304?2048:k<57344?2160:4208);}}J t=x-P4;F(i,5)B[--an]=t&255,t>>=8;_u b0=B+an;I lm=U2(b0);dz(b0+4,MB);_u pb=b0+4+lm,*Qd=pb+U2(b0+2),*x0=MB+{w_bytes_len},*x1=x0+{n_fsm},*ln=x1+{n_fsm},*up=ln+{h0 + 1},*MP=up+{n_up_inf};F(i,{n_fsm})nx[i]=x0[i],nx[256+i]=x1[i];ct(ln,{h0 + 1},lr,rr,vr);I q=0;F(i,{n_masks})mo[i]=q,q+=1+MP[q];T=MP+q;q=0;F(i,37){{ao[i]=q;for(;;){{u8 c=T[q++];_(c>8&&c<16)q++;_(c&32)break;}}}}F(i,46)A[i]=(_c)(T+q),AL[i]=strlen(A[i]),q+=AL[i]+1;CS=(_c)T;F(i,{n_ch}){{CO[i]=q;u8 b=T[q];q+=CL[i]=1+(b>191)+(b>223)+(b>239);}}Z(UP,T+q,{n_ch});_u zp=T+q+{n_ch};F(a,{s_stride})F(b,{s_stride})C0[a*{s_stride}+b]=zp[a];zp+={s_stride};I nb=U2(zp);zp+=2;F(i,nb)C0[zp[i*3]*{s_stride}+zp[i*3+1]]=zp[i*3+2];zp+=nb*3;_u _is=zp;H iu0=_is[{s_stride}];_u _il=_is+{s_stride + 1};zp=_il+{s_stride};F(a,{s_stride})F(b,{s_stride})IS[a*{s_stride}+b]=_is[b],IL[a*{s_stride}+b]=_il[b];nb=U2(zp);zp+=2;F(i,nb){{I x=zp[i*3]*{s_stride}+zp[i*3+1];IS[x]=IL[x]=zp[i*3+2];}}zp+=nb*3;nb=U2(zp);zp+=2;F(i,nb)ae(IS,zp[i*4+1]*{s_stride}+zp[i*4+2],zp[i*4],zp[i*4+3]),ae(IL,zp[i*4+1]*{s_stride}+zp[i*4+2],zp[i*4],zp[i*4+3]);zp+=nb*4;nb=U2(zp);zp+=2;F(i,nb)ae2(IS,zp[i*5+2]*{s_stride}+zp[i*5+3],zp[i*5+1],zp[i*5],zp[i*5+4]),ae2(IL,zp[i*5+2]*{s_stride}+zp[i*5+3],zp[i*5+1],zp[i*5],zp[i*5+4]);zp+=nb*5;_u nu=zp,*rp=nu+{n_tot_ctx},*fq=rp;F(c,{n_tot_ctx})fq+=nu[c];F(c,{n_tot_ctx}){{I tn=0,pv=0;F(j,nu[c]){{pv+=*rp++;I f=*fq+++1;V h=((V)pv<<20)|((V)f<<8);F(k,f)RP[c][tn++]=h+k;}}}}V xp=U4(pb);I po=4;F(i,{sig_pad * 2 + 1})S[i]=i<{sig_pad - sig_cut}?4095:i>{sig_pad + sig_cut}?1:(H)X(4095,rint(4096.0/(1.0+exp((i-{sig_pad})/{denom}.))));const i8*WB=(const i8*)MB,*q1=WB+{h0},*q2=q1+{n_fsm},*q3=q2+{n_fsm},*qw=q3+{n_fsm};F(u,{n_ch}){{I z=WB[u]*16+{sig_pad};F(s,{n_fsm})T1[u][s]=z+qw[u*3]*q1[s],T2[u][s]=qw[u*3+1]*q2[s],T3[u][s]=qw[u*3+2]*q3[s];}}R=1ULL<<32;J L=U4(Qd),rv;V di=4,pc[{pc_buf}]={{0}},pbo[{pc_buf}]={{0}},o1,b2,b3,k1,k2,k3,s1,s2,s3;I u;_c PB=W;I pl=0,pp=0;F(i,5)pc[i]={bos};F(_w,{n_base}){{I upv=pl&&UP[pc[5]];DR(vp,(X(pl,11)<<2)|((pp>0)<<1)|upv)I p=vp>>20;V lo=p==pl?{bos}:pc[5+p],x1_=pc[4+p];DR(vc,48+C0[lo*{s_stride}+x1_])I c0=vc>>20;D*Wb=W+wp;I wl=pbo[p],cln=p;Z(Wb,PB,wl);EC(c0)I u0=UP[pc[5]];V c1=c0,c2=pc[4+p],c3=pc[3+p],c4=pc[2+p],rk=0;for(;;){{o1=(c1*{f1_k}+X(rk,{f1_k - 1}))*{h_stride};rk++;{b2_expr}b3=((c3<<{cb_shift * 2})^(c2<<{cb_shift})^c1^0x8888u)*0x9E3779B1u;b3=(b3^(b3>>16))&{mask3}u;u=0;_W(u<{h0}){{ST(u,lr[u],rr[u])}}I tk=vr[u-{h0}];_(tk=={eow})break;EC(tk)c4=c3;c3=c2;c2=c1;c1=tk;}}H v=u0?iu0:(cln>=7?IL:IS)[c2*{s_stride}+c1];_(v>=512)v=EX[v-512][c3];_(v>=512)v=EX[v-512][c4];DR(vi,{48 + ac0}+v)I inf=u0?up[vi>>20]:(I)(vi>>20);G(Wb,wl);_u mp=MP+mo[inf];for(I k=1;k<=mp[0];k++)AP(Wb,wl,mp[k],u0);PB=Wb;pl=cln;pp=p;}}I n=nw;F(i,n){{J y=PA[i];CT[(y>>40)&4095]++;C2[y>>52]++;}}V s1_=0,s2_=0;F(i,4096){{V a=CT[i],b=C2[i];CT[i]=s1_;C2[i]=s2_;s1_+=a;s2_+=b;}}F(i,n){{J y=PA[i];PX[CT[(y>>40)&4095]++]=y;}}F(i,n){{J y=PX[i];PA[C2[y>>52]++]=y;}}for(I i=1;i<n;i++){{J y=PA[i];I j=i;_W(j&&(PA[j-1]>>40)==(y>>40)&&gt(PA[j-1],y))PA[j]=PA[j-1],j--;_(j<i)PA[j]=y;}}I oi=0;F(i,n){{{ol_type} o=OL[(V)PA[i]&{id_mask_hex}u];I l=o&{len_mask};_c s=W+(o>>{len_bits});Z(O+oi,s,8);_(l>8)Z(O+oi+8,s+8,l-8);oi+=l;O[oi++]='\\n';}}fwrite(O,1,oi,stdout);}}"""
        )
        return cpp_code
