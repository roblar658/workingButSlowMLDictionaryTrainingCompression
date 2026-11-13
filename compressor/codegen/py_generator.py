"""
Python Solver Generator Module
==============================
Emits standalone C-in-Python solver (solve.py < 88 KiB) backed by an embedded
zero-header NN+PAQ C decoder compiled via gcc, with a pure-Python fallback when
gcc is unavailable.
"""

import struct
from typing import Any, Dict, Tuple

from compressor.encoding.lzss import DPOptimalLZSS
from compressor.encoding.utf8_rans import UTF8Encoder


class PySolverGenerator:
    """Emits standalone C-in-Python dictionary decompressor with pure-Python fallback."""

    @staticmethod
    def _build_nn_c_decoder(a: Dict[str, Any]) -> str:
        """Build the headerless C99/GNU-C NN+PAQ decoder shared library source."""
        n_fsm = a["n_fsm"]
        f1_k = a["f1_k"]
        m3b = a["m3b"]
        n_tot_ctx = a["n_tot_ctx"]
        sig_pad = a["sig_pad"]
        sig_cut = a["sig_cut"]
        denom = a["denom"]
        n_ch = a["n_ch"]
        n_ch_safe = max(1, n_ch)
        n_masks = a["n_masks"]
        n_masks_safe = max(1, n_masks)
        n_base = a["n_base"]
        total_words = a["total_words"]
        total_out_bytes = a["total_out_bytes"]
        max_bchars = a["max_bchars"]
        max_wbytes = a["max_wbytes"]
        use_h2_hash = a["use_h2_hash"]
        cb_shift = a["cb_shift"]
        h_stride = a["h_stride"]
        ac0 = a["ac0"]
        ex_rows = max(16, a["ex_cap"])

        stride = 96 if n_fsm <= 96 else 128
        m3 = 1 << m3b
        mask3 = m3 - 1
        eow = n_ch
        bos = n_ch + 1
        s_stride = n_ch + 2
        h0 = max(1, n_ch)

        m2_size = ((1 << 18) + h0) if use_h2_hash else (s_stride * s_stride * h0)
        b2_expr = (
            f"{{V h=((c2<<{cb_shift})^c1^0x7777u)*0x9E3779B1u;b2=(h^(h>>16))&262143u;}}"
            if use_h2_hash
            else f"b2=(c2*_S+c1)*{h0};"
        )

        w_buf = max(total_out_bytes + 64, 1024)
        nw_cap = max(total_words + 16, 64)
        len_bits = max(6, max_wbytes.bit_length())
        len_mask = (1 << len_bits) - 1
        id_bits = max(17, total_words.bit_length())
        id_mask_hex = hex((1 << id_bits) - 1)
        cmp_start = (64 - id_bits) // 8
        ol_type = "V" if (w_buf.bit_length() + len_bits <= 32) else "J"

        pc_buf = max(max_bchars + 16, 64)
        wb_buf = max(max_wbytes + 16, 64)

        parts = [
            "typedef unsigned char u8;typedef signed char i8;typedef short i16;typedef unsigned short H;typedef unsigned V;typedef unsigned long long J;typedef char D;",
            "\n#define I int",
            "\n#define _u const u8*",
            "\n#define _c const D*",
            "\n#define F(i,n) for(I i=0;i<(n);i++)",
            "\n#define Z __builtin_memcpy",
            "\n#define X(a,b) ((a)<(b)?(a):(b))",
            "\n#define _R return",
            "\n#define _(x) if(x)",
            "\n#define _W while",
            f"\n#define _S {s_stride}",
            "\n#define U2(p) ((p)[0]|((p)[1]<<8))",
            "\n#define U4(p) (((V)(p)[0]<<24)|((V)(p)[1]<<16)|((V)(p)[2]<<8)|(p)[3])",
            "\n#define DR(v,c) V v=RP[c][xp&255];xp=((v>>8)&4095)*(xp>>8)+(v&255);_W(xp<65536)xp=(xp<<8)|pb[po++];v>>=20;",
            "\n#define ST(nd,lu,ru) k1=o1+nd;k2=b2+nd;k3=b3+nd;s1=F1[k1];s2=F2[k2];s3=F3[k3];rv=(R>>12)*S[T1[nd][s1]+T2[nd][s2]+T3[nd][s3]];{V b=(L>=rv);L-=b?rv:0;R=b?R-rv:rv;u=b?(ru):(lu);_u _n=nx+(b<<8);F1[k1]=_n[s1];F2[k2]=_n[s2];F3[k3]=_n[s3];}_W(R<1ULL<<24)R<<=8,L=(L<<8)|Qd[di++];",
            "\n#define EC(t) pc[5+cln++]=t;Z(Wb+wl,CS+CO[t],CL[t]);wl+=CL[t];pbo[cln]=wl;\n",
            f"_c A[46],*CS;u8 F1[_S*{f1_k * h_stride}],F2[{m2_size}],F3[{m3 + h0}],nx[512],C0[_S*_S],CL[{n_ch_safe}],UP[_S];"
            f"D W[{w_buf}];H S[{2 * sig_pad + 1}],CO[{n_ch_safe}],IS[_S*_S],IL[_S*_S],EX[{ex_rows}][_S];"
            f"i16 T1[{h0}][{stride}],T2[{h0}][{stride}],T3[{h0}][{stride}];{ol_type} OL[{nw_cap}];"
            f"V RP[{max(1, n_tot_ctx)}][256],CT[4096],C2[4096],lr[{h0}],rr[{h0}],vr[{h0 + 1}],mo[{n_masks_safe}],ao[37],AL[46],wp,nw;"
            f"I ex;J PA[{nw_cap}],PX[{nw_cap}];_u T;",
            "void ct(_u ln,I q,V*L,V*R,V*V0){V nt=1,li=0,c=0,pl=0;for(I l=1;l<33;l++)F(s,q)_(ln[s]==l){c<<=(l-pl);pl=l;V u=0;for(I b=l-1;b>0;b--){V*t=(c>>b)&1?&R[u]:&L[u];_(!*t)*t=nt++;u=*t;}(c&1?R:L)[u]=q-1+li;V0[li++]=s;c++;}}",
            "I aa(H*r,I i){_(r[i]<512){H d=r[i];I q=ex++;F(j,_S)EX[q][j]=d;r[i]=512+q;}_R r[i]-512;}",
            "double _ex(double x){double y=x/128.,t=1+y*(1+y*(.5+y*(1./6+y*(1./24+y*(1./120+y/720)))));F(i,7)t*=t;_R t;}",
            f"void G(_c s,I l){{V o=wp;_(s!=W+o)Z(W+o,s,l);wp+=l;OL[nw]=(({ol_type})o<<{len_bits})|({ol_type})l;J k=0;Z(&k,W+o,8);PA[nw]=(__builtin_bswap64(k)&~{id_mask_hex}ULL)|nw;nw++;}}",
            "I VV(D c){_R c=='a'||c=='e'||c=='i'||c=='o'||c=='u';}I Y(_c w,I l){_R l>1&&w[l-1]=='y'&&!VV(w[l-2]);}I M(D*d,_c w,I l,I k,I i){I m=l-k;Z(d,w,m);Z(d+m,A[i],AL[i]);_R m+AL[i];}I Q(D*d,I i,_c w,I l){I m=AL[i];Z(d,A[i],m);Z(d+m,w,l);_R m+l;}",
            "I SF(I k,D*d,_c w,I l,I u0){D e=w[l-1],f=l>1?w[l-2]:0;I y=Y(w,l),z=e=='e';"
            "\n#define S_(x,i) _R M(d,w,l,x,i)\n"
            "_(!k){_(!u0){_(e=='s'||e=='x'||e=='z'||(e=='h'&&(f=='c'||f=='s')))S_(0,39);_(y)S_(1,40);}S_(0,41);}_(k==1||k==3||k==4){I j=k>>1;S_(z?0:y,z?33+j:y?36+j:j==0?7:j==1?10:11);}_(k==2)S_(z&&!(l>1&&(f=='e'||f=='y'||f=='o')),8);_(k==5){I t=l>2&&z&&f=='l'&&!VV(w[l-3]);S_(y||t,y?42:t?12:l>1&&f=='i'&&e=='c'?14:17);}_(k==6)S_(z&&!(l>1&&(f=='e'||f=='c'||f=='g')),43);_(k==7)S_(z||e=='y',e=='y'?44:13);S_(z||e=='y',45);}",
            f"void AP(_c w,I l,I o,I u0){{D*d=W+wp;_(o>=37){{G(d,Q(d,o-19,w,l));_R;}}_(o==0){{Z(d,w,l);d[l]=39;d[l+1]=115;G(d,l+2);_R;}}_(o<4){{G(d,SF(o-1,d,w,l,u0));_R;}}D a[{wb_buf}],b[{wb_buf}];d=a;_c s=w;I sl=l,dl=0,c=0;_u p=T+ao[o];_W(!(c&32)){{c=*p++;I u=c&31;_(u<9)dl=SF(u,d,s,sl,u0);else _(u<14){{I i=*p++,y=Y(s,sl),e=s[sl-1]=='e',k=u==10?e:u==11?y:u==12?(y||e):u>12;dl=M(d,s,sl,k,u==11&&!y?i+1:i);}}else _(u==14){{Z(d,s,sl);d[sl]=s[sl-1];dl=Q(d+sl+1,*p++,s,0)+sl+1;}}else _(u==15)dl=Q(d,*p++,s,sl);else _(u==16)G(d,dl);else _(u<19){{G(d,dl);D u2[{wb_buf}];G(u2,M(u2,d,dl,0,0));_(u>17)G(u2,SF(0,u2,d,dl,u0));}}else s=a,sl=dl,d=b;}}}}",
            f"I gt(J a,J b){{_((a>>{id_bits})!=(b>>{id_bits}))_R(a>>{id_bits})>(b>>{id_bits});{ol_type} oa=OL[(V)a&{id_mask_hex}u],ob=OL[(V)b&{id_mask_hex}u];I la=oa&{len_mask},lb=ob&{len_mask},m=X(la,lb);_c sa=W+(oa>>{len_bits}),*sb=W+(ob>>{len_bits});for(I i={cmp_start};i<m;i++)_(sa[i]!=sb[i])_R(u8)sa[i]>(u8)sb[i];_R la>lb;}}",
            f"I d(_u MB,_u pb,_u Qd,D*O){{"
            f"_u x0=MB+{a['w_bytes_len']},*x1=x0+{n_fsm},*ln=x1+{n_fsm},*up=ln+{n_ch + 1},*MP=up+{a['n_up_inf']};"
            f"F(i,{n_fsm})nx[i]=x0[i],nx[256+i]=x1[i];ct(ln,{n_ch + 1},lr,rr,vr);"
            f"I q=0;F(i,{n_masks})mo[i]=q,q+=1+MP[q];T=MP+q;q=0;"
            f"F(i,37){{ao[i]=q;for(;;){{u8 c=T[q++];_(c>8&&c<16)q++;_(c&32)break;}}}}"
            f"F(i,46){{A[i]=(_c)(T+q);I _l=0;_W(A[i][_l])_l++;AL[i]=_l;q+=_l+1;}}"
            f"CS=(_c)T;F(i,{n_ch}){{CO[i]=q;u8 b=T[q];q+=CL[i]=1+(b>191)+(b>223)+(b>239);}}"
            f"Z(UP,T+q,{n_ch});_u zp=T+q+{n_ch};"
            f"F(a,_S)F(b,_S)C0[a*_S+b]=zp[a];zp+=_S;I nb=U2(zp);zp+=2;"
            f"F(i,nb)C0[zp[0]*_S+zp[1]]=zp[2],zp+=3;"
            f"_u _is=zp;H iu0=_is[_S];_u _il=_is+_S+1;zp=_il+_S;"
            f"F(a,_S)F(b,_S)IS[a*_S+b]=_is[b],IL[a*_S+b]=_il[b];"
            f"nb=U2(zp);zp+=2;F(i,nb){{I x_=zp[0]*_S+zp[1];IS[x_]=IL[x_]=zp[2];zp+=3;}}"
            f"nb=U2(zp);zp+=2;F(i,nb){{F(t,2)EX[aa(t?IL:IS,zp[1]*_S+zp[2])][zp[0]]=zp[3];zp+=4;}}"
            f"nb=U2(zp);zp+=2;F(i,nb){{F(t,2)EX[aa(EX[aa(t?IL:IS,zp[2]*_S+zp[3])],zp[1])][zp[0]]=zp[4];zp+=5;}}"
            f"_u nu=zp,*rp=nu+{n_tot_ctx},*fq=rp;F(c,{n_tot_ctx})fq+=nu[c];"
            f"F(c,{n_tot_ctx}){{I tn=0,pv=0;F(j,nu[c]){{pv+=*rp++;I f=*fq+++1;V h=((V)pv<<20)|((V)f<<8);F(k,f)RP[c][tn++]=h+k;}}}}"
            f"V xp=U4(pb);I po=4;"
            f"F(i,{sig_pad * 2 + 1})S[i]=i<{sig_pad - sig_cut}?4095:i>{sig_pad + sig_cut}?1:(H)X(4095,(I)(4096./(1.+_ex((i-{sig_pad})/{denom}.))+.5));"
            f"const i8*WB=(const i8*)MB,*q1=WB+{h0},*q2=q1+{n_fsm},*q3=q2+{n_fsm},*qw=q3+{n_fsm};"
            f"F(u,{n_ch}){{I z=WB[u]*16+{sig_pad};F(s,{n_fsm})T1[u][s]=z+qw[u*3]*q1[s],T2[u][s]=qw[u*3+1]*q2[s],T3[u][s]=qw[u*3+2]*q3[s];}}"
            f"J R=1ULL<<32,L=U4(Qd),rv;V di=4,pc[{pc_buf}]={{0}},pbo[{pc_buf}]={{0}},o1,b2,b3,k1,k2,k3,s1,s2,s3;"
            f"I u;_c PB=W;I pl=0,pp=0;F(i,5)pc[i]={bos};"
            f"F(_w,{n_base}){{I upv=pl&&UP[pc[5]];DR(vp,(X(pl,11)<<2)|((pp>0)<<1)|upv)I p=(I)vp;"
            f"V lo=(I)p==pl?{bos}:pc[5+p],x1_=pc[4+p];DR(vc,48+C0[lo*_S+x1_])V c0=vc;"
            f"D*Wb=W+wp;I wl=pbo[p],cln=p;Z(Wb,PB,wl);EC(c0)I u0=UP[pc[5]];"
            f"V c1=c0,c2=pc[4+p],c3=pc[3+p],c4=pc[2+p],rk=0;"
            f"for(;;){{o1=(c1*{f1_k}+X(rk,{f1_k - 1}))*{h_stride};rk++;{b2_expr}b3=((c3<<{cb_shift * 2})^(c2<<{cb_shift})^c1^0x8888u)*0x9E3779B1u;b3=(b3^(b3>>16))&{mask3}u;u=0;_W(u<{n_ch}){{ST(u,lr[u],rr[u])}}I tk=vr[u-{n_ch}];_(tk=={eow})break;EC(tk)c4=c3;c3=c2;c2=c1;c1=tk;}}"
            f"H v=u0?iu0:(cln>=7?IL:IS)[c2*_S+c1];_(v>=512)v=EX[v-512][c3];_(v>=512)v=EX[v-512][c4];DR(vi,{48 + ac0}+v)I inf=u0?up[vi]:(I)vi;"
            f"G(Wb,wl);_u mp=MP+mo[inf];for(I k=1;k<=mp[0];k++)AP(Wb,wl,mp[k],u0);PB=Wb;pl=cln;pp=p;}}"
            f"I n=nw;F(i,n){{J y=PA[i];CT[(y>>40)&4095]++;C2[y>>52]++;}}V s1_=0,s2_=0;F(i,4096){{V a=CT[i],b=C2[i];CT[i]=s1_;C2[i]=s2_;s1_+=a;s2_+=b;}}"
            f"F(i,n){{J y=PA[i];PX[CT[(y>>40)&4095]++]=y;}}F(i,n){{J y=PX[i];PA[C2[y>>52]++]=y;}}"
            f"for(I i=1;i<n;i++){{J y=PA[i];I j=i;_W(j&&(PA[j-1]>>40)==(y>>40)&&gt(PA[j-1],y))PA[j]=PA[j-1],j--;_(j<i)PA[j]=y;}}"
            f"I oi=0;F(i,n){{{ol_type} o=OL[(V)PA[i]&{id_mask_hex}u];I l=o&{len_mask};Z(O+oi,W+(o>>{len_bits}),l);oi+=l;O[oi++]=10;}}_R oi;}}",
        ]
        return "".join(parts)

    @staticmethod
    def _build_nn_py_fallback(a: Dict[str, Any]) -> str:
        """Build pure-Python NN+PAQ fallback source compressed inside K."""
        n_fsm = a["n_fsm"]
        f1_k = a["f1_k"]
        m3b = a["m3b"]
        mask3 = (1 << m3b) - 1
        n_tot_ctx = a["n_tot_ctx"]
        sp = a["sig_pad"]
        sc = a["sig_cut"]
        dn = float(a["denom"])
        n_ch = a["n_ch"]
        h0 = max(1, n_ch)
        eow = n_ch
        bos = n_ch + 1
        s_stride = n_ch + 2
        cb_shift = a["cb_shift"]
        h_stride = a["h_stride"]
        ac0 = a["ac0"]
        b2_py = (
            f"z_=((c2<<{cb_shift})^c1^30583)*2654435761&4294967295;b2=(z_^(z_>>16))&262143"
            if a["use_h2_hash"]
            else f"b2=(c2*{s_stride}+c1)*{h0}"
        )

        return (
f"""import math,struct
W_=struct.unpack_from('<{a['w_bytes_len']}b',M);q={a['w_bytes_len']}
NX=M[q:q+{n_fsm}],M[q+{n_fsm}:q+{2 * n_fsm}];q+={2 * n_fsm}
ln=M[q:q+{n_ch + 1}];q+={n_ch + 1};up=M[q:q+{a['n_up_inf']}];q+={a['n_up_inf']}
mo=[]
for _ in range({a['n_masks']}):mo.append(q);q+=1+M[q]
ao=[];A=[]
for _ in range(37):
 ao.append(q)
 while 1:
  c=M[q];q+=1
  if 8<c<16:q+=1
  if c&32:break
for _ in range(46):e=M.index(0,q);A.append(M[q:e]);q=e+1
CH=[]
for _ in range({n_ch}):b=M[q];l=1+(b>191)+(b>223)+(b>239);CH.append(M[q:q+l]);q+=l
UP=list(M[q:q+{n_ch}])+[0,0];q+={n_ch}
c0f=M[q:q+{s_stride}];q+={s_stride};nb,=struct.unpack_from('<H',M,q);q+=2;CO={{(M[q+i*3],M[q+i*3+1]):M[q+i*3+2] for i in range(nb)}};q+=nb*3
isf=M[q:q+{s_stride}];iu0=M[q+{s_stride}];ilf=M[q+{s_stride + 1}:q+{2 * s_stride + 1}];q+={2 * s_stride + 1};IM={{}}
for k in(2,3,4):
 nb,=struct.unpack_from('<H',M,q);q+=2
 for _ in range(nb):IM[tuple(M[q:q+k])]=M[q+k];q+=k+1
nu=M[q:q+{n_tot_ctx}];rp=q+{n_tot_ctx};fq=rp+sum(nu);RP=[]
for c in range({n_tot_ctx}):
 t=[];pv=0
 for _ in range(nu[c]):pv+=M[rp];rp+=1;f=M[fq]+1;fq+=1;t.extend((pv,f,k) for k in range(f))
 RP.append(t)
q_={n_ch + 1};lr=[0]*q_;rr=[0]*q_;vr=[0]*q_;nt=1;li=c=pl=0
for l in range(1,33):
 for s in range(q_):
  if ln[s]==l:
   c<<=l-pl;pl=l;u=0
   for b in range(l-1,0,-1):
    tb=rr if c>>b&1 else lr
    if not tb[u]:tb[u]=nt;nt+=1
    u=tb[u]
   (rr if c&1 else lr)[u]=q_-1+li;vr[li]=s;li+=1;c+=1
S=[4095 if i<{sp - sc} else 1 if i>{sp + sc} else min(4095,round(4096/(1+math.exp((i-{sp})/{dn})))) for i in range({2 * sp + 1})]
h0={n_ch};nf={n_fsm};qw=W_[h0+3*nf:]
T1=[[W_[u]*16+{sp}+qw[u*3]*W_[h0+s] for s in range(nf)] for u in range(h0)]
T2=[[qw[u*3+1]*W_[h0+nf+s] for s in range(nf)] for u in range(h0)]
T3=[[qw[u*3+2]*W_[h0+2*nf+s] for s in range(nf)] for u in range(h0)]
F1=[0]*{s_stride * f1_k * h_stride};F2=[0]*({(1 << 18) + h0 if a['use_h2_hash'] else s_stride * s_stride * h0});F3=[0]*{(1 << m3b) + h0}
VV=lambda c:c in b'aeiou'
Y=lambda w:len(w)>1 and w[-1]==121 and not VV(w[-2])
MW=lambda w,k,i:(w[:-k] if k else w)+A[i]
def SF(k,w,u0):
 m=len(w)>1;e=w[-1];f=m and w[-2];y=Y(w);z=e==101
 if k==0:return MW(w,0,39) if not u0 and(e in b'sxz' or e==104 and f in b'cs') else MW(w,1,40) if not u0 and y else MW(w,0,41)
 if k in(1,3,4):j=k>>1;return MW(w,0 if z else y,33+j if z else 36+j if y else(7,10,11)[j])
 if k==2:return MW(w,z and not(m and f in b'eyo'),8)
 if k==5:t=len(w)>2 and z and f==108 and not VV(w[-3]);return MW(w,y or t,42 if y else 12 if t else 14 if m and f==105 and e==99 else 17)
 if k==6:return MW(w,z and not(m and f in b'ecg'),43)
 return MW(w,z or e==121,45 if k>7 else 44 if e==121 else 13)
OW=[]
def AP(w,o,u0):
 if o>=37:OW.append(A[o-19]+w);return
 if o==0:OW.append(w+b"'s");return
 if o<4:OW.append(SF(o-1,w,u0));return
 p=ao[o];s=w;d=b"";c=0
 while not c&32:
  c=M[p];p+=1;u=c&31
  if u<9:d=SF(u,s,u0)
  elif u<14:i=M[p];p+=1;y=Y(s);k=s[-1]==101 if u==10 else y if u==11 else(y or s[-1]==101) if u==12 else u>12;d=MW(s,k,i+1 if u==11 and not y else i)
  elif u==14:d=s+s[-1:]+A[M[p]];p+=1
  elif u==15:d=A[M[p]]+s;p+=1
  elif u==16:OW.append(d)
  elif u<19:
   OW.append(d);OW.append(d+A[0])
   if u>17:OW.append(SF(0,d,u0))
  else:s=d
xp=int.from_bytes(pb[:4],'big');po=4;Rn=1<<32;L=int.from_bytes(Qd[:4],'big');di=4;pc=[{bos}]*5;PB=[];pl=pp=0
def dr(c):
 global xp,po
 v=RP[c][xp&255];xp=v[1]*(xp>>8)+v[2]
 while xp<65536:xp=(xp<<8)|pb[po];po+=1
 return v[0]
for _ in range({a['n_base']}):
 p=dr((min(pl,11)<<2)|((pp>0)<<1)|(pl and UP[pc[5]]));lo={bos} if p==pl else pc[5+p]
 c0=dr(48+CO.get((lo,pc[4+p]),c0f[lo]))
 pc=pc[:5+p]+[c0];Wb=PB[:p]+[CH[c0]];u0=UP[pc[5]];c1=c0;c2=pc[4+p];c3=pc[3+p];c4=pc[2+p];rk=0
 while 1:
  o1=(c1*{f1_k}+min(rk,{f1_k - 1}))*{h_stride};rk+=1;{b2_py}
  z_=(((c3<<{cb_shift * 2})^(c2<<{cb_shift})^c1^34952)*2654435761)&4294967295;b3=(z_^(z_>>16))&{mask3};u=0
  while u<h0:
   k1=o1+u;k2=b2+u;k3=b3+u;s1=F1[k1];s2=F2[k2];s3=F3[k3];rv=(Rn>>12)*S[T1[u][s1]+T2[u][s2]+T3[u][s3]];b=1 if L>=rv else 0
   if b:L-=rv;Rn-=rv;u=rr[u]
   else:Rn=rv;u=lr[u]
   F1[k1]=NX[b][s1];F2[k2]=NX[b][s2];F3[k3]=NX[b][s3]
   while Rn<16777216:Rn<<=8;L=((L<<8)|Qd[di])&4294967295;di+=1
  tk=vr[u-h0]
  if tk=={eow}:break
  pc.append(tk);Wb.append(CH[tk]);c4,c3,c2,c1=c3,c2,c1,tk
 cv=iu0 if u0 else IM.get((c4,c3,c2,c1),IM.get((c3,c2,c1),IM.get((c2,c1),(ilf if len(Wb)>6 else isf)[c1])))
 vi=dr({48 + ac0}+cv);inf=up[vi] if u0 else vi;wb=b"".join(Wb);OW.append(wb)
 for k in range(M[mo[inf]]):AP(wb,M[mo[inf]+1+k],u0)
 PB=Wb;pl=len(Wb);pp=p
OW.sort();_w(b"\\n".join(OW)+(b"\\n" if OW else b""))"""
        )

    @classmethod
    def generate(cls, a: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate standalone solve.py (< 88 KiB) with C decoder + pure-Python fallback."""
        meta_lz = a["meta_lz"]
        pci_b = a["pci_b"]
        ml_bin = a["ml_bin"]

        py_fallback = cls._build_nn_py_fallback(a)

        blob = (
            struct.pack(">II", len(meta_lz), len(pci_b))
            + meta_lz
            + pci_b
            + ml_bin
        )
        u_str, x_end = UTF8Encoder.encode_py_nobidi(blob)

        c_src = cls._build_nn_c_decoder(a)
        c_lz = DPOptimalLZSS.compress((c_src + "\x00" + py_fallback).encode("utf-8"))
        uc_str, x_end_c = UTF8Encoder.encode_py_nobidi(c_lz)

        o_buf = max(a["total_out_bytes"] + 64, 1024)
        py_src = (
            "import ctypes as C,os,struct as T,sys\n"
            f'U="{u_str}"\n'
            f'K="{uc_str}"\n'
            "nt=os.name=='nt'\n"
            "def _w(b):\n"
            " if nt:import msvcrt;msvcrt.setmode(1,os.O_BINARY)\n"
            " sys.stdout.buffer.write(b)\n"
            "def du(s,x):\n"
            " U0=(1<<40)//2322190;B=bytearray()\n"
            " for c in reversed(s):\n"
            "  k=ord(c);sh=22 if k<128 else 15 if k<2048 else 8\n"
            "  while x>=(U0<<sh):B.append(x&255);x>>=8\n"
            "  x=(x>>14)*2322190+((k-1-(k>10)-(k>13)-(k>34)-(k>92))<<14)+(x&16383)if k<128 else(x>>7)*2322190+2015232+((k-128-(k>1564))<<7)+(x&127)if k<2048 else x*2322190+2260864+k-(2048 if k<8304 else 2160 if k<57344 else 4208)\n"
            " return(x-(1<<40)).to_bytes(5,'big')+B[::-1]\n"
            "def rd(b,m):\n"
            " nu,nl=T.unpack_from('<HH',b);p=4;s=0;t=[]\n"
            " for _ in range(nu):\n"
            "  d=b[p];p+=1\n"
            "  if d==255:d+=b[p];p+=1\n"
            "  s+=d;f=b[p];p+=1\n"
            "  if f>=240:f=((f-240)<<8)|b[p];p+=1\n"
            "  t.extend((s,f,j)for j in range(f))\n"
            " x=int.from_bytes(b[p:p+4],'big');p+=4;o=[]\n"
            " for _ in range(nl):\n"
            "  u,f,j=t[x&(1<<m)-1];x=f*(x>>m)+j\n"
            "  while x<65536:x=(x<<8)|b[p];p+=1\n"
            "  o.append(u)\n"
            " return o\n"
            "def dz(b):\n"
            " l1,l2,l3=T.unpack_from('<HHH',b);W=rd(b[6:],11);D=rd(b[6+l1:],8);lp=6+l1+l2;e=b[lp+l3:];o=bytearray();di=li=bp=0\n"
            " for s in W:\n"
            "  if s<256:o.append(s)\n"
            "  else:\n"
            "   l=34+b[lp+li]if s==287 else 3+s-256;li+=s==287;d=D[di];di+=1\n"
            "   if d<2:q=d+1\n"
            "   else:\n"
            "    g=d-1;q=1+(1<<g)\n"
            "    for k in range(g):q+=((e[bp>>3]>>(bp&7))&1)<<k;bp+=1\n"
            "   for _ in range(l):o.append(o[-q])\n"
            " return bytes(o)\n"
            f"cb,pf=dz(du(K,{x_end_c})).split(b'\\0',1);D=du(U,{x_end});lm,lp=T.unpack_from('>II',D);M=dz(D[8:8+lm]);pb=D[8+lm:];Qd=pb[lp:];sf=os.path.join(os.getenv('TEMP','.')if nt else'/tmp',f'_{{os.getpid()}}.'+(nt and'dll'or'so'))\n"
            "try:\n"
            " import subprocess as S;S.run(['gcc','-O0','-pipe','-shared','-fPIC','-s','-x','c','-','-o',sf],input=cb,check=1,stderr=S.DEVNULL)\n"
            f" L=C.CDLL(sf);O=(C.c_char*{o_buf})();_w(memoryview(O)[:L.d(M,pb,Qd,O)])\n"
            " if nt:import _ctypes;_ctypes.FreeLibrary(L._handle)\n"
            "except Exception:exec(pf.decode(),{'M':M,'pb':pb,'Qd':Qd,'_w':_w})\n"
            "finally:\n"
            " try:os.remove(sf)\n"
            " except OSError:pass\n"
        )

        info = {
            "blob_len": len(blob),
            "c_src_len": len(c_src),
            "c_lz_len": len(c_lz),
            "py_lz_len": len(py_fallback),
            "u8_len": len(u_str.encode("utf-8")) + len(uc_str.encode("utf-8")),
            "c_src": c_src,
        }
        return py_src, info
