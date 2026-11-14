"""
Rust Solver Generator Module
============================
Emits standalone, self-contained, single-file pure-Rust solver (`solve.rs`).
Uses only `std` (no external crates), compiles cleanly under both Rust Edition
2015 and Edition 2021 with zero warnings, and shares layout calculations with
`CppSolverGenerator` via `compute_solver_layout`.
"""

import math

from compressor.codegen.layout import compute_solver_layout
from compressor.encoding.utf8_rans import UTF8Encoder


class RsSolverGenerator:
    """Emits standalone Rust dictionary decompressor source code."""

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
        """Generate standalone single-file Rust decompressor source code."""
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
        hashes = UTF8Encoder.rs_raw_hashes(u_str)
        stride = lay["stride"]
        mask3 = lay["mask3"]
        eow = lay["eow"]
        bos = lay["bos"]
        s_stride = lay["s_stride"]
        ss = lay["ss"]
        h0 = lay["h0"]
        n_ch_safe = lay["n_ch_safe"]
        n_masks_safe = lay["n_masks_safe"]
        rs_ex_rows = lay["rs_ex_rows"]
        m2_size = lay["m2_size"]
        w_buf = lay["w_buf"]
        o_buf = lay["o_buf"]
        nw_cap = lay["nw_cap"]
        len_bits = lay["len_bits"]
        len_mask = lay["len_mask"]
        id_bits = lay["id_bits"]
        id_mask_hex = lay["id_mask_hex"]
        cmp_start = lay["cmp_start"]
        ol_type = "u32" if lay["is_ol_32"] else "u64"
        pc_buf = lay["pc_buf"]
        wb_buf = lay["wb_buf"]
        n_tot_ctx_safe = lay["n_tot_ctx_safe"]
        x0o = lay["x0o"]
        upo = lay["upo"]
        mpo = lay["mpo"]
        q1o = lay["q1o"]
        q2o = lay["q2o"]
        q3o = lay["q3o"]
        qwo = lay["qwo"]

        denom_s = f"{int(denom)}." if denom == int(denom) else repr(float(denom))
        need_cut = (sig_pad > sig_cut) and (
            (4096.0 / (1.0 + math.exp(-(sig_cut + 1) / denom))) <= 4094.500001
        )
        sm_expr = (
            f"if i<{sig_pad - sig_cut}{{4095}}else if i>{sig_pad + sig_cut}{{1}}else"
            f"{{(4096./(1.+((i as f64-{sig_pad}.)/{denom_s}).exp())).round_ties_even().clamp(1.,4095.)as u16}}"
            if need_cut
            else f"(4096./(1.+((i as f64-{sig_pad}.)/{denom_s}).exp())).round_ties_even().clamp(1.,4095.)as u16"
        )

        fx_fn = (
            "fn fx(h:u32)->Z{let z=h.wrapping_mul(0x9E3779B1);(z^(z>>16))as Z}"
            if use_h2_hash
            else ""
        )
        b2_expr = (
            f"let b2=fx(((c2<<{cb_shift})^c1^0x7777)as u32)&262143;"
            if use_h2_hash
            else f"let b2=(c2*{s_stride}+c1)*{h0};"
        )
        b3_expr = (
            f"let b3=fx(((c3<<{cb_shift * 2})^(c2<<{cb_shift})^c1^0x8888)as u32)&{mask3};"
            if use_h2_hash
            else f"let z=(((c3<<{cb_shift * 2})^(c2<<{cb_shift})^c1^0x8888)as u32).wrapping_mul(0x9E3779B1);let b3=((z^(z>>16))as Z)&{mask3};"
        )
        mo_slice = "" if n_masks > 0 else "[..0]"
        cs_slice = "" if n_ch > 0 else "[..0]"
        u0l_q = (1 << 40) // 2354958

        return (
            f'use std::io::*;type Z=usize;type Q=u64;const U:&str=r{hashes}"{u_str}"{hashes};'
            f'fn u2(b:&[u8],i:Z)->Z{{b[i]as Z|(b[i+1]as Z)<<8}}'
            f'fn u4(b:&[u8])->u32{{(0..4).fold(0,|a,i|a<<8|b[i]as u32)}}'
            f'fn cp(d:&mut[u8],s:&[u8]){{d[..s.len()].copy_from_slice(s)}}'
            f'fn pa(d:&mut[u8],a:&[u8],b:&[u8])->Z{{cp(d,a);cp(&mut d[a.len()..],b);a.len()+b.len()}}'
            f'{fx_fn}'
            f'fn rd(b:&[u8],m:Z)->Vec<Z>{{let(mut p,mut s,k,mut rt)=(4,0,(1<<m)-1,vec![]);let mut g=||{{p+=1;b[p-1]as Z}};for _ in 0..u2(b,0){{let mut d=g();if d==255{{d+=g();}}s+=d;let mut f=g();if f>=240{{f=(f-240)<<8|g();}}let h=s<<(m+12)|f<<m;rt.extend(h..h+f);}}let mut x=(0..4).fold(0,|a,_|a<<8|g());(0..u2(b,2)).map(|_|{{let r=rt[x&k];x=((r>>m)&4095)*(x>>m)+(r&k);while x<65536{{x=x<<8|g();}}r>>(m+12)}}).collect()}}'
            f'fn q(s:&[u8],c:u8)->Z{{s.contains(&c)as Z}}'
            f'fn iy(w:&[u8])->Z{{let l=w.len();if l>1&&w[l-1]==121{{1-q(b"aeiou",w[l-2])}}else{{0}}}}'
            f'struct S<\'a>(Vec<u8>,Vec<{ol_type}>,Vec<Q>,Z,Z,&\'a[u8],[&\'a[u8];46]);'
            f'impl S<\'_>{{'
            f'fn gw(&mut self,l:Z){{let s=self;let o=s.3;s.3+=l;s.1[s.4]=(o<<{len_bits}|l)as {ol_type};s.2[s.4]=((0..8).fold(0,|a:Q,i|a<<8|s.0[o+i]as Q)>>{id_bits}<<{id_bits})|s.4 as Q;s.4+=1;}}'
            f'fn g(&mut self,x:&[u8]){{let s=self;cp(&mut s.0[s.3..],x);s.gw(x.len());}}'
            f'fn mw(&self,d:&mut[u8],w:&[u8],k:Z,i:Z)->Z{{pa(d,&w[..w.len()-k],self.6[i])}}'
            f'fn sf(&self,k:Z,d:&mut[u8],w:&[u8],u0:u8)->Z{{let l=w.len();let(e,f,y,z)=(w[l-1],if l>1{{w[l-2]}}else{{0}},iy(w),(w[l-1]==101)as Z);let(a,b)=match k{{0=>if u0>0{{(0,41)}}else if q(b"sxz",e)>0||e==104&&(f==99||f==115){{(0,39)}}else{{(y,41-y)}},1|3|4=>{{let j=k>>1;((1-z)*y,if z>0{{33+j}}else if y>0{{36+j}}else{{[7,10,11][j]}})}},2=>(z*(1-q(b"eyo",f)),8),5=>{{let v=if l>2&&f==108{{z*(1-q(b"aeiou",w[l-3]))}}else{{0}};(y|v,if y>0{{42}}else if v>0{{12}}else if f==105&&e==99{{14}}else{{17}})}},6=>(z*(1-q(b"ecg",f)),43),_=>(z|(e==121)as Z,if k>7{{45}}else if e==121{{44}}else{{13}})}};self.mw(d,w,a,b)}}'
            f'fn ap(&mut self,at:&[Z],wb:Z,wl:Z,o:Z,u0:u8){{let s=self;let(mut a,mut b)=([0u8;{wb_buf}],[0u8;{wb_buf}]);cp(&mut b,&s.0[wb..wb+wl]);if o<4||o>36{{let n=if o==0{{pa(&mut a,&b[..wl],b"\'s")}}else if o>36{{pa(&mut a,s.6[o-19],&b[..wl])}}else{{s.sf(o-1,&mut a,&b[..wl],u0)}};s.g(&a[..n]);}}else{{let(mut p,mut dl,mut sl)=(at[o],0,wl);while{{let c=s.5[p];p+=1;let u=(c&31)as Z;let i=if u>8&&u<16{{p+=1;s.5[p-1]as Z}}else{{0}};match u{{0..=8=>dl=s.sf(u,&mut a,&b[..sl],u0),9..=13=>{{let r=&b[..sl];let(y,e)=(iy(r),(r[sl-1]==101)as Z);dl=s.mw(&mut a,r,[0,e,y,y|e,1][u-9],i+(u==11&&y==0)as Z);}},14=>{{b[sl]=b[sl-1];dl=s.mw(&mut a,&b[..sl+1],0,i);}},15=>dl=pa(&mut a,s.6[i],&b[..sl]),16..=18=>{{let r=&a[..dl];s.g(r);if u>16{{let n=s.mw(&mut b,r,0,0);s.g(&b[..n]);if u>17{{let n=s.sf(0,&mut b,r,u0);s.g(&b[..n]);}}}}}},_=>{{sl=dl;cp(&mut b,&a[..sl]);}}}};c&32==0}}{{}}}}}}}}'
            f'fn main(){{'
            f'let(mut bv,r)=(vec![],2354958);let mut x={x_end}u64;'
            f'for c in U.chars().rev(){{let k=c as Q;let(d,f,p)=if k<128{{(14,k-1-(k>10)as Q-(k>13)as Q,0)}}else if k<2048{{(7,k-128-(k>1564)as Q,2048000)}}else{{(0,k-if k<8304{{2048}}else if k<57344{{2160}}else{{4208}},2293632)}};while x>={u0l_q}<<(d+8){{bv.push(x as u8);x>>=8;}}x=(x>>d)*r+p+(f<<d)+(x&((1<<d)-1));}}'
            f'for _ in 0..5{{bv.push(x as u8);x>>=8;}}bv.reverse();'
            f'let(lm,a,c,e)=(u2(&bv,0),u2(&bv,4),u2(&bv,6),u2(&bv,8));let(wv,db)=(rd(&bv[10..],11),rd(&bv[10+a..],8));let(mut lp,mut bp,mut di,mut mb,ex)=(10+a+c,0,0,vec![],&bv[10+a+c+e..]);for s in wv{{if s<256{{mb.push(s as u8);}}else{{let l=if s==287{{lp+=1;34+bv[lp-1]as Z}}else{{s-253}};let d=db[di];di+=1;let q=if d<2{{d+1}}else{{let g=d-1;let mut v=1+(1<<g);for k in 0..g{{v+=((ex[bp>>3]>>(bp&7)&1)as Z)<<k;bp+=1;}}v}};for _ in 0..l{{let x=mb[mb.len()-q];mb.push(x);}}}}}}'
            f'let(pb,qd)=(&bv[4+lm..],&bv[4+lm+u2(&bv,2)..]);'
            f'let(mut ch,mut vr,mut n,mut i,mut c,mut p)=([[0;2];{h0}],[0;{h0 + 1}],1,0,0u64,0);for lv in 1..33{{for s in 0..{n_ch + 1}{{if mb[{lay["lno"]}+s]as Z==lv{{c<<=lv-p;p=lv;let mut u=0;for b in(1..lv).rev(){{let w=&mut ch[u][(c>>b)as Z&1];if *w==0{{*w=n;n+=1;}}u=*w;}}ch[u][c as Z&1]={h0}+i;vr[i]=s;i+=1;c+=1;}}}}}}'
            f'let(mut mo,mut q)=([0;{n_masks_safe}],0);for v in&mut mo{mo_slice}{{*v=q;q+=1+mb[{mpo}+q]as Z;}}'
            f'let t=&mb[{mpo}+q..];q=0;let mut at=[0;37];for v in&mut at{{*v=q;while{{let c=t[q];q+=1+((c>8&&c<16)as Z);c&32==0}}{{}}}}'
            f'let mut ax=[&[][..];46];for v in&mut ax{{let p=q;while t[q]!=0{{q+=1;}}*v=&t[p..q];q+=1;}}'
            f'let mut cs=[&[][..];{n_ch_safe}];for v in&mut cs{cs_slice}{{let l=(t[q].leading_ones()as Z).max(1);*v=&t[q..q+l];q+=l;}}'
            f'let up=&t[q..q+{n_ch}];let mut zp=q+{n_ch};'
            f'macro_rules! B{{($k:expr)=>(t[zp+$k]as Z)}}'
            f'let mut c0m=vec![0u8;{ss}];for i in 0..{ss}{{c0m[i]=t[zp+i/{s_stride}];}}zp+={s_stride};'
            f'zp+=2;for _ in 0..u2(t,zp-2){{c0m[B!(0)*{s_stride}+B!(1)]=t[zp+2];zp+=3;}}'
            f'let iu0=B!({s_stride});let mut ex=vec![[0;{s_stride}];{rs_ex_rows}];let mut xn={2 * s_stride};for a in 0..{s_stride}{{for b in 0..{s_stride}{{ex[a][b]=B!(b);ex[a+{s_stride}][b]=B!({s_stride + 1}+b);}}}}zp+={2 * s_stride + 1};'
            f'for d in 0..3{{zp+=2;for _ in 0..u2(t,zp-2){{for&q in&[0,{s_stride}]{{let(mut r,mut k)=(q+B!(d),B!(d+1));for m in(0..d).rev(){{let mut v=ex[r][k];if v<512{{ex[xn].fill(v);v=512+xn;ex[r][k]=v;xn+=1;}}r=v-512;k=B!(m);}}ex[r][k]=B!(d+2);}}zp+=d+3;}}}}'
            f'let(mut rpp,mut fqp)=(zp+{n_tot_ctx_safe},zp+{n_tot_ctx_safe});for&v in&t[zp..fqp]{{fqp+=v as Z;}}'
            f'let mut rp=vec![[0u32;256];{n_tot_ctx_safe}];for c in 0..{n_tot_ctx_safe}{{let(mut tn,mut pv)=(0,0);for _ in 0..t[zp+c]{{pv+=t[rpp]as u32;rpp+=1;let f=t[fqp]as Z+1;fqp+=1;for k in 0..f{{rp[c][tn]=pv<<20|(f as u32)<<8|k as u32;tn+=1;}}}}}}'
            f'let(mut xp,mut po)=(u4(pb),4);let sm:Vec<u16>=(0..{2 * sig_pad + 1}).map(|i|{sm_expr}).collect();'
            f'let mut tt=vec![[[0i16;{stride}];{h0}];3];'
            f'let w=|i|mb[i]as i8 as i16;for u in 0..{n_ch}{{let z=w(u)*16+{sig_pad};let q={qwo}+u*3;let(a,b,c)=(w(q),w(q+1),w(q+2));for s in 0..{n_fsm}{{tt[0][u][s]=z+a*w({q1o}+s);tt[1][u][s]=b*w({q2o}+s);tt[2][u][s]=c*w({q3o}+s);}}}}'
            f'let(mut f1,mut f2,mut f3)=(vec![0u8;{s_stride * f1_k * h_stride}],vec![0u8;{m2_size}],vec![0u8;{lay["m3"] + h0}]);'
            f'let(mut rg,mut lv,mut di,mut pbw,mut pl,mut pp)=(1u64<<32,u4(qd)as Q,4,0,0,0);'
            f'let(mut pc,mut pbo)=([{bos};{pc_buf}],[0;{pc_buf}]);'
            f'let mut st=S(vec![0u8;{w_buf}],vec![0;{nw_cap}],vec![0;{nw_cap}],0,0,t,ax);'
            f'let mut dr=|c:Z|{{let v=rp[c][(xp&255)as Z];xp=((v>>8)&4095)*(xp>>8)+(v&255);while xp<65536{{xp=xp<<8|pb[po]as u32;po+=1;}}(v>>20)as Z}};'
            f'for _ in 0..{n_base}{{let p=dr(pl.min(11)*4|(pp>0)as Z*2|if pl!=0{{up[pc[5]]as Z}}else{{0}});'
            f'let c0=dr(48+c0m[(if p==pl{{{bos}}}else{{pc[5+p]}})*{s_stride}+pc[4+p]]as Z);'
            f'let(wb,mut wl,mut cln)=(st.3,pbo[p],p);if wl>0{{st.0.copy_within(pbw..pbw+wl,wb);}}'
            f'macro_rules! P{{($c:expr)=>{{pc[5+cln]=$c;cln+=1;let z=cs[$c];cp(&mut st.0[wb+wl..],z);wl+=z.len();pbo[cln]=wl;}}}}'
            f'P!(c0);let u0=up[pc[5]];let(mut c1,mut c2,mut c3,mut c4,mut rk)=(c0,pc[4+p],pc[3+p],pc[2+p],0);'
            f'loop{{let o1=(c1*{f1_k}+rk.min({f1_k - 1}))*{h_stride};rk+=1;{b2_expr}{b3_expr}'
            f'let mut u=0;while u<{h0}{{let(s1,s2,s3)=(f1[o1+u]as Z,f2[b2+u]as Z,f3[b3+u]as Z);let rv=(rg>>12)*sm[(tt[0][u][s1]+tt[1][u][s2]+tt[2][u][s3])as Z]as Q;let b=if lv>=rv{{lv-=rv;rg-=rv;1}}else{{rg=rv;0}};let x=&mb[{x0o}+b*{n_fsm}..];f1[o1+u]=x[s1];f2[b2+u]=x[s2];f3[b3+u]=x[s3];u=ch[u][b];while rg<(1<<24){{rg<<=8;lv=lv<<8|qd[di]as Q;di+=1;}}}}'
            f'let tk=vr[u-{h0}];if tk=={eow}{{break;}}P!(tk);c4=c3;c3=c2;c2=c1;c1=tk;}}'
            f'let mut v=if u0!=0{{iu0}}else{{ex[if cln>=7{{{s_stride}}}else{{0}}+c2][c1]}};if v>=512{{v=ex[v-512][c3];}}if v>=512{{v=ex[v-512][c4];}}'
            f'let vi=dr({48 + ac0}+v);let inf=if u0!=0{{mb[{upo}+vi]as Z}}else{{vi}};st.gw(wl);'
            f'let mp={mpo}+mo[inf];for k in 1..=mb[mp]as Z{{st.ap(&at,wb,wl,mb[mp+k]as Z,u0);}}pbw=wb;pl=cln;pp=p;}}'
            f'let n=st.4;let(w,ol,pa)=(&st.0,&st.1,&mut st.2[..n]);let x=|p:Q|{{let o=ol[(p&{id_mask_hex})as Z]as Z;(o>>{len_bits},o&{len_mask})}};pa.sort_unstable();let mut i=0;while i<n{{let k=pa[i]>>{id_bits};let mut j=i+1;while j<n&&pa[j]>>{id_bits}==k{{j+=1;}}if j-i>1{{pa[i..j].sort_unstable_by_key(|&p|{{let(s,l)=x(p);&w[s+{cmp_start}.min(l)..s+l]}});}}i=j;}}'
            f'let mut out=Vec::with_capacity({o_buf});for&p in&*pa{{let(s,l)=x(p);out.extend(&w[s..s+l]);out.push(10);}}'
            f'let _=stdout().write_all(&out);}}'
        )
