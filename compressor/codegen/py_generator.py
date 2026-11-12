"""
Python Solver Generator Module
==============================
Emits standalone, self-contained Python solver (solve.py).
"""

class PySolverGenerator:
    """Emits standalone Python dictionary decompressor source code."""

    @staticmethod
    def generate(
        u_str: str,
        x_end: int,
        cs_str: str,
        x_cs: int,
        l_w: int,
        l_m: int,
        l_p: int
    ) -> str:
        """Generate standalone Python decompressor source code."""
        return (
            f'import os,sys,struct,subprocess\n'
            f'CS="{cs_str}"\n'
            f'U="{u_str}"\n'
            f'def D(s,x):\n'
            f' S=[i for i in range(1,128) if i not in (10,13,34,92)]\n'
            f' M={{c:i for i,c in enumerate(S)}}\n'
            f' P=1<<40;R=len(S)*16384+1919*128+61326;O2=len(S)*16384;O3=O2+1919*128;B=bytearray();ap=B.append\n'
            f' for c in reversed(s):\n'
            f'  k=ord(c)\n'
            f'  if k<128:sh=22;x=(x>>14)*R+M[k]*16384+(x&16383)\n'
            f'  elif k<2048:sh=15;x=(x>>7)*R+O2+((k-128-(k>1564))<<7)+(x&127)\n'
            f'  else:sh=8;x=x*R+O3+k-(2048 if k<8304 else(2160 if k<57344 else 4208))\n'
            f'  while x>=(P<<sh)//R:ap(x&255);x>>=8\n'
            f' t=x-P\n'
            f' for _ in range(5):ap(t&255);t>>=8\n'
            f' return bytes(reversed(B))\n'
            f'def rd(b,mb=8):\n'
            f' nu,nl=struct.unpack_from("<2H",b,0);p=4;ps=0;RT=[0]*2048;tn=0\n'
            f' for _ in range(nu):\n'
            f'  d=b[p];p+=1\n'
            f'  if d==255:d+=b[p];p+=1\n'
            f'  ps+=d;f=b[p];p+=1\n'
            f'  if f>=240:f=((f-240)<<8)|b[p];p+=1\n'
            f'  h=(ps<<(mb+12))|(f<<mb)\n'
            f'  for j in range(f):RT[tn]=h|j;tn+=1\n'
            f' x=struct.unpack_from(">I",b,p)[0];p+=4;o=[0]*nl;mk=(1<<mb)-1\n'
            f' for i in range(nl):\n'
            f'  v=RT[x&mk];x=((v>>mb)&4095)*(x>>mb)+(v&mk)\n'
            f'  while x<65536:x=(x<<8)|b[p];p+=1\n'
            f'  o[i]=v>>(mb+12)\n'
            f' return o\n'
            f'def dlz(b,pref=b""):\n'
            f' bo=len(pref);out=bytearray(pref)\n'
            f' l1,l2,lx=struct.unpack_from("<3H",b,0);p=6\n'
            f' wv=rd(b[p:p+l1],11);p+=l1;db=rd(b[p:p+l2],8);p+=l2;lx_b=b[p:p+lx];p+=lx;ex=b[p:]\n'
            f' di=0;li=0;bp=0\n'
            f' for s in wv:\n'
            f'  if s<256:out.append(s)\n'
            f'  else:\n'
            f'   l=3+s-256\n'
            f'   if s==287:l=34+lx_b[li];li+=1\n'
            f'   b_=db[di];di+=1\n'
            f'   if b_<2:d=b_+1\n'
            f'   else:\n'
            f'    eb=b_-1;d=1+(1<<eb)\n'
            f'    for k in range(eb):\n'
            f'     if(ex[bp>>3]>>(bp&7))&1:d+=(1<<k)\n'
            f'     bp+=1\n'
            f'   p_=len(out)-d\n'
            f'   if d>=l:out.extend(out[p_:p_+l])\n'
            f'   else:\n'
            f'    for k in range(l):out.append(out[p_+k])\n'
            f' return bytes(out[bo:])\n'
            f'cb=dlz(D(CS,{x_cs}))\n'
            f'sp_f=(os.getenv("TEMP")or"/tmp")+"/dc_"+str(os.getpid())+(".dll" if os.name=="nt" else ".so");pr=None\n'
            f'try:\n'
            f' pr=subprocess.Popen(["gcc","-O0","-pipe","-shared","-s"]+([] if os.name=="nt" else ["-fPIC"])+["-x","c","-","-o",sp_f],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n'
            f' pr.stdin.write(cb);pr.stdin.close()\n'
            f'except Exception:pr=None\n'
            f'A=D(U,{x_end})\n'
            f'lw,lm,lp=struct.unpack_from("<3H",A,0);o=6\n'
            f'wb=bytes(rd(A[o:o+lw]));o+=lw;mb=dlz(A[o:o+lm]);o+=lm;pyb=dlz(A[o:o+lp],cb);o+=lp\n'
            f'ns={{}};exec(pyb,ns)\n'
            f'ns["solve"](A[o:],wb+mb,pr,sp_f)\n'
        )
