import sys, numpy as np
sys.argv=[sys.argv[0],'--native','build/qwen38-r1/test_fast_kernels','--report','/dev/null']
sys.path.insert(0,'tests')
import test_qwen38_fast_kernels as T
T.NATIVE=__import__('pathlib').Path('build/qwen38-r1/test_fast_kernels')
rng=np.random.default_rng(7)
worst=(0,None); bad=[]
ms=list(range(900,1101))+list(range(1101,2052,37))+[2047,2048,2049,2050,2051]
for m in ms:
    inp=T.attend_inputs(rng,T.ATTEND_REAL,m+40,m)
    f=T.run_attend('fast',*inp); o=T.run_attend('oracle',*inp)
    ref,mag,fac=T.attend_fp64(*inp)
    st=T.against_oracle(f[2],o[2],mag)
    if f[0] or st['max']>2**-12: bad.append((m,f[0],st['max']))
    if st['max']>worst[0]: worst=(st['max'],m)
print('cases',len(ms),'worst',worst,'bad',bad[:10])
