import sys, math; sys.path.insert(0, "scripts")
from refgen import normals
import numpy as np

def cholesky(a):
    return np.linalg.cholesky(np.array(a, dtype=np.float64))

FWD=[0.03,0.035,0.04]; VOL=[0.1,0.15,0.2]
CORR=[[1.0,0.5,0.2],[0.5,1.0,0.3],[0.2,0.3,1.0]]
L=cholesky(CORR)
print("L")
for row in L:
    print([repr(v) for v in row])

def simulate(fwd, vol, corr, t, n, seed):
    fwd=np.array(fwd); vol=np.maximum(np.array(vol),0.0)
    L=cholesky(corr)
    allnonneg = bool((fwd>=0).all())
    base = fwd.mean() if allnonneg else None
    sqrt_t=math.sqrt(max(t,0))
    r = __import__('refgen').normals  # noqa
    from refgen import normals
    it = normals(seed, n*len(fwd))
    out=[]
    for i in range(n):
        z=np.array(it[i*len(fwd):(i+1)*len(fwd)])
        noise=z@L.T
        if allnonneg:
            vals=fwd + base*vol*sqrt_t*noise
            vals=np.maximum(vals,0.0)
        else:
            vals=fwd + vol*sqrt_t*noise
        out.append(vals)
    return out

for row in simulate(FWD, VOL, CORR, 1.0, 3, 2006):
    print([repr(v) for v in row])
for row in simulate([-0.01,0.02],[0.1,0.2],[[1.0,0.0],[0.0,1.0]],1.0,2,2006):
    print([repr(v) for v in row])
