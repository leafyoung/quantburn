import sys, math; sys.path.insert(0, "scripts")
from refgen import normals

A, SIGMA, R0 = 0.1, 0.01, 0.03

def interp(x, xp, fp):
    if len(xp) == 1: return fp[0]
    c = min(max(x, xp[0]), xp[-1])
    upper = len(xp)
    for i, v in enumerate(xp):
        if v >= c:
            upper = i; break
    upper = min(max(upper,1), len(xp)-1)
    lower = upper-1
    w = (c-xp[lower])/max(xp[upper]-xp[lower],1e-12)
    return fp[lower] + w*(fp[upper]-fp[lower])

class BK:
    def __init__(self, a, sigma, r0, curve=None):
        self.a, self.sigma, self.r0 = a, sigma, r0
        self.curve = curve
    def df(self, m):
        if self.curve is None: return math.exp(-max(self.r0,1e-8)*m)
        return interp(m, self.curve[0], self.curve[1])
    def zcb(self, m, rt=None, steps=256):
        if m <= 0: return 1.0
        if rt is None and self.curve is not None: return self.df(m)
        a=max(self.a,1e-12); s=max(self.sigma,0.0); r=max(rt if rt is not None else self.r0, 1e-8)
        n=max(steps,2); dt=m/(n-1)
        integral=0.0
        for i in range(n):
            t=dt*i
            mlr=math.log(r)*math.exp(-a*t)
            vl=s*s*(1-math.exp(-2*a*t))/max(2*a,1e-12)
            esr=math.exp(mlr+0.5*vl)
            integral += (0.5 if i in (0,n-1) else 1.0)*esr*dt
        return math.exp(-integral)
    def simulate(self, t, n, seed):
        a=max(self.a,0.0); s=max(self.sigma,0.0); r0=max(self.r0,1e-8)
        mean=math.log(r0)*math.exp(-a*t)
        if s==0: return [math.exp(mean)]*n
        var=s*s*(1-math.exp(-2*a*t))/max(2*a,1e-12)
        std=math.sqrt(max(var,0))
        return [math.exp(mean+std*z) for z in normals(seed,n)]

bk=BK(A,SIGMA,R0)
for m in (1.0,5.0,10.0): print("p", m, repr(bk.zcb(m)))
print("y1", repr(-math.log(max(bk.zcb(1.0),1e-12))/1.0))
bkf=BK(A,SIGMA,R0,curve=([1.0,5.0],[0.97,0.86]))
print("curve p 3.0", repr(bkf.zcb(3.0)))
print("sim", [repr(v) for v in bk.simulate(2.0,8,2005)])
print("sim0", [repr(v) for v in BK(0.1,0.0,0.03).simulate(2.0,4,2005)])
