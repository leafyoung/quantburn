import sys, math; sys.path.insert(0, "scripts")
from refgen import normals

def interp(x, xp, fp):
    if len(xp) == 1: return fp[0]
    c = min(max(x, xp[0]), xp[-1])
    # searchsorted right=False: first idx with xp[idx] >= c
    upper = 0
    for i, v in enumerate(xp):
        if v >= c:
            upper = i; break
    else:
        upper = len(xp)
    upper = min(max(upper,1), len(xp)-1)
    lower = upper-1
    w = (c-xp[lower])/max(xp[upper]-xp[lower],1e-12)
    return fp[lower] + w*(fp[upper]-fp[lower])

A, SIGMA, R0 = 0.1, 0.01, 0.03

class HW:
    def __init__(self, a, sigma, r0, curve=None):
        self.a, self.sigma, self.r0 = a, sigma, r0
        self.curve = curve
    def df(self, m):
        if self.curve is None: return math.exp(-self.r0*m)
        return interp(m, self.curve[0], self.curve[1])
    def fwd(self, m):
        bump=1e-4
        up=m+bump; down=max(m-bump,0)
        lu=-math.log(max(self.df(up),1e-12)); ld=-math.log(max(self.df(down),1e-12))
        return (lu-ld)/max(up-down,1e-12)
    def theta(self, m):
        a=max(self.a,1e-12); s=max(self.sigma,0.0); bump=1e-4
        dfdt=(self.fwd(m+bump)-self.fwd(max(m-bump,0)))/max(2*bump,1e-12)
        return dfdt + a*self.fwd(m) + (s*s/(2*a))*(1-math.exp(-2*a*m))
    def zcb(self, m, rt=None):
        # curve-consistent HW (Brigo-Mercurio 3.39) at t=0; sigma term vanishes
        if m <= 0: return 1.0
        if rt is None and self.curve is not None: return self.df(m)
        a=max(self.a,1e-12); r=rt if rt is not None else self.r0
        b=(1-math.exp(-a*m))/a
        market=self.df(m); bump=1e-4
        f0=(-math.log(max(self.df(bump),1e-12))+math.log(max(self.df(0.0),1e-12)))/bump
        return market*math.exp(b*f0)*math.exp(-b*r)
    def simulate(self, t, n, seed):
        a=max(self.a,0.0); s=max(self.sigma,0.0)
        mean=self.r0*math.exp(-a*t)
        if s==0: return [mean]*n
        var=s*s*(1-math.exp(-2*a*t))/max(2*a,1e-12)
        std=math.sqrt(max(var,0))
        return [mean+std*z for z in normals(seed,n)]

xp=[0.5,1.0,2.0,5.0]; fp=[0.9753,0.9512,0.9048,0.7788]
print("interp 1.25", repr(interp(1.25,xp,fp)))
print("interp 0.2", repr(interp(0.2,xp,fp)))
print("interp 7.0", repr(interp(7.0,xp,fp)))
print("interp 3.7", repr(interp(3.7,xp,fp)))

hw=HW(A,SIGMA,R0)
for m in (1.0,5.0,10.0): print("p", m, repr(hw.zcb(m)))
hwf=HW(A,SIGMA,R0,curve=([1.0,5.0],[0.97,0.86]))
print("curve p 2.5", repr(hwf.zcb(2.5)))
print("fwd 2.0", repr(hw.fwd(2.0)))
print("theta 2.0", repr(hw.theta(2.0)))
print("sim", [repr(v) for v in hw.simulate(2.0,8,2004)])
