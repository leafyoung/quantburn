import sys, math; sys.path.insert(0, "scripts")
from refgen import normals

KAPPA, THETA, SIGMA, R0 = 0.15, 0.05, 0.02, 0.03

def zcb(kappa, theta, sigma, r0, maturity):
    if maturity <= 0: return 1.0
    kappa = max(kappa, 1e-12); sigma = max(sigma, 0.0)
    b = (1-math.exp(-kappa*maturity))/kappa
    a = math.exp((theta - sigma**2/(2*kappa**2))*(b-maturity) - sigma**2*b**2/(4*kappa))
    return a*math.exp(-b*r0)

def zcy(kappa, theta, sigma, r0, maturity):
    return -math.log(max(zcb(kappa,theta,sigma,r0,maturity),1e-12))/max(maturity,1e-12)

def simulate(kappa, theta, sigma, r0, t, n, seed):
    kappa = max(kappa,0.0); sigma = max(sigma,0.0)
    mr = math.exp(-kappa*t)
    mean = theta + (r0-theta)*mr
    if kappa*t >= 1.0: mean = theta
    if sigma == 0: return [mean]*n
    var = sigma**2*(1-math.exp(-2*kappa*t))/max(2*kappa,1e-12)
    std = math.sqrt(max(var,0))
    return [mean + std*z for z in normals(seed, n)]

for m in (1.0, 5.0, 10.0): print("p", m, repr(zcb(KAPPA,THETA,SIGMA,R0,m)))
print("y1", repr(zcy(KAPPA,THETA,SIGMA,R0,1.0)))
print("sim", [repr(v) for v in simulate(KAPPA,THETA,SIGMA,R0,2.0,8,2002)])
