import sys, math; sys.path.insert(0, "scripts")
from refgen import normals

KAPPA, THETA, SIGMA, R0 = 0.15, 0.05, 0.02, 0.03

def feller_theta(kappa, theta, sigma):
    if 2*kappa*theta - sigma**2 <= 0:
        return sigma**2/(2*max(kappa,1e-12)) + 1e-6
    return theta

def zcb(kappa, theta, sigma, r0, maturity):
    if maturity <= 0: return 1.0
    kappa = max(kappa,1e-12); theta = max(theta,0.0); sigma = max(sigma,1e-12)
    r0 = max(r0,0.0)
    gamma = math.sqrt(kappa**2 + 2*sigma**2)
    egt = math.exp(gamma*maturity)
    den = (gamma+kappa)*(egt-1) + 2*gamma
    den = max(den,1e-12)
    b = 2*(egt-1)/den
    a = max(2*gamma*math.exp((kappa+gamma)*maturity/2)/den, 1e-12) ** (2*kappa*theta/sigma**2)
    return a*math.exp(-b*r0)

def zcy(kappa, theta, sigma, r0, maturity):
    return -math.log(max(zcb(kappa,theta,sigma,r0,maturity),1e-12))/max(maturity,1e-12)

def simulate(kappa, theta, sigma, r0, t, n, seed):
    theta = feller_theta(kappa, theta, sigma)
    kappa = max(kappa,0.0); sigma = max(sigma,0.0); r0 = max(r0,0.0)
    mr = math.exp(-kappa*t)
    mean = theta + (r0-theta)*mr
    if kappa*t >= 1.0: mean = theta
    if sigma == 0: return [max(mean,0.0)]*n
    sv = sigma**2*theta/max(2*kappa,1e-12)
    var = sv*(1-math.exp(-2*kappa*t))
    std = math.sqrt(max(var,0))
    return [max(mean + std*z, 0.0) for z in normals(seed, n)]

for m in (1.0, 5.0, 10.0): print("p", m, repr(zcb(KAPPA,THETA,SIGMA,R0,m)))
print("y5", repr(zcy(KAPPA,THETA,SIGMA,R0,5.0)))
print("sim", [repr(v) for v in simulate(KAPPA,THETA,SIGMA,R0,2.0,8,2003)])
print("feller", [repr(v) for v in simulate(0.1, 0.1, 0.3, 0.03, 0.0, 1, 2003)])
