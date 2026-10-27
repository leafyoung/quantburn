import sys, math, cmath; sys.path.insert(0, "scripts")
from refgen import normals

EPS=1e-8; PI=math.pi
KAPPA,THETA,SIGMA_V,RHO,V0,MU = 1.0,0.04,0.5,-0.5,0.04,0.0

def simulate_terminal_paths(s0,t,n,steps,drift,antithetic,normalize,seed):
    spot=max(s0,EPS)
    if t<=0: return [spot]*n
    steps=max(1,steps); dt=t/steps; sqrt_dt=math.sqrt(dt)
    base_count=(n+1)//2 if antithetic else n
    rho_res=math.sqrt(max(1.0-RHO**2,EPS))
    prices=[spot]*base_count; variance=[V0]*base_count
    if antithetic:
        prices=(prices*2)[:n]; variance=(variance*2)[:n]
    rng_n = normals(seed, base_count*2*steps)
    idx=0
    for _ in range(steps):
        z1=[rng_n[idx+k] for k in range(base_count)]; idx+=base_count
        z2=[rng_n[idx+k] for k in range(base_count)]; idx+=base_count
        if antithetic:
            z1=(z1+[-v for v in z1])[:n]; z2=(z2+[-v for v in z2])[:n]
        new_var=[]
        for i in range(n):
            zv=RHO*z1[i]+rho_res*z2[i]
            vp=max(variance[i],0.0)
            nv=max(variance[i]+KAPPA*(THETA-vp)*dt+SIGMA_V*math.sqrt(max(vp,EPS))*sqrt_dt*zv, 0.0)
            prices[i]*=math.exp((drift-0.5*vp)*dt+math.sqrt(max(vp,EPS))*sqrt_dt*z1[i])
            new_var.append(nv)
        variance=new_var
    if normalize:
        target=spot*math.exp(drift*t)
        m=max(sum(prices)/len(prices),EPS)
        prices=[p*(target/m) for p in prices]
    return prices

def cf(u, spot, expiry, rate, div):
    a=KAPPA*THETA
    iu=1j*u
    d=cmath.sqrt((RHO*SIGMA_V*iu-KAPPA)**2 + SIGMA_V**2*(u*u+iu))
    g=(KAPPA-RHO*SIGMA_V*iu-d)/(KAPPA-RHO*SIGMA_V*iu+d)
    e=cmath.exp(-d*expiry)
    c=(rate-div)*iu*expiry + (a/SIGMA_V**2)*((KAPPA-RHO*SIGMA_V*iu-d)*expiry - 2.0*cmath.log((1-g*e)/(1-g)))
    dt_=(KAPPA-RHO*SIGMA_V*iu-d)/SIGMA_V**2*((1-e)/(1-g*e))
    return cmath.exp(c + dt_*V0 + iu*cmath.log(spot))

def cf_price(s0,k,t,r,option_type,q=0.0,limit=100.0,points=2000):
    spot=max(s0,EPS); strike=max(k,EPS)
    if t<=0:
        return max(spot-strike,0) if option_type=="call" else max(strike-spot,0)
    n=max(256,points)
    u=[1e-5 + (limit-1e-5)*i/(n-1) for i in range(n)]
    logk=math.log(strike)
    phi_mi=cf(0.0-1j,spot,t,r,q)
    p1i=[];p2i=[]
    for uu in u:
        kernel=cmath.exp(-1j*uu*logk)/(1j*uu)
        p1i.append((kernel*cf(uu-1j,spot,t,r,q)/phi_mi).real)
        p2i.append((kernel*cf(uu,spot,t,r,q)).real)
    def trapz(y):
        return sum(0.5*(y[i]+y[i+1])*(u[i+1]-u[i]) for i in range(n-1))
    p1=0.5+trapz(p1i)/PI
    p2=0.5+trapz(p2i)/PI
    dd=math.exp(-q*t); dr=math.exp(-r*t)
    call=spot*dd*p1-strike*dr*p2
    if option_type=="call": return max(call,0.0)
    return max(call-spot*dd+strike*dr,0.0)

def mc_price(s0,k,t,r,option_type,q=0.0,N=64,steps=32,seed=2007):
    tp=simulate_terminal_paths(s0,t,N,steps,r-q,True,True,seed)
    if option_type=="call":
        pay=sum(max(s-k,0) for s in tp)/len(tp)
    else:
        pay=sum(max(k-s,0) for s in tp)/len(tp)
    return math.exp(-r*t)*pay

print("sim", [repr(v) for v in simulate_terminal_paths(100.0,1.0,6,32,MU,False,False,2007)])
print("cf call", repr(cf_price(100.0,100.0,1.0,0.05,"call")))
print("cf put", repr(cf_price(100.0,100.0,1.0,0.05,"put")))
print("cf otm", repr(cf_price(100.0,120.0,0.5,0.03,"call",q=0.01)))
print("mc call", repr(mc_price(100.0,100.0,1.0,0.05,"call")))
print("mc put", repr(mc_price(100.0,100.0,1.0,0.05,"put")))
