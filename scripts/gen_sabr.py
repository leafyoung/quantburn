import sys, math; sys.path.insert(0, "scripts")
from refgen import normals

EPS=1e-8; ATMTOL=1e-7
ALPHA,BETA,RHO,NU,F0,SHIFT = 0.2,0.5,0.0,0.3,100.0,0.0

def ncdf(x): return 0.5*(1.0+math.erf(x/math.sqrt(2)))
def npdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def black76(ot,f,k,t,v,r):
    df=math.exp(-r*t)
    intr=max(f-k,0) if ot=="call" else max(k-f,0)
    if t<=0 or v<=0: return df*intr
    vt=v*math.sqrt(t)
    d1=(math.log(f/k)+0.5*v*v*t)/vt; d2=d1-vt
    if ot=="call": return df*(f*ncdf(d1)-k*ncdf(d2))
    return df*(k*ncdf(-d2)-f*ncdf(-d1))

def bachelier(ot,f,k,t,v,r):
    df=math.exp(-r*t)
    intr=max(f-k,0) if ot=="call" else max(k-f,0)
    if t<=0 or v<=0: return df*intr
    se=math.sqrt(t)
    den=max(v*se,EPS)
    d=(f-k)/den
    call=df*((f-k)*ncdf(d)+v*se*npdf(d))
    return call if ot=="call" else call-df*(f-k)

def atm_ln(f,t):
    f=max(f,EPS); omb=1-BETA; fp=f**omb
    corr=(omb**2/24)*(ALPHA**2/max(f**(2*omb),EPS)) + (RHO*BETA*NU*ALPHA)/(4*max(fp,EPS)) + ((2-3*RHO**2)/24)*NU**2
    return max((ALPHA/max(fp,EPS))*(1+corr*t),EPS)

def hagan_ln(k,t,forward=None,shift=None):
    if t<=0: return 0.0
    bf=F0 if forward is None else forward
    sh=SHIFT if shift is None else shift
    sf=max(bf+sh,EPS); sk=max(k+sh,EPS)
    logfk=math.log(sf/sk)
    if abs(logfk)<ATMTOL: return atm_ln(sf,t)
    omb=1-BETA
    fkb=max((sf*sk)**(0.5*omb),EPS)
    z=(NU/max(ALPHA,EPS))*fkb*logfk
    st=math.sqrt(max(1-2*RHO*z+z*z,EPS))
    xz=math.log(max((st+z-RHO)/(1-RHO),EPS))
    ld=1+(omb**2/24)*logfk**2+(omb**4/1920)*logfk**4
    pre=ALPHA/max(fkb*ld,EPS)
    corr=(omb**2/24)*(ALPHA**2/max((sf*sk)**omb,EPS)) + (RHO*BETA*NU*ALPHA)/(4*max(fkb,EPS)) + ((2-3*RHO**2)/24)*NU**2
    ratio=1.0 if abs(xz)<ATMTOL else z/xz
    return max(pre*ratio*(1+corr*t),EPS)

def atm_n(f,t):
    f=max(f,EPS); fb=f**BETA; omb=1-BETA
    corr=-BETA*(2-BETA)*ALPHA**2/(24*max(f**(2*omb),EPS)) + (RHO*BETA*NU*ALPHA)/(4*max(f**omb,EPS)) + ((2-3*RHO**2)/24)*NU**2
    return max(ALPHA*fb*(1+corr*t),EPS)

def hagan_n(k,t,forward=None,shift=None):
    if t<=0: return 0.0
    bf=F0 if forward is None else forward
    sh=SHIFT if shift is None else shift
    sf=bf+sh; sk=k+sh
    diff=sf-sk
    if abs(diff)<ATMTOL: return atm_n(max(sf,EPS),t)
    fkb=max(max(sf*sk,EPS)**(0.5*BETA),EPS)
    zeta=(NU/max(ALPHA,EPS))*diff/fkb
    st=math.sqrt(max(1-2*RHO*zeta+zeta*zeta,EPS))
    xz=math.log(max((st+zeta-RHO)/(1-RHO),EPS))
    prod=max(sf*sk,EPS)
    corr=-BETA*(2-BETA)*ALPHA**2/(24*max(prod**(1-BETA),EPS)) + (RHO*BETA*NU*ALPHA)/(4*max(prod**(0.5*(1-BETA)),EPS)) + ((2-3*RHO**2)/24)*NU**2
    ratio=1.0 if abs(xz)<ATMTOL else diff/xz
    return max(ALPHA*ratio*(1+corr*t),EPS)

def base_forward(level=None,t=0.0,rate=None,div=0.0):
    f=F0 if level is None else level
    if rate is None: return f
    return f*math.exp((rate-div)*t)

def iv(k,t,r=0.0,q=0.0,model="lognormal",shift=None):
    fwd=base_forward(t=t,rate=r,div=q)
    if model=="lognormal": return hagan_ln(k,t,forward=fwd,shift=shift)
    return hagan_n(k,t,forward=fwd,shift=shift)

def option_price(k,t,r,ot,q=0.0,model="lognormal",shift=None):
    fwd=base_forward(t=t,rate=r,div=q)
    if model=="lognormal":
        sh=SHIFT if shift is None else shift
        return black76(ot,fwd+sh,k+sh,t,iv(k,t,r,q,"lognormal",shift),r)
    sh=SHIFT if shift is None else shift
    return bachelier(ot,fwd+sh,k+sh,t,iv(k,t,r,q,"normal",shift),r)

def simulate(t,n,steps,seed):
    steps=max(1,steps); dt=t/steps; sdt=math.sqrt(dt)
    rho_res=math.sqrt(max(1-RHO**2,EPS))
    fwd=[max(F0,EPS)]*n; alp=[ALPHA]*n
    z=normals(seed, n*2*steps)
    idx=0
    for _ in range(steps):
        z1=z[idx:idx+n]; idx+=n
        z2=z[idx:idx+n]; idx+=n
        new=[]
        for i in range(n):
            za=RHO*z1[i]+rho_res*z2[i]
            fb=max(fwd[i],EPS)
            diff=alp[i]*(fb**BETA)*sdt*z1[i]
            fwd[i]=max(fwd[i]+diff,EPS)
            alp[i]=max(alp[i]*math.exp(-0.5*NU*NU*dt+NU*sdt*za),EPS)
    return fwd

print("iv ln 110", repr(iv(110.0,1.0)))
print("iv ln atm", repr(iv(100.0,1.0)))
print("iv n 110", repr(iv(110.0,1.0,model="normal")))
print("iv n atm", repr(iv(100.0,1.0,model="normal")))
print("call ln 105 r3%", repr(option_price(105.0,1.0,0.03,"call")))
print("put ln 95 r3%", repr(option_price(95.0,1.0,0.03,"put")))
print("call n 105 r3%", repr(option_price(105.0,1.0,0.03,"call",model="normal")))
print("shifted call 0.01", repr(option_price(0.01,1.0,0.0,"call",model="lognormal",shift=0.05)))
print("sim", [repr(v) for v in simulate(1.0,6,32,2008)])
