import sys, math; sys.path.insert(0, "scripts")
from refgen import normals

def simulate(lv, s0, t, n, steps, rate, seed):
    dt=t/steps; sdt=math.sqrt(dt)
    spot=[s0]*n
    z=normals(seed, n*steps); idx=0
    for step in range(steps):
        time=(step+1)*dt
        diff=[lv(s,time)*sdt for s in spot]
        new=[]
        for i in range(n):
            zi=z[idx]; idx+=1
            new.append(spot[i]*math.exp(rate*dt+diff[i]*zi))
        spot=new
    return spot

print("const", [repr(v) for v in simulate(lambda s,t: 0.2, 100.0, 1.0, 6, 32, 0.05, 2009)])
print("state", [repr(v) for v in simulate(lambda s,t: 0.1+0.2*min(s/100.0,2.0)*(1.0/(1.0+t)), 100.0, 0.5, 4, 16, 0.02, 2009)])
