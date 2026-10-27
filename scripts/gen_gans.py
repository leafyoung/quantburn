import numpy as np

def linear(x, W, b):
    W=np.array(W,dtype=np.float64); b=np.array(b,dtype=np.float64); x=np.array(x,dtype=np.float64)
    return W@x + b
def relu(x): return np.maximum(np.array(x),0.0)
def lrelu(x,ns=0.2): return np.where(np.array(x)>=0, np.array(x), ns*np.array(x))
def tanh(x): return np.tanh(np.array(x))
def sigmoid(x): return 1.0/(1.0+np.exp(-np.array(x)))

# generator_forward: Linear(3->2 test shape per Rust test)
x=[0.5,-1.25,2.0]
w1=[[0.1,-0.2,0.3],[0.4,0.5,-0.6]]; b1=[0.05,-0.05]
w2=[[1.0,-1.0],[0.25,0.75],[-0.5,2.0]]; b2=[0.1,0.0,-0.1]
w3=[[0.3,-0.4,0.5],[0.6,0.7,-0.8]]; b3=[0.02,-0.02]
h1=relu(linear(x,w1,b1)); h2=relu(linear(h1,w2,b2)); y=tanh(linear(h2,w3,b3))
print("gen", [repr(float(v)) for v in y])

# discriminator
xd=[0.3,-0.7]
w1d=[[0.2,0.4],[-0.1,0.3],[0.5,-0.25]]; b1d=[0.0,0.1,-0.1]
w2d=[[0.3,-0.2,0.6],[0.1,0.9,-0.4]]; b2d=[0.05,-0.05]
w3d=[[0.7,-0.3]]; b3d=[0.01]
h1=lrelu(linear(xd,w1d,b1d)); h2=lrelu(linear(h1,w2d,b2d)); d=sigmoid(linear(h2,w3d,b3d))
print("disc", repr(float(d[0])))

p=[0.9,0.2,0.7]; tg=[1.0,0.0,1.0]
loss=-np.mean([t*max(pp,1e-12)*0 + t*np.log(max(pp,1e-12)) + (1-t)*np.log(max(1-pp,1e-12)) for pp,t in zip(p,tg)])
print("bce", repr(float(loss)))
