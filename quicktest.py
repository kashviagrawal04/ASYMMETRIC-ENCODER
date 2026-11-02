import numpy as np, time
from asym_ae.data import load
from asym_ae.models import AutoEncoder
from asym_ae import evaluate as E
Xtr,Xte,ytr,yte=load()
# gradient check
ae=AutoEncoder("symmetric",4,seed=0); x=Xtr[:8]
def loss():
    return ((ae.reconstruct(x)-x)**2).sum()/x.size
z=ae.enc.forward(x,train=True); y=ae.dec.forward(z,train=True); dy=2*(y-x)/x.size
dWd,_,dz=ae.dec.backward(dy); dWe,_,_=ae.enc.backward(dz)
i,j=3,5; eps=1e-6; W=ae.enc.W[0]; W[i,j]+=eps; l1=loss(); W[i,j]-=2*eps; l2=loss(); W[i,j]+=eps
print("grad check", dWe[0][i,j], (l1-l2)/(2*eps))
t=time.time(); ae=AutoEncoder("symmetric",8,seed=0); h=ae.fit(Xtr,epochs=50); print("50 ep",time.time()-t,"s mse",h[-1])
print("probe",E.probe(ae.encode(Xtr),ytr,ae.encode(Xte),yte))
