#coding=GBK
global t
t=np.linspace(0,1.1, 101)
k=4
#print momentum
x100=x[100,:,:]
for i in range(N):
    P[i].set_data(x100[i,0]-x100[k,0],x100[i,1]-x100[k,1])
