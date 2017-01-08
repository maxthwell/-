#coding=GBK
sDescribe='''ģ��̫��ϵ�и�����������й���'''
from scipy.integrate import odeint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

G=1.0 #������������
#�ڴ˴�������壬�����������������ɫ������ĳ�ʼλ�ã�����ĳ�ʼ�ٶȿ�������׷��
param={
    'P0':{'m':100,      'color':'red'     ,'init_state':{'x':0,'y':0,'vx':0,'vy':0}},
    'P1':{'m':1,        'color':'orange'  ,'init_state':{'x':100,'y':0,'vx':0,'vy':1.0}},
    'P2':{'m':0.001,    'color':'yellow'  ,'init_state':{'x':101,'y':0,'vx':0,'vy':2.0}},
    'P3':{'m':1,        'color':'green'   ,'init_state':{'x':400,'y':0,'vx':0,'vy':0.5}},
    'P4':{'m':0.001,    'color':'cyan'    ,'init_state':{'x':404,'y':0,'vx':0,'vy':1}},
    'P5':{'m':1,        'color':'blue'    ,'init_state':{'x':900,'y':0,'vx':0,'vy':0.3333333}},
    'P6':{'m':0.001,    'color':'purple'  ,'init_state':{'x':916,'y':0,'vx':0,'vy':0.5833333}},
    'P7':{'m':1,        'color':'blue'    ,'init_state':{'x':1600,'y':0,'vx':0,'vy':0.25}},
    'P8':{'m':0.001,    'color':'peru'    ,'init_state':{'x':1601,'y':0,'vx':0,'vy':1.25}},
    'P9':{'m':0.001,    'color':'seagreen','init_state':{'x':1604,'y':0,'vx':0,'vy':0.75}},
    'P10':{'m':0.001,   'color':'brown'   ,'init_state':{'x':1616,'y':0,'vx':0,'vy':0.5}},
    'P11':{'m':1,       'color':'blue'    ,'init_state':{'x':2500,'y':0,'vx':0,'vy':0.2}},
    'P12':{'m':0.001,   'color':'navy'    ,'init_state':{'x':2501,'y':0,'vx':0,'vy':1.2}},
    'P13':{'m':0.001,   'color':'coral'   ,'init_state':{'x':2504,'y':0,'vx':0,'vy':0.7}},
    'P14':{'m':0.04,    'color':'maroon'  ,'init_state':{'x':2525,'y':0,'vx':0,'vy':0.4}},
    'P15':{'m':0.0001,  'color':'plum'    ,'init_state':{'x':2525.25,'y':0,'vx':0,'vy':0.8}},
    'P16':{'m':0.0001,  'color':'pink'    ,'init_state':{'x':2526,'y':0,'vx':0,'vy':0.6}},
}


#���岢��ʼ��һЩȫ�ֱ���
global x #ϵͳ�ĵ�ǰ״̬
global N #����ĸ���
global M#�����������
global init_state#ϵͳ�ĳ�ʼ״̬
global momentum #��ϵ�Ķ���
global P#����������ͼ���϶�Ӧ�ĵ�
global t#ÿ����һ�ε�ʱ������
t = np.linspace(0,1, 101)  # ����ʱ���
momentum=[0,0]
N=len(param)
P=[]
M=[]
x=np.zeros(101*N*4).reshape(101,N,4) #4����ռ�����꣨x,y,vx,vy)
for i in range(N):
    M.append(param['P%d'%(i)]['m'])
    sd=param['P%d'%(i)]['init_state']
    state=np.array([sd['x'],sd['y'],sd['vx'],sd['vy']])
    if(i==0):
        init_state=state
    else:
        init_state=np.vstack((init_state,state))
    


#����������΢�ַ���
def Planet(w, t, par):
    #w������x,y,vx,vy���꣬tʱ�䣬param�������ݵ�����������������������Ϣ
    G=par[0]#������������
    N=par[1]#�������
    M=par[2]#���е�����������
    
    #�������������������ΪN*4�ľ���
    w=w.reshape(N,4) #�����Ա���״̬
    ret=np.zeros((N,4)) #���ظ���״̬�ĵ���

    #��Ҫ��������м�ֵ
    DX=np.zeros((N,N)) #��������֮��x�����ֵ
    DY=np.zeros((N,N)) #��������֮��y����Ĳ�ֵ
    R=np.zeros((N,N)) #��������֮��ľ���
    for i in range(N):
        for j in range(N):
            DX[i,j]=w[i,0]-w[j,0]
            DY[i,j]=w[i,1]-w[j,1]
            R[i,j]=np.sqrt(DX[i,j]**2+DY[i,j]**2)

    #�󷵻ؾ���
    for i in range(N):
        ret[i,0]=w[i,2] #x�ĵ���Ϊvx
        ret[i,1]=w[i,3] #y�ĵ���Ϊvy
        #���i�������ܵ����������������µļ��ٶ�
        for j in range(N):
            if(i==j):
                pass #���������������������ͬ������֮���������
            else:
                K=-G*M[i]*M[j]/(R[i,j]**3)#��������������ʽ���i��j֮����໥������
                ret[i,2]+=K*DX[i,j]/M[i]#x�����ϵļ��ٶȵ�����
                ret[i,3]+=K*DY[i,j]/M[i]#y�����ϵļ��ٶȵ�����
        
    return ret.reshape(N*4)#���뷵��һά�������ʽ


def UpdateTrack(state):
    # ���� odeint ������켣�������
    global x,P,N,M,G,momentum,t
    x = odeint(Planet, state, t, args=([G,N,M],))
    x=x.reshape(101,N,4)
    #�����ʵ��λ��
    momentum=[0,0]
    for i in range(N):
        momentum[0]+=M[i]*x[100,i,2]
        momentum[1]+=M[i]*x[100,i,3]
        #P[i].set_data(x[100,i,0],x[100,i,1])
    execfile('001.py')

def UpdateState():
    print 'yield'
    global N,init_state
    yield init_state.reshape(N*4)#��ʼ����
    print 'hello'
    while True:
        global x
        yield x[100,:].reshape(N*4)

def Run():
    fig = plt.figure()
    global P
    for i in range(N):
        ls=param['P%d'%(i)]
        planet,=plt.plot(0,0,'o',color=ls['color'])
        P.append(planet)
    ani = animation.FuncAnimation(fig, UpdateTrack, UpdateState, interval=100) #������������
    plt.xlim(-2800,2800)#����x�ķ�Χ
    plt.ylim(-2800,2800)#����y�ķ�Χ
    plt.show()    
    
if __name__ == '__main__':
    Run()
