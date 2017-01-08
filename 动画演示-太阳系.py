#coding=GBK
sDescribe='''模拟太阳系中各个天体的运行规律'''
from scipy.integrate import odeint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

G=1.0 #万有引力常数
#在此处添加天体，包含天体的质量，颜色，天体的初始位置，天体的初始速度可以随意追加
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


#定义并初始化一些全局变量
global x #系统的当前状态
global N #天体的个数
global M#天体的质量表
global init_state#系统的初始状态
global momentum #体系的动量
global P#各个天体在图像上对应的点
global t#每更新一次的时间间采样
t = np.linspace(0,1, 101)  # 创建时间点
momentum=[0,0]
N=len(param)
P=[]
M=[]
x=np.zeros(101*N*4).reshape(101,N,4) #4个相空间的坐标（x,y,vx,vy)
for i in range(N):
    M.append(param['P%d'%(i)]['m'])
    sd=param['P%d'%(i)]['init_state']
    state=np.array([sd['x'],sd['y'],sd['vx'],sd['vy']])
    if(i==0):
        init_state=state
    else:
        init_state=np.vstack((init_state,state))
    


#定义该问题的微分方程
def Planet(w, t, par):
    #w包含，x,y,vx,vy坐标，t时间，param中心天梯的质量，万有引力常数等信息
    G=par[0]#万有引力常数
    N=par[1]#天体个数
    M=par[2]#所有的天体质量表
    
    #重新整理各个变量，成为N*4的矩阵
    w=w.reshape(N,4) #各个自变量状态
    ret=np.zeros((N,4)) #返回各个状态的导数

    #需要求出以下中间值
    DX=np.zeros((N,N)) #各个星球之间x方向差值
    DY=np.zeros((N,N)) #各个星球之间y方向的差值
    R=np.zeros((N,N)) #各个星球之间的距离
    for i in range(N):
        for j in range(N):
            DX[i,j]=w[i,0]-w[j,0]
            DY[i,j]=w[i,1]-w[j,1]
            R[i,j]=np.sqrt(DX[i,j]**2+DY[i,j]**2)

    #求返回矩阵
    for i in range(N):
        ret[i,0]=w[i,2] #x的导数为vx
        ret[i,1]=w[i,3] #y的导数为vy
        #求第i个天体受到的万有引力作用下的加速度
        for j in range(N):
            if(i==j):
                pass #万有引力是相对于两个不同的物体之间的作用力
            else:
                K=-G*M[i]*M[j]/(R[i,j]**3)#根据万有引力公式求出i与j之间的相互作用力
                ret[i,2]+=K*DX[i,j]/M[i]#x方向上的加速度的增量
                ret[i,3]+=K*DY[i,j]/M[i]#y方向上的加速度的增量
        
    return ret.reshape(N*4)#必须返回一维数组的形式


def UpdateTrack(state):
    # 调用 odeint 对天体轨迹进行求解
    global x,P,N,M,G,momentum,t
    x = odeint(Planet, state, t, args=([G,N,M],))
    x=x.reshape(101,N,4)
    #修正质点的位置
    momentum=[0,0]
    for i in range(N):
        momentum[0]+=M[i]*x[100,i,2]
        momentum[1]+=M[i]*x[100,i,3]
        #P[i].set_data(x[100,i,0],x[100,i,1])
    execfile('001.py')

def UpdateState():
    print 'yield'
    global N,init_state
    yield init_state.reshape(N*4)#初始条件
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
    ani = animation.FuncAnimation(fig, UpdateTrack, UpdateState, interval=100) #创建动画对象
    plt.xlim(-2800,2800)#限制x的范围
    plt.ylim(-2800,2800)#限制y的范围
    plt.show()    
    
if __name__ == '__main__':
    Run()
