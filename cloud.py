from typing import AnyStr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as nimation
from matplotlib import cm

maxIter = 200
Nx = 80
Ny = 40
lam = 0.4
eta = 0.2
nu = 0.2
beta = 0.2
V = 0.02
gamma = 0.1
a = 0.2
cp = 0.01
Q = 0.2
T0 = 3.0
W=0.006

class cal():
    def __init__(self):
        self.Nx = Nx
        self.Ny = Ny
        self.vx = np.zeros((self.Ny,self.Nx))
        self.vy = np.zeros((self.Ny,self.Nx))
        self.T = np.zeros((self.Ny,self.Nx))
        self.wv = np.full((self.Ny,self.Nx),W/(self.Nx*self.Ny*2))
        self.wl = np.full((self.Ny,self.Nx),W/(self.Nx*self.Ny*2))
    def buoyancy(self):
        self.vy = self.vy+cp*(np.roll(self.T,+1,1)+np.roll(self.T,-1,1)-2*self.T)/2
    def viscosity(self):
        self.vx = self.vx + eta*lap(self.vx)
        self.vy = self.vy + eta*lap(self.vy)
    def t_diff(self):
        self.T = self.T +lam*lap(self.T)
    def diff(self):
        self.wv = self.wv +lam*lap(self.wv)
    def incompressive(self):
        self.vx = impres(self.vx,self.vy)[0]
        self.vy = impres(self.vx,self.vy)[1]
    def adv(self):
        self.T = advection(self.T,self.vx,self.vy-V)
        self.vx = advection(self.vx,self.vx,self.vy-V)
        self.vy = advection(self.vy,self.vx,self.vy-V)
        print(self.vx)
        print(self.vy)
        self.wl = advection(self.wl,self.vx,self.vy-V)
        self.wv = advection(self.wv,self.vx,self.vy-V)
    def expansion(self):
        self.T = self.T -beta*self.vy
    def dragging(self):
        self.vy = self.vy -gamma*self.wl*(self.vy -V)
    def trans(self):
        self.wv += transmission(self.wv,self.wl,self.T)[0]
        self.wl += transmission(self.wv,self.wl,self.T)[1]
        self.T += transmission(self.wv,self.wl,self.T)[2]
    def boundary(self):
        self.vx[:,0] = self.vx[:,self.Nx-1]
        self.vy[:,0] = self.vy[:,self.Nx-1]
        self.T[:,0] = self.T[:,self.Nx-1]
        self.wv[:,0] = self.wv[:,self.Nx-1]
        self.wl[:,0] = self.wl[:,self.Nx-1]
        self.vx[0,:] = 0
        self.vx[self.Ny-1,:] = 0
        self.vy[0,:] = 0
        self.vy[self.Ny-1,:] = 0
        self.T[self.Ny-1,:] = T0
        self.T[0,:] = self.T[1,:]

def lap(A):
    return (np.roll(A,+1,1)+np.roll(A,-1,1)+np.roll(A,+1,0)+np.roll(A,-1,0)-4*A)/4

def advection(A,vx,vy):
    for i in range(Ny-1):
        for j in range(Nx-1):
            y1 = int(np.floor(i+vy[i,j])) if int(np.floor(i+vy[i,j])) <= Ny-1 else Ny-1
            y2 = int(np.floor(i+vy[i,j])) if int(np.floor(i+vy[i,j])) <= Ny-1 else Ny-1
            y3 = int(np.floor(i+vy[i,j]+1)) if int(np.floor(i+vy[i,j])) <= Ny-1 else Ny-1
            y4 = int(np.floor(i+vy[i,j])+1) if int(np.floor(i+vy[i,j])) <= Ny-1 else Ny-1

            A[y1,int((np.floor(j+vx[i,j]))%(Nx-1))] = (1-vx[i,j])*(1-vy[i,j])*A[i,j]
            A[y2,int((np.floor(j+vx[i,j]+1))%(Nx-1))] = vx[i,j]*(1-vy[i,j])*A[i,j]
            A[y3,int((np.floor(j+vx[i,j]))%(Nx-1))] = (1-vx[i,j])*vy[i,j]*A[i,j]
            A[y4,int((np.floor(j+vx[i,j]+1))%(Nx-1))] = vx[i,j]*vy[i,j]*A[i,j]
    #A[int(np.floor(Ny-1+vy[Ny-1,Nx-1])),int(np.floor(Nx-1+vx[Ny-1,Nx-1]))] = (1-vx[i,j])*(1-vy[i,j])*A[i,j]
    #A[int(np.floor(Ny-1+vy[Ny-1,Nx-1]+1)),int(np.floor(Nx-1+vx[Ny-1,Nx-1]))] = vx[i,j]*(1-vy[i,j])*A[i,j]
    #A[int(np.floor(Ny-1+vy[Ny-1,Nx-1])),int(np.floor(vx[Ny-1,Nx-1]))] = (1-vx[i,j])*vy[i,j]*A[i,j]
    #A[int(np.floor(Ny-1+vy[Ny-1,Nx-1]+1)),int(np.floor(vx[Ny-1,Nx-1]))] = vx[i,j]*vy[i,j]*A[i,j]
    return A

def transmission(wv,wl,T):
    for i in range(Ny):
        for j in range(Nx):
            dwvdt = a*(wv[i,j]-np.exp(Q*wl[i,j]/(cp*T[i,j])))
            dwldt = -a*(wv[i,j]-np.exp(Q*wl[i,j]/(cp*T[i,j])))
            dTdt = -Q*(dwvdt-dwldt)
    return [dwvdt,dwldt,dTdt]

def impres(vx,vy):
    for i in range(1,Ny-2):
        for j in range(Nx-2):
            vy[i,j] += vy[i+2,j]+vy[i,j]-2*vx[i+1,j]+vx[i+1,j+1]-vx[i+1,j]-vx[i,j+1]+vy[i,j]
            vx[i,j] += vx[i,j+2]+vx[i,j]-2*vx[i,j+1]+vy[i+1,j+1]-vy[i,j+1]-vy[i+1,j]+vy[i,j]
    vx[:,Nx-1] = vx[:,0]
    vy[:,Nx-1] = vy[:,0]
    vx[0,:] = 0
    vx[Ny-1,:] = 0
    vy[Ny-1,:] = 0
    vy[Ny-1,:] = 0
    for i in range(Ny-2):
        vy[i:Nx-2] += vy[i+2,Nx-2]+vy[i,Nx-2]-2*vx[i+1,Nx-2]+vx[i+1,Nx-1]-vx[i+1,Nx-2]-vx[i,Nx-1]+vy[i,Nx-2]
        vx[i,Nx-2] += vx[i,Nx-1]+vx[i,Nx-2]-2*vx[i,Nx-1]+vy[i+1,Nx-1]-vy[i,Nx-1]-vy[i+1,j]+vy[i,Nx-2]
    for j in range(1,Nx-2):
        vy[Ny-2,j] += vy[2,j]+vy[Ny-2,j]-2*vx[Ny-1,j]+vx[Ny-1,j+1]-vx[Ny-1,j]-vx[Ny-2,j+1]+vy[Ny-2,j]
        vx[Ny-2,j] += vx[Ny-2,j+2]+vx[Ny-2,j]-2*vx[Ny-2,j+1]+vy[Ny-1,j+1]-vy[Ny-2,j+1]-vy[Ny-1,j]+vy[Ny-2,j]
    vy[Ny-2,Nx-2] += vy[Ny-2,Nx-1]+vy[Ny-2,Nx-2]-2*vx[Ny-2,Nx-1]+vx[Ny-2,Nx-2]-vx[Ny-2,Nx-1]-vx[Ny-1,Nx-2]+vy[Ny-2,Nx-2]
    vx[Ny-2,Nx-2] += vx[Ny-2,Nx-1]+vx[Ny-2,Nx-2]-2*vx[Ny-2,Nx-1]+vy[Ny-2,Nx-2]-vy[Ny-2,Nx-1]-vy[Ny-1,Nx-2]+vy[Ny-2,Nx-2]
    return [vx,vy]

cloud = cal()

for time in range(maxIter):
    cloud.boundary()
    cloud.buoyancy()
    cloud.dragging()
    cloud.viscosity()
    cloud.incompressive()
    cloud.diff()
    cloud.t_diff()
    cloud.expansion()
    cloud.trans()
    cloud.adv()
