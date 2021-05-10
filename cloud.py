from typing import AnyStr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as nimation
from matplotlib import cm

maxIter = 100
#lattice size
Nx = 80
Ny = 40
#heat diffusion
lam = 0.2 #0.1 to 0.4
#viscosity
eta = 0.2 #0.1 to 0.4
nu = 0.2  #0.1 to 0.4
#adiabatic expansion rate
beta = 0.2 #0.15 to 0.35
#velosity of liquid drops
V = 0.2 # 0.05 to 0.5
#coefficient for the dragging force
gamma = 0.2 #0.05 to 0.5
#phase transition rate
a = 0.2 
#coefficient of bouyancy
cp = 0.2 #0.1 to 0.4
#latent heat
Q = 0.2

# change these parameter. W is conserved
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
    def buoyancy_dragging(self):
        self.vy = self.vy+cp*(np.roll(self.T,+1,1)+np.roll(self.T,-1,1)-2*self.T)/2 -gamma*self.wl*(self.vy -V)
    def viscosity_impres(self):
        self.vx = self.vx + nu*lap(self.vx)+ eta*impres(self.vx,self.vy)[0]
        self.vy = self.vy + nu*lap(self.vy)+eta*impres(self.vx,self.vy)[1]
    def t_diff_expansion(self):
        self.T = self.T +lam*lap(self.T)-beta*self.vy
    def diff(self):
        self.wv = self.wv +lam*lap(self.wv)
    def transition(self):
        self.wv += transmission(self.wv,self.wl,self.T)[0]
        self.wl += transmission(self.wv,self.wl,self.T)[1]
        self.T += transmission(self.wv,self.wl,self.T)[2]
    def adv(self):
        self.T  = advection(self.T,self.vx,self.vy-V)
        self.vx = advection(self.vx,self.vx,self.vy-V)
        self.vy = advection(self.vy,self.vx,self.vy-V)
        self.wl = advection(self.wl,self.vx,self.vy-V)
        self.wv = advection(self.wv,self.vx,self.vy-V)
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
    return A

def transmission(wv,wl,T):
    for i in range(Ny):
        for j in range(Nx):
            if np.exp(Q/(T[i,j]+T0)) > wl[i,j]+wv[i,j]:
                dwvdt = a*(wv[i,j]-np.exp(Q/(T[i,j]+T0)))
                dwldt = -a*(wv[i,j]-np.exp(Q/(T[i,j]+T0)))
                dTdt = -Q*(dwvdt-dwldt)
            else:
                dwvdt = a*(wv[i,j]-wl[i,j]+wv[i,j])
                dwldt = -a*(wv[i,j]-wl[i,j]+wv[i,j])
                dTdt = -Q*(dwvdt-dwldt)
    return [dwvdt,dwldt,dTdt]

#ここがうまくいっていない
def impres(vx,vy):
    imx = np.zeros((Ny,Nx))
    imy = np.zeros((Ny,Nx))
    for i in range(1,Ny-2):
        for j in range(Nx-2):
            imx[i,j] = vy[i+2,j]+vy[i,j]-2*vx[i+1,j]+vx[i+1,j+1]-vx[i+1,j]-vx[i,j+1]+vy[i,j]
            imy[i,j] = vx[i,j+2]+vx[i,j]-2*vx[i,j+1]+vy[i+1,j+1]-vy[i,j+1]-vy[i+1,j]+vy[i,j]
    imx[:,Nx-1] = vx[:,0]
    imy[:,Nx-1] = vy[:,0]
    imx[0,:] = 0
    imx[Ny-1,:] = 0
    imy[Ny-1,:] = 0
    imy[Ny-1,:] = 0
    for i in range(Ny-2):
        imy[i:Nx-2] = vy[i+2,Nx-2]+vy[i,Nx-2]-2*vx[i+1,Nx-2]+vx[i+1,Nx-1]-vx[i+1,Nx-2]-vx[i,Nx-1]+vy[i,Nx-2]
        imx[i,Nx-2] = vx[i,Nx-1]+vx[i,Nx-2]-2*vx[i,Nx-1]+vy[i+1,Nx-1]-vy[i,Nx-1]-vy[i+1,j]+vy[i,Nx-2]
    for j in range(1,Nx-2):
        imy[Ny-2,j] = vy[2,j]+vy[Ny-2,j]-2*vx[Ny-1,j]+vx[Ny-1,j+1]-vx[Ny-1,j]-vx[Ny-2,j+1]+vy[Ny-2,j]
        imx[Ny-2,j] = vx[Ny-2,j+2]+vx[Ny-2,j]-2*vx[Ny-2,j+1]+vy[Ny-1,j+1]-vy[Ny-2,j+1]-vy[Ny-1,j]+vy[Ny-2,j]
    imy[Ny-2,Nx-2] = vy[Ny-2,Nx-1]+vy[Ny-2,Nx-2]-2*vx[Ny-2,Nx-1]+vx[Ny-2,Nx-2]-vx[Ny-2,Nx-1]-vx[Ny-1,Nx-2]+vy[Ny-2,Nx-2]
    imx[Ny-2,Nx-2] = vx[Ny-2,Nx-1]+vx[Ny-2,Nx-2]-2*vx[Ny-2,Nx-1]+vy[Ny-2,Nx-2]-vy[Ny-2,Nx-1]-vy[Ny-1,Nx-2]+vy[Ny-2,Nx-2]
    return [imx,imy]

cloud = cal()

for time in range(maxIter):
    cloud.boundary()
    cloud.t_diff_expansion()
    cloud.buoyancy_dragging()
    cloud.viscosity_impres()
    cloud.diff()
    cloud.transition()
    cloud.adv
    print(time)
