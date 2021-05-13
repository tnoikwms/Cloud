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
lam = 0.4 #0.1 to 0.4
#viscosity
eta = 0.2 #0.1 to 0.4
nu = 0.2  #0.1 to 0.4
#adiabatic expansion rate
beta = 0.2 #0.15 to 0.35
#velosity of liquid drops
V = 0.02 # 0.05 to 0.5
#coefficient for the dragging force
gamma = 0.1 #0.05 to 0.5
#phase transition rate
a = 0.2 
#coefficient of bouyancy
cp = 0.2 #0.1 to 0.4
#latent heat
Q = 0.2
#gas constant(vapor gas)
Rv = 0.04
# change these parameter. W is conserved
T0 = 3.0
W=0.006

class cal():
    def __init__(self):
        self.Nx = Nx
        self.Ny = Ny
        self.vx = np.zeros((self.Ny,self.Nx))
        self.vy = np.zeros((self.Ny,self.Nx))
        self.T  = np.zeros((self.Ny,self.Nx))
        self.wv = np.zeros((self.Ny,self.Nx))
        self.wl = np.zeros((self.Ny,self.Nx))
        self.wl[self.Ny-1,:] = W/(self.Ny-1)
        self.T[self.Ny-1,:] = T0
    def buoyancy_dragging(self):
        self.vy_1 = self.vy+cp*(np.roll(self.T,+1,1)+np.roll(self.T,-1,1)-2*self.T)/2 -gamma*self.wl*(self.vy -V)
        self.vx_1 = self.vx
        self.vx_1[abs(self.vx_1)>1] = self.vx_1[abs(self.vx_1)>1]/(Nx-1)
        self.vy_1[abs(self.vy_1)>1] = self.vy_1[abs(self.vy_1)>1]/(Nx-1)
    def viscosity_impres(self):
        self.imp = impres(self.vx_1,self.vy_1)
        self.vx_2 = self.vx_1 + nu*lap(self.vx_1) + eta*self.imp[0]
        self.vy_2 = self.vy_1 + nu*lap(self.vy_1) + eta*self.imp[1]
        self.vx_2[abs(self.vx_2)>1] = self.vx_2[abs(self.vx_2)>1]/(Nx-1)
        self.vy_2[abs(self.vy_2)>1] = self.vy_2[abs(self.vy_2)>1]/(Nx-1)
    def t_diff_expansion(self):
        self.T_1 = self.T + lam*lap(self.T)-beta*self.vy
    def diff(self):
        self.wv = self.wv + lam*lap(self.wv)
    def transition(self):
        self.trans= transition(self.wv,self.wl,self.T_1)
        self.wv += self.trans[0]
        self.wl += self.trans[1]
        self.T_2  = self.T_1 + self.trans[2]
    def adv(self):
        self.T  = advection(self.T_2,self.vx_2,self.vy_2-V)
        self.wl = advection(self.wl,self.vx_2,self.vy_2-V)
        self.wv = advection(self.wv,self.vx_2,self.vy_2-V)
        self.vx = advection(self.vx_2,self.vx_2,self.vy_2-V)
        self.vy = advection(self.vy_2,self.vx_2,self.vy_2-V)

        self.vx[abs(self.vx)>1] = self.vx[abs(self.vx)>1]/(Nx-1)
        self.vy[abs(self.vy)>1] = self.vy[abs(self.vy)>1]/(Nx-1)
    def no(self):
        self.wv[self.wv<0] =0
        self.wl[self.wl<0] = 0
        self.sum = np.sum(self.wv)+np.sum(self.wl)
        self.wl = self.wl/(self.sum/W)
        self.wv = self.wv/(self.sum/W)
    def boundary(self):
        self.vx[:,0] = self.vx[:,self.Nx-1]
        self.vy[:,0] = self.vy[:,self.Nx-1]
        self.T[:,0] = self.T[:,self.Nx-1]
        self.wv[:,0] = self.wv[:,self.Nx-1]
        self.wl[:,0] = self.wl[:,self.Nx-1]
        self.vx[0,:] = self.vx[self.Ny-1,:] = 0
        #slip condition
        self.vy[0,:] = self.vy[self.Ny-1,:] = 0
        self.T[self.Ny-1,:] = T0
        self.T[0,:] = self.T[1,:]
        self.wl[0,:] = self.wv[0,:] =0

def lap(A):
    return (np.roll(A,+1,1)+np.roll(A,-1,1)+np.roll(A,+1,0)+np.roll(A,-1,0)-4*A)/4


def advection(A,vx,vy):
    for i in range(Ny-1):
        for j in range(Nx-1):
            A[int(np.floor(i+vy[i,j])),int((np.floor(j+vx[i,j]))%(Nx-1))] = (1-vx[i,j])*(1-vy[i,j])*A[i,j]
            A[int(np.floor(i+vy[i,j])),int((np.floor(j+vx[i,j]+1))%(Nx-1))] = vx[i,j]*(1-vy[i,j])*A[i,j]
            A[int(np.floor(i+vy[i,j]+1)),int((np.floor(j+vx[i,j]))%(Nx-1))] = (1-vx[i,j])*vy[i,j]*A[i,j]
            A[int(np.floor(i+vy[i,j])+1) ,int((np.floor(j+vx[i,j]+1))%(Nx-1))] = vx[i,j]*vy[i,j]*A[i,j]
    return A

def transition(wv,wl,T):
    for i in range(Ny):
        for j in range(Nx):
            w_sat = 0.2*1e-6*np.exp(-(0.002/(T[i,j]+T0)))
            if w_sat > wl[i,j]+wv[i,j]:
                dwvdt = a*(wv[i,j]-w_sat)
                dwldt = -a*(wv[i,j]-w_sat)
                dTdt = -Q*(dwvdt-dwldt)
                print("do",w_sat) if i==10 and j == 10 else None
            else:
                dwvdt = a*(wv[i,j]-(wl[i,j]+wv[i,j]))
                dwldt = -a*(wv[i,j]-(wl[i,j]+wv[i,j]))
                dTdt = -Q*(dwvdt-dwldt)
    #print(np.amin(w_sat))
    return [dwvdt,dwldt,dTdt]

def impres(vx,vy):
    imx = (np.roll(vx,-1,1)+np.roll(vx,1,1)-2*vx)/2+(np.roll(np.roll(vy,-1,0),-1,1)+np.roll(np.roll(vy,1,0),1,1)-np.roll(np.roll(vy,1,0),-1,1)-np.roll(np.roll(vy,-1,0),1,1))/4
    imy = (np.roll(vy,-1,0)+np.roll(vy,1,0)-2*vy)/2+(np.roll(np.roll(vx,-1,0),-1,1)+np.roll(np.roll(vx,1,0),1,1)-np.roll(np.roll(vx,-1,0),+1,1)-np.roll(np.roll(vx,+1,0),-1,1))/4
    imx[:,Nx-1] = vx[:,0]
    imy[:,Nx-1] = vy[:,0]
    imx[0,:] = 0
    imx[Ny-1,:] = 0
    imy[0,:] = 0
    imy[Ny-1,:] = 0
    return [imx,imy]

cloud = cal()

for time in range(maxIter):
    print(time,np.amax(cloud.vy),np.amin(cloud.vy),np.amin(cloud.T))
    cloud.boundary()
    #print(np.amax(cloud.T))
    cloud.t_diff_expansion()
    cloud.no()
    cloud.buoyancy_dragging()
    cloud.viscosity_impres()
    cloud.diff()
    cloud.no()
    cloud.transition()
    cloud.no()
    cloud.adv()
    cloud.no()
    cloud.boundary()

