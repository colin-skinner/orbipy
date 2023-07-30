#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

import PlanetaryData as pd
import OrbitTools as t

def null_perts():
    return {
        '32':False,
        'aero':False,
        'moon_grav':False,
        'solar_grav':False,
    }

class OrbitPropogator:
    def __init__(self,state0,tspan,dt,coes=False,degree=True,cb=pd.earth,propogate=True,perts=null_perts()):
        if coes:
            self.r0, self.v0,self.name = t.coes2rv(state0,deg=degree,mu=cb['mu'])
        else:
            self.r0 = state0[:3]
            self.v0 = state0[3:6]
            self.name = state0[6]
        self.tspan = tspan
        self.dt = dt 
        self.cb = cb

        self.n_steps = int(np.ceil(self.tspan/self.dt))

        self.ys = np.zeros((self.n_steps,6)) # 3D position, 3D velocity
        self.ts = np.zeros((self.n_steps,1)) # time array

        # initial conditions
        self.y0 = self.r0.tolist() + self.v0.tolist()
        self.ys[0] = np.array(self.y0)
        self.step = 1

        # initiate solver
        self.solver = ode(self.diffy_q)
        self.solver.set_integrator('lsoda')
        self.solver.set_initial_value(self.y0,0)

        # define perturbations dictionary
        self.perts = perts

        if propogate:
            self.propogate_orbit()
    



    def propogate_orbit(self):

        while self.solver.successful() and self.step<self.n_steps:
            self.solver.integrate(self.solver.t+self.dt)
            self.ts[self.step]=self.solver.t
            self.ys[self.step]=self.solver.y
            self.step+=1

        self.rs=self.ys[:,:3]
        self.vs=self.ys[:,3:]




    def diffy_q(self,t,y):
        # unpack components from the "state" y
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx,ry,rz])

        # norm of radius vector
        norm_r = np.linalg.norm(r)

        # two-body acceleration

        a = -r*self.cb['mu']/(norm_r**3) # this is a 3D vector

        if (self.perts['J2']):
            z2 = r[2]**2
            r2 = norm_r**2
            tx = r[0]/norm_r * (5*z2/r2-1)
            ty = r[1]/norm_r * (5*z2/r2-1)
            tz = r[2]/norm_r * (5*z2/r2-3)

            a_j2 = 1.5*self.cb['J2']*self.cb['mu']*self.cb['radius']**2/norm_r**4*np.array([tx,ty,tz])

            a += a_j2

        

        return [vx,vy,vz,a[0],a[1],a[2]]

    


    def plot_3d(self,show_plot=False,save_plot=False,title="Test Title",set_pad=10,show_body=True):

        # 3d plot
        fig = plt.figure(figsize=(18,8))
        # fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')

        

        # plot Central Body
        # Mesh grid of polar coordinates _u=theta _v=phi
        _u,_v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        if show_body:
            # Determines 3D coordinates
            _x = self.cb['radius']*np.cos(_u)*np.sin(_v)
            _y = self.cb['radius']*np.sin(_u)*np.sin(_v)
            _z = self.cb['radius']*np.cos(_v)

            ax.plot_surface(_x, _y, _z, cmap = "Blues")

        # plot trajectory and starting point
        ax.plot(self.rs[:,0], self.rs[:,1], self.rs[:,2], 'w', label = 'Trajectory')
        ax.plot([self.rs[0,0]], [self.rs[0,1]], [self.rs[0,2]], 'wo', label = 'Starting Position')

        # Coord System Origin
        l = self.cb['radius']*2.0
        x,y,z = [[0,0,0], [0,0,0], [0,0,0]]
        u,v,w = [[l,0,0], [0,l,0], [0,0,l]]
        ax.quiver(x, y, z, u, v, w, color = 'k')

        # Check for custom axes limits
        max_val = np.max(np.abs(self.rs))

        

        # Set labels and title
        ax.set_xlim([-max_val,max_val])
        ax.set_ylim([-max_val,max_val])
        ax.set_zlim([-max_val,max_val])

        ax.xaxis.set_major_formatter('{x:1.2e}')
        ax.yaxis.set_major_formatter('{x:1.2e}')
        ax.zaxis.set_major_formatter('{x:1.2e}')
        # plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
        # plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
        # plt.ticklabel_format(style='plain', axis='z', scilimits=(0,0))

        ax.xaxis.labelpad=set_pad
        ax.yaxis.labelpad=set_pad
        ax.zaxis.labelpad=set_pad
        

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_aspect('equal')
        ax.set_title(title)
        plt.legend()

        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(title+'.png',dpi=300) # 300 dots per inch







