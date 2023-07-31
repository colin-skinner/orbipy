#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

import PlanetaryData as pd
import OrbitTools as ot
import AtmosphericTools as at

def null_perts():
    return {
        'oblateness':False,
        'aero':False,
        'moon_grav':False,
        'solar_grav':False,
    }

class OrbitPropogator:
    def __init__(self,state0,tspan,dt,mass0=10.0,coes=False,deg=True,cb=pd.earth,propogate=True,perts=null_perts()):
        if coes:
            self.r0, self.v0 = ot.coes2rv(state0,deg=deg,mu=cb['mu'])
        else:
            self.r0 = state0[:3]
            self.v0 = state0[3:6]
        
        self.y0 = self.r0.tolist() + self.v0.tolist()
        self.tspan = tspan
        self.dt = dt 
        self.cb = cb
        self.mass = mass0


        if type(tspan)==str:
            self.tspan = float(tspan)*ot.rv2period(self.r0,self.v0,self.cb)

        self.n_steps = int(np.ceil(self.tspan/self.dt))

        # initial conditions
        self.ts = np.zeros((self.n_steps,1)) # time array
        self.ys = np.zeros((self.n_steps,6)) # 3D position, 3D velocity
        self.ts[0] = 0
        self.ys[0,:] = self.y0
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
        self.alts=(np.linalg.norm(self.rs,axis=1) - self.cb['radius']).reshape((self.n_steps,1))
        # print(len(self.alts))
        # self.alts

    def diffy_q(self,t,y):
        # unpack components from the "state" y
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx,ry,rz])
        v = np.array([vx,vy,vz])

        # norm of radius vector
        norm_r = np.linalg.norm(r)

        # two-body acceleration

        a = -r*self.cb['mu']/norm_r**3 # this is a 3D vector

        # For J2 Perturbation
        if (self.perts['oblateness']):
            z2 = r[2]**2
            r2 = norm_r**2
            tx = r[0]/norm_r * (5*z2/r2-1)
            ty = r[1]/norm_r * (5*z2/r2-1)
            tz = r[2]/norm_r * (5*z2/r2-3)
            
            a += 1.5*self.cb['J2']*self.cb['mu']*self.cb['radius']**2/norm_r**4*np.array([tx,ty,tz])

        # For aerodynamic drag
        if self.perts['aero']:
            # calculate altitude and air density
            z = norm_r - self.cb['radius']
            rho = at.calc_atmospheric_density(z)

            # calculate motion of s/c with respect to a rotating atmosphere
            v_rel = v - np.cross(self.cb['atm_rot_vector'],r)

            drag = -v_rel*0.5*rho*ot.norm(v_rel)*self.perts['Cd']*self.perts['A']/self.mass
            
            a+=drag

            
            

        

        return [v[0],v[1],v[2],a[0],a[1],a[2]]
  
    def calculate_coes(self,degree=True,print_results=False):
        print('Calculating COEs...')

        self.coes = np.zeros((self.n_steps,6))

        for n in range(self.n_steps):
            self.coes[n,:] = ot.rv2coes(self.rs[n,:], self.vs[n,:], mu=self.cb['mu'],deg=degree,print_results=print_results)

    def plot_coes(self, hours=False, days=False, show_plot=False, save_plot=False, title='COEs',figsize=(20,10),dpi=500):
        print("Plotting COEs...")

        # Create figure and axes instances
        fig,axs = plt.subplots(nrows=2,ncols=3,figsize=figsize)

        # Figure Title
        fig.suptitle(title,fontsize=20)
        
        # X axis
        if hours:
            ts = self.ts / 3600.0
            xlabel = "Time Elapsed (hours)"
        elif days:
            ts = self.ts / 3600.0 / 24.0
            xlabel = "Time Elapsed (days)"
        else:
            ts = self.ts
            xlabel = "Time Elapsed (s)"
        
        # [a,e_norm,i*R2D,ta*R2D,aop*R2D,raan*R2D]

        # Plot true anomaly
        axs[0,0].plot(ts, self.coes[:,3])
        axs[0,0].set_title("True Anomaly vs. Time")
        axs[0,0].grid(True)
        axs[0,0].set_ylabel("Angle (\u00B0)")
        axs[0,0].set_xlabel(xlabel)

        # Plot semi major axis
        axs[1,0].plot(ts, self.coes[:,0])
        axs[1,0].set_title("Semi-Major Axis vs. Time")
        axs[1,0].grid(True)
        axs[1,0].set_ylabel("Semi-Major Axis (km)")
        axs[1,0].set_xlabel(xlabel)

        # Plot Eccentricity
        axs[0,1].plot(ts, self.coes[:,1])
        axs[0,1].set_title("Eccentricity vs. Time")
        axs[0,1].grid(True)
        axs[0,1].set_ylabel("Eccentricity")
        axs[0,1].set_xlabel(xlabel)

        # Plot Argument of Periapse
        axs[0,2].plot(ts, self.coes[:,4])
        axs[0,2].set_title("Argument of Periapse vs. Time")
        axs[0,2].grid(True)
        axs[0,2].set_ylabel("Argument of Periapse (\u00B0)")
        axs[0,2].set_xlabel(xlabel)

        # Plot Inclination
        axs[1,1].plot(ts, self.coes[:,2])
        axs[1,1].set_title("Inclination vs. Time")
        axs[1,1].grid(True)
        axs[1,1].set_ylabel("Inclination (\u00B0)")
        axs[1,1].set_xlabel(xlabel)

        # Plot RAAN
        axs[1,2].plot(ts, self.coes[:,5])
        axs[1,2].set_title("RAAN vs. Time")
        axs[1,2].grid(True)
        axs[1,2].set_ylabel("RAAN (\u00B0)")
        axs[1,2].set_xlabel(xlabel)

        

        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(title+'.png',dpi=dpi) 

    def calculate_apoapse_periapse(self):
        # define empty arrays
        self.apoapses = self.coes[:,0]*(1+self.coes[:,1])
        self.periapses = self.coes[:,0]*(1-self.coes[:,1])

    def plot_apoapse_periapse(self,  hours=False, days=False, show_plot=False, save_plot=False, title='Apoapse and Periapse',figsize=(20,10),dpi=500):

        plt.figure(figsize=figsize)

        # X axis
        if hours:
            ts = self.ts / 3600.0
            xlabel = "Time Elapsed (hours)"
        elif days:
            ts = self.ts / 3600.0 / 24.0
            xlabel = "Time Elapsed (days)"
        else:
            ts = self.ts
            xlabel = "Time Elapsed (s)"
        
        # Plot each
        plt.plot(ts, self.apoapses, 'b', label='Apoapse')
        plt.plot(ts, self.periapses, 'r', label='Periapse')

        # labels
        plt.ylabel("Altitude (km)")
        plt.xlabel(xlabel)
        
        # other parameters
        plt.grid(True)
        plt.title(title)
        plt.legend()

        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(title+'.png',dpi=dpi) 

    # Plot altitude over time
    def plot_alts(self, hours=False, days=False, show_plot=False, save_plot=False, title='Radial Distance vs. Time',figsize=(20,10),dpi=500):

        plt.figure(figsize=figsize)

        # X axis
        if hours:
            ts = self.ts / 3600.0
            xlabel = "Time Elapsed (hours)"
        elif days:
            ts = self.ts / 3600.0 / 24.0
            xlabel = "Time Elapsed (days)"
        else:
            ts = self.ts
            xlabel = "Time Elapsed (s)"
        
        plt.plot(ts,self.alts,'w')
        plt.grid(True)
        plt.ylabel("Altitude (km)")
        plt.xlabel(xlabel)

        plt.title(title)


        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(title+'.png',dpi=dpi) 


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

            ax.plot_surface(_x, _y, _z, cmap = "Blues",alpha=0.3,zorder=0)

        # plot trajectory and starting point
        ax.plot(self.rs[:,0], self.rs[:,1], self.rs[:,2], 'w', label = 'Trajectory',zorder=10)
        ax.plot([self.rs[0,0]], [self.rs[0,1]], [self.rs[0,2]], 'wo', label = 'Starting Position',zorder=10)

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







