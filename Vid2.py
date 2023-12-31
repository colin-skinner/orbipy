import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

def plot(r):

    # 3d plot
    fig = plt.figure(figsize = [18,6])
    ax = fig.add_subplot(111, projection = '3d')

    # plot trajectory and starting point
    ax.plot(r[:,0],r[:,1],r[:,2],'k')
    ax.plot(r[0,0],r[0,1],r[0,2],'ko')

    r_plot = earth_radius

    # plot Earth

    # Mesh grid of polar coordinates _u=theta _v=phi
    _u,_v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    # Determines 3D coordinates
    _x = r_plot*np.cos(_u)*np.sin(_v)
    _y = r_plot*np.sin(_u)*np.sin(_v)
    _z = r_plot*np.cos(_v)

    ax.plot_surface(_x, _y, _z, cmap = "Blues")

    # Coord System Origin
    l = r_plot*2.0
    x,y,z = [[0,0,0], [0,0,0], [0,0,0]]
    u,v,w = [[l,0,0], [0,l,0], [0,0,l]]
    ax.quiver(x, y, z, u, v, w, color = 'k')

    # Check for custom axes limits
    max_val = np.max(np.abs(r))

    # Set labels and title
    ax.set_xlim([-max_val,max_val])
    ax.set_ylim([-max_val,max_val])
    ax.set_zlim([-max_val,max_val])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_aspect('equal')
    # ax.plot_title('Earth and location')
    plt.legend(['Trajectory','Starting Position'])


    plt.show()

earth_radius = 6378.0 # km
earth_mu = 398600.0 # km^3/s^2

def diffy_q(t,y,mu):
    # unpack components from the "state" y
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx,ry,rz])

    # norm of radius vector
    norm_r = np.linalg.norm(r)

    # two-body acceleration

    ax,ay,az = -r*mu/(norm_r**3) # this is a 3D vector

    return [vx,vy,vz,ax,ay,az]






if __name__ == '__main__':

    # Initial conditions of orbital parameters

    r_mag = earth_radius + 500.0 # km
    v_mag = np.sqrt(earth_mu/r_mag) # km/s

    # initial position and velocity ectors
    r0 = [r_mag,0,0]
    v0 = [0,v_mag,0]

    # timespan
    tspan = 60.0 * 100

    dt = 100.0

    n_steps = int(np.ceil(tspan/dt))

    ys = np.zeros((n_steps,6)) # 3D position, 3D velocity
    ts = np.zeros((n_steps,1)) # time array

    # initial conditions
    y0 = r0 + v0
    ys[0] = np.array(y0)
    step = 1

    # initiate solver
    solver = ode(diffy_q)
    solver.set_integrator('lsoda')
    solver.set_initial_value(y0,0)
    solver.set_f_params(earth_mu)


    while solver.successful() and step<n_steps:
        solver.integrate(solver.t+dt)
        ts[step]=solver.t
        ys[step]=solver.y
        step+=1

    rs=ys[:,:3]

    plot(rs)


