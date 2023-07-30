import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

from sys import path
# path.append('C:\\Users\\mcmak\\python_tools')
from OrbitPropogator import OrbitPropogator as OP
import PlanetaryData as pd

cb = pd.sun



if __name__ == '__main__':

    # Initial conditions of orbital parameters


    r_mag = cb['radius'] + 14500 # km
    v_mag = np.sqrt(cb['mu']/r_mag) # km/s

    # initial position and velocity ectors
    r0 = [r_mag,r_mag*0.01,r_mag*-0.5]
    v0 = [0,v_mag,v_mag*0.8]

    # timespan
    tspan = 6*3600*6.0

    dt = 100.0

    op = OP(r0,v0,tspan,dt,cb=cb)
    op.propogate_orbit()
    print("uh")
    op.plot_3d(show_plot=True,set_pad=20)

    
    





