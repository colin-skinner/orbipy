import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

from OrbitPropogator import OrbitPropogator as OP
import PlanetaryData as pd
import OrbitTools as t

cb = pd.sun

tspan = 3600*24*6.0

dt = 100.0


if __name__ == '__main__':

    r_mag = pd.earth['radius']+400
    v_mag = sqrt(pd.earth['mu']/r_mag)

    r0 = np.array([r_mag,0,0])
    v0 = np.array([0,v_mag,0])

    r_mag0 = pd.earth['radius']+1000
    v_mag0 = sqrt(pd.earth['mu']/r_mag0)

    r00 = np.array([r_mag0,0,0])
    v00 = np.array([0,0,v_mag0])
    
    op0 = OP(r0,v0,tspan,dt)
    op00 = OP(r00,v00,tspan,dt)

    op0.propogate_orbit()
    op00.propogate_orbit()

    t.plot_n_orbits([op0.rs, op00.rs],labels=[0,1],colors=['g','r'],show_plot=True)




