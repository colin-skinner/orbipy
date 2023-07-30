import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

from OrbitPropogator import OrbitPropogator as OP
import PlanetaryData as pd
import OrbitTools as t

cb = pd.earth

tspan = 3600*24*1.0

dt = 10.0

# '.\Desktop\Orbital Mechanics\OMwithPython\SatData\ISS.txt' for ISS


if __name__ == '__main__':

    # ISS
    c0 = [cb['radius']+414.0, 0.0006189, 51.6393,0.0, 234.1955, 105.6372]

    # GEO
    c1 = [cb['radius']+35800.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#    Random
    c2 = [cb['radius']+3000,0.3,20.0,0.0,15.0,40.0]

    op0 = OP(c0, tspan, dt, coes=True)
    op1 = OP(c1, tspan, dt, coes=True)
    op2 = OP(c2, tspan, dt, coes=True)
    
    op0.propogate_orbit()
    op1.propogate_orbit()
    op2.propogate_orbit()

    t.plot_n_orbits([op0.rs, op1.rs, op2.rs], labels=['ISS','GEO','Random'], show_plot=True, colors=['b','purple','g'])




