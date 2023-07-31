import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

from OrbitPropogator import OrbitPropogator as OP
from OrbitPropogator import null_perts
import PlanetaryData as pd
import OrbitTools as t

cb = pd.earth

tspan = 3600*24*3

dt = 100.0



# '.\Desktop\Orbital Mechanics\OMwithPython\SatData\ISS.txt' for ISS


if __name__ == '__main__':

    perts = null_perts()
    perts['oblateness'] = True
    perts['aero'] = True
    perts['Cd'] = 2.2
    perts['A'] = (1e-3)**2/4.0 # km^2\

    mass0 = 10.0 # kg

    # apogee and perigee
    rp = 215 + cb['radius']
    ra = 300 + cb['radius']

    # COEs
    raan = 340.0
    i = 65.2
    aop = 58.0
    ta = 332.0

    # Other orbital elements
    a = (rp+ra)/2.0
    e = (ra-rp)/(ra+rp)

    # Initial state vector
    state0 = [a,e,i,ta,aop,raan]

    op = OP(state0, tspan, dt, deg=True, coes = True, mass0 = mass0, perts = perts)
    op.plot_alts(show_plot=True,hours=True)
    op.plot_3d(show_plot=True)
    op.calculate_coes()
    op.plot_coes(show_plot=True,hours=True)
    op.calculate_apoapse_periapse()
    op.plot_apoapse_periapse(show_plot=True,hours=True)

    
    