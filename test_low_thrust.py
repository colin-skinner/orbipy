import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

from OrbitPropogator import OrbitPropogator as OP
from OrbitPropogator import null_perts
import PlanetaryData as pd
import OrbitTools as ot

cb = pd.earth

tspan = 3600*46

dt = 100.0



# '.\Desktop\Orbital Mechanics\OMwithPython\SatData\ISS.txt' for ISS


if __name__ == '__main__':

    perts = null_perts()
    perts['thrust'] = 0.327 # N
    perts['isp'] = 4300 # s
    perts['thrust_direction'] = -1
    
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
    state0 = [cb['radius']+800,e,i,ta,aop,raan]
    # state0 = ot.tle2coes('.\SatData\ISS.txt')
    # print(len(cb['rhos']))
    # for i in cb['rhos']:
    #     print(i)
    op = OP(state0, tspan, dt, deg=True, coes = True, mass0 = mass0, perts = perts)
    # print(op.ys[0])
    op.calculate_Esp()
    op.plot_Esp(show_plot=True,hours=True)
    # print(op.cb['mu']/2/a)
    op.plot_3d(show_plot=True)
    op.calculate_coes(print_results=False)
    op.plot_coes(show_plot=True,hours=True)
    op.calculate_apoapse_periapse()
    op.plot_apoapse_periapse(show_plot=True,hours=True)
    op.plot_alts(show_plot=True)

    # plt.hist(op.periapses)
    