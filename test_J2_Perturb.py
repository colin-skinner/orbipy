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

tspan = 3600*24*2.0

dt = 100.0



# '.\Desktop\Orbital Mechanics\OMwithPython\SatData\ISS.txt' for ISS


if __name__ == '__main__':

    perts = null_perts()
    perts['J2'] = True
    

    
    r0 = np.array([-2384.46, 5729.01, 3050.46])
    v0 = np.array([-7.36138, -2.98997, 1.64354])
    # state0 = np.array(t.rv2coes(r0,v0,print_results=True,deg=True))
    # state0 = np.array([cb['radius']+600,0.1,63.435,0.0,0.0,0.0])
    # state0 = np.array([cb['radius']+600,0.1,90.01,0.0,0.0,0.0])
    state0 = t.tle2coes('.\SatData\ISS.txt',deg=True)




    test1 = OP(state0,tspan,dt,coes=True,perts=perts)
    test1.plot_3d(show_plot=True)
    test1.calculate_coes()
    test1.plot_coes(show_plot=True,hours=True)
    # state0 = t.tle2coes('.\SatData\ISS.txt')
    
    # ISS
    # issTest = OP(state0,tspan,dt,coes=True,degree=True,perts=perts)
    # issTest.plot_3d(show_plot=True,show_body=False)
    # issTest.calculate_coes()
    # op.plot_coes(show_plot=False,hours=True)
    # op0.plot_3d(show_body=False,show_plot=True)
    # op0.calculate_coes()

    # op0.plot_coes(show_plot=True,hours=True)

    # t.plot_n_orbits([op0.rs],labels=[op0.name],colors=['b','r','g'],show_plot=True,show_body=False)