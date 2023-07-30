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

tspan = 3600*24*1.0

dt = 10.0



# '.\Desktop\Orbital Mechanics\OMwithPython\SatData\ISS.txt' for ISS


if __name__ == '__main__':

    perts = null_perts()
    perts['J2'] = True

    # ISS
    op0 = OP(t.tle2coes('.\SatData\ISS.txt'),tspan,dt,coes=True,degree=False,perts=perts)

    op0.plot_3d(show_body=False,show_plot=True)

    # t.plot_n_orbits([op0.rs],labels=[op0.name],colors=['b','r','g'],show_plot=True,show_body=False)