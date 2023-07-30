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
    op0 = OP(t.tle2coes('.\SatData\ISS.txt'),tspan,dt,coes=True,degree=False)

    # Progress
    op1 = OP(t.tle2coes('.\SatData\progress.txt'),tspan,dt,coes=True,degree=False)

    # EUCLID
    op2 = OP(t.tle2coes('.\SatData\EUCLID.txt'),tspan,dt,coes=True,degree=False)

    t.plot_n_orbits([op0.rs, op1.rs, op2.rs],labels=[op0.name, op1.name, op2.name],colors=['b','r','g'],show_plot=True,show_body=False)