import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from OrbitPropogator import OrbitPropogator as OP
import PlanetaryData as pd
import OrbitTools as t
from OrbitPropogator import null_perts

cb = pd.earth

tspan = 3600*24*100.0

dt = 5000

date0 = '2020-02-23'

if __name__ == '__main__':

    iss_coes = t.tle2coes('SATdata/ISS.txt')

    state0 = [42164.0, 0.001, 0.0, 0.0, 0.0, 0.0]

    perts = null_perts()
    # Add lunar gravity perturbation
    perts['n_bodies'] = [pd.luna]

    # create orbit propogator instance for GEO and ISS
    op0 = OP(state0, tspan, dt, coes=True, deg=True, perts=perts, date0=date0, propagator='dopri5')
    op_ISS = OP(iss_coes, tspan, 1000, coes=True, deg=True, perts=perts, date0=date0, propagator='dopri5')

    op_ISS.calculate_coes(parallel=False)
    op_ISS.plot_coes(days=True, show_plot=True)

    op0.calculate_coes(parallel=False)
    op0.plot_coes(days=True, show_plot=True)

    t.plot_n_orbits([op_ISS.rs, op0.rs, op0.perts['n_bodies'][0]['states'][:,:3]],labels=['ISS','GEO','Moon'],show_plot=True)





