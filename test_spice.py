#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice

plt.style.use('dark_background')

import PlanetaryData as pd
import OrbitTools as ot
import spice_tools as st
from OrbitPropogator import OrbitPropogator as OP
from OrbitPropogator import null_perts

# central body
cb = pd.earth

# total steps of ephemeris data
STEPS = 100000

# Reference frame of data
FRAME = 'ECLIPJ2000'

# observer of planetary bodies
OBSERVER = 'SUN'

if __name__ == '__main__':

    # Load metakernal
    spice.furnsh('.\spice_data\solar_system_kernal.mk')
    ids, names, tcs_sec, tcs_cal = st.get_objects('C:\\Users\\mcmak\\Desktop\\Orbital Mechanics\\OMwithPython\\orbipy\\spice_data\\de432s.bsp',display = True)

    # exit()

    # only include barycenters
    names = [f for f in names if 'BARYCENTER' in f]

    # create time array for ephemeris
    times = st.tc2array(tcs_sec[0],STEPS)

    # create empty list
    rs = []

    for name in names:

        rs.append(st.get_ephemeris_data(name,times,FRAME,OBSERVER))
    # print(len(rs))
    ot.plot_n_orbits(rs,names,show_plot=True,AU=True,cb=pd.sun,figsize=(20,10),show_body=True)
        


