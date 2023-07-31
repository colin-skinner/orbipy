#!/usr/bin/env python
import numpy as np

G_meters = 6.67408e-11

G = G_meters * 10**-9

day2sec = 24*3600.0

sun = {
    'name' : 'Sun',
    'mass' : 1.989e30,
    'mu' : 1.989e30*G,
    'radius' : 695510.0,
    'G1' : 10.0**8

}

# atm[:,0] = altitude (km), atm[:,0] = density (kg/km^3) 
atm = np.array([[63.096,2.059e-4],[251.189,5.909e-11],[1000.0,3.561e-15]])
earth = {
    'name' : 'Earth',
    'mass' : 5.972e24,
    'mu' : 5.972e24*G,
    'radius' : 6378.0,
    'J2': 1.082635854e-3,
    # 'spice_file' : 'ADD EVENTUALLY',
    'zs' : atm[:,0], # km
    'rhos' : atm[:,1]*10**9, # kg/km^3
    'atm_rot_vector' : np.array([0.0,0.0,72.9211e-6]) # rad/s

}