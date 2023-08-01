#!/usr/bin/env python
import numpy as np

  
G_meters = 6.67408e-11 # m^3/kg/s^2

G = G_meters * 10**-9 # km^3/kg/s^2

day2sec = 24*3600.0 # sec/day

sun = {
    'name' : 'Sun',
    'mass' : 1.989e30, # kg
    'mu' : 1.989e30*G, # km^3/s^2
    'radius' : 695510.0, # km
    # 'G1' : 10.0**8

}

# atm[:,0] = altitude (km), atm[:,0] = density (kg/m^3) 
# atm = np.array([[63.096,2.059e-4],[251.189,5.909e-11],[1000.0,3.561e-15]])
earth_atm = np.array([[0.0,1.225],[100.0,5.25e-7],[150.0,1.73e-9],[200.0,2.41e-10],[250.0,5.97e-11],[300.0,1.87e-11],[350.0,6.66e-12],[400.0,2.62e-12],[500.0,4.76e-13],[700.0,2.36e-14],[850.0,4.22e-15],[1000.0,1.49e-15]])

earth = {
    'name' : 'Earth',
    'mass' : 5.972e24, # kg
    'mu' : 5.972e24*G, # km^3/s^2
    'radius' : 6378.0, # km
    'J2': 1.082635854e-3,
    # 'spice_file' : 'ADD EVENTUALLY',
    'zs' : earth_atm[:,0], # km
    'rhos' : earth_atm[:,1]*10**9, # kg/km^3
    'atm_rot_vector' : np.array([0.0,0.0,72.9211e-6]) # rad/s

}