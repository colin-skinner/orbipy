import numpy as np
import matplotlib.pyplot as plt
import PlanetaryData as pd  
import math as m
import datetime

D2R = np.pi/180
R2D = 180/np.pi

def norm(v):
    return np.linalg.norm(v)

# Calculate atmospheric density from given altitude
def calc_atmospheric_density(z):
    rhos,zs = find_rho_z(z)
    if rhos[0] == 0:
        return 0.0
    # print(rhos[0])
    Hi = -(zs[1]-zs[0])/m.log(rhos[1]/rhos[0])
    
    return rhos[0]*m.exp(-(z-zs[0])/Hi)

# Find endpoints of altitude and density surrounding input altitude
def find_rho_z(z,zs=pd.earth['zs'],rhos=pd.earth['rhos']):
    
    # z is not in atmosphere anymore
    if not 1.0<z<1000.0:
        return [[0.0,0.0],[0.0,0.0]]
    
    # find the two points surrounding given input altitude
    for n in range(len(rhos)-1):
        if zs[n]<z<zs[n+1]:
            return [[rhos[n],rhos[n+1]],[zs[n],zs[n+1]]]
    
    # if out of range return zeros
    return [[0.0,0.0],[0.0,0.0]]
