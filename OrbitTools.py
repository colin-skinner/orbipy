#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import PlanetaryData as pd  
import math as m
import datetime

D2R = np.pi/180
R2D = 180/np.pi

def norm(v):
    return np.linalg.norm(v)

def plot_n_orbits(rs,labels,colors=['w','w','w','w'],cb=pd.earth,show_plot=False,save_plot=False,title="Many Orbits",set_pad=10,show_body=True):

    # 3d plot
    fig = plt.figure(figsize=(18,8))
    # fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')

    

    # plot Central Body
    # Mesh grid of polar coordinates _u=theta _v=phi
    _u,_v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    if show_body:
        # Determines 3D coordinates
        _x = cb['radius']*np.cos(_u)*np.sin(_v)
        _y = cb['radius']*np.sin(_u)*np.sin(_v)
        _z = cb['radius']*np.cos(_v)

        ax.plot_surface(_x, _y, _z, cmap = "Blues")


    # Defines labels
    # print(len(rs))
    # if len(rs) == len(labels):
    #     lab = [i for i in labels]
    # else:
    #     lab = [""]*len(rs)
    #     print(len(lab))
    #     for i in range(len(rs)):
    #         lab[i] = rs[i].name
        


    # plot trajectory and starting point
    n=0
    for r in rs:
        
        ax.plot(r[:,0], r[:,1], r[:,2], colors[n], label = labels[n])
        ax.plot([r[0,0]], [r[0,1]], [r[0,2]], 'wo')
        n+=1

    # Coord System Origin
    l = cb['radius']*2.0
    x,y,z = [[0,0,0], [0,0,0], [0,0,0]]
    u,v,w = [[l,0,0], [0,l,0], [0,0,l]]
    ax.quiver(x, y, z, u, v, w, color = 'k')

    # Check for custom axes limits
    max_val = np.max(np.abs(rs))

    

    # Set labels and title
    ax.set_xlim([-max_val,max_val])
    ax.set_ylim([-max_val,max_val])
    ax.set_zlim([-max_val,max_val])

    ax.xaxis.set_major_formatter('{x:1.2e}')
    ax.yaxis.set_major_formatter('{x:1.2e}')
    ax.zaxis.set_major_formatter('{x:1.2e}')
    # plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    # plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    # plt.ticklabel_format(style='plain', axis='z', scilimits=(0,0))

    ax.xaxis.labelpad=set_pad
    ax.yaxis.labelpad=set_pad
    ax.zaxis.labelpad=set_pad
    

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.legend()

    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(title+'.png',dpi=300) # 300 dots per inch


# Converts classical orbital elements to r and v vectors
def coes2rv(coes, deg=False, mu=pd.earth['mu']):
    
    a,e,i,ta,aop,raan,date,name=coes
    if deg:
        i*=D2R
        ta*=D2R
        aop*=D2R
        raan*=D2R

    E = ecc_anomaly([ta,e],'tae')

    r_norm = a*(1-e**2)/(1+e*np.cos(ta))

    # Calc r and v vectorals in perifocal form
    r_perif = r_norm*np.array([m.cos(ta),m.sin(ta),0])
    v_perif = m.sqrt(mu*a)/r_norm*np.array([-m.sin(E),m.cos(E)*m.sqrt(1-e**2),0])

    # rotation matrix from perifocal to ECI
    perif2eci = np.transpose(eci2perif(raan,aop,i))

    # Calc r and v vectors in inertial frames
    r = np.dot(perif2eci, r_perif)
    v = np.dot(perif2eci, v_perif)

    return r,v,name

# Converts r and v vectors to classical orbital elements
def rv2coes(r,v,mu=pd.earth['mu'],deg=False,print_results=False):

    r_norm = norm(r)

    # Specific angular momentum
    h = np.cross(r,v)
    h_norm = norm(h)

    i = m.acos(h[2]/h_norm) # inclination
    e = ((norm(v)**2-mu/r_norm)*r - np.dot(r,v))/mu # Eccentricity vector
    e_norm = norm(e)

    N = np.cross([0,0,1],h) # Node line
    N_norm = norm(N)

    raan = m.acos(N[0]/N_norm) # Right ascension of ascending node
    if N[1]<0: # To make it positive
        raan = 2*np.pi - raan

    aop = m.acos(np.dot(N,e)/N_norm/e_norm) # Argument of Perigee
    if e[2]<0:
        aop = 2*np.pi - aop

    ta = m.acos(np.dot(e,r)/e_norm/r_norm) # True Anomaly
    if np.dot(r,v)<0:
        ta = 2*np.pi - ta
    
    a = r_norm * (1+e_norm*m.cos(ta))/(1-e_norm**2) # Semi-major Axis

    if print_results:
        print('a',a)
        print('e',e_norm)
        print('i',i*R2D)
        print('RAAN',raan*R2D)
        print('AOP',aop*R2D)
        print('TA',ta*R2D)

    if deg:
        return [a,e_norm,i*R2D,ta*R2D,aop*R2D,raan*R2D]
    else:
        return [a,e_norm,i,ta,aop,raan]


# inertial to perifocal rotation matrix
def eci2perif(raan,aop,i): 
    row0 = [-m.sin(raan)*m.cos(i)*m.sin(aop) + m.cos(raan)*m.cos(aop), m.cos(raan)*m.cos(i)*m.sin(aop) + m.sin(raan)*m.cos(aop), m.sin(i)*m.sin(aop)]
    row1 = [-m.sin(raan)*m.cos(i)*m.cos(aop) - m.cos(raan)*m.sin(aop), m.cos(raan)*m.cos(i)*m.cos(aop) - m.sin(raan)*m.sin(aop), m.sin(i)*m.cos(aop)]
    row2 = [m.sin(raan)*m.sin(i), -m.cos(raan)*m.sin(i), m.cos(i)]
    return np.array([row0, row1, row2])


# Calc eccentric anomaly (E)    
def ecc_anomaly(arr, method, tol=1e-8):
    if method=='newton':
        # Newton's method for iteratively finding E
        Me,e = arr
        if Me < np.pi/2.0:
            E0 = Me + e/2.0
        else:
            E0 = Me - e
        
        for n in range(200): # arbitrary 
            ratio = (E0-e*np.sin(E0)-Me)/(1-e*np.cos(E0))
            if abs(ratio)<tol:
                if n==0:
                    return E0
                else:
                    return E1
            else:
                E1 = E0 - ratio
                E0 = E1
        # Did not converge
        return False
    elif method=='tae':
        ta,e = arr
        return 2*m.atan(m.sqrt((1-e)/(1+e))*m.tan(ta/2.0))
    else:
        print("Invalid method for eccentric anomaly")




# Takes text file containing TLE and returns classical orbital elements and a few other parameters
def tle2coes(tle_filename, mu=pd.earth['mu']):
    # read the file
    with open(tle_filename,'r') as f:
        lines = f.readlines()

    name = lines[0].strip() # name of satellite
    line1 = lines[1].strip().split() # Made into list
    line2 = lines[2].strip().split()

    # Time and day
    epoch = line1[3]
    year, month, day, hour = calc_epoch(epoch)

    # Collect coes

    i = float(line2[2])*D2R # Inclination (rad)
    raan = float(line2[3])*D2R # Right ascension of ascending node (rad)

    e_string = line2[4]
    e = float('0.'+e_string) # Eccentricity
    aop = float(line2[5])*D2R # Argument of Perigee (rad)
    Me = float(line2[6])*D2R # Mean anomaly (rad)'
    mean_motion = float(line2[7]) # Mean motion (rev/day)
    T = 1/mean_motion*24*3600 # Orbital period (s)
    a = (T**2*mu/4.0/np.pi**2)**(1/3.0) # Semi-major Axis 

    E = ecc_anomaly([Me,e],'newton') # Eccentric Anomaly
    ta = true_anomaly([E,e]) # True Anomaly


    return a, e, i, ta, aop,raan, [year,month,day,hour], name

# Calculates year, month, and date based off the epoch number in 2 Line Elements
def calc_epoch(epoch):

    # year
    year = int('20'+epoch[:2])

    epoch = epoch[2:].split('.')

    # day of year
    day_of_year = int(epoch[0])-1

    # decimal hour of day
    hour = float('0.'+epoch[1])*24.0

    # YYY-MM-DD
    date = datetime.date(year,1,1) + datetime.timedelta(day_of_year)

    month = float(date.month)
    day = float(date.day)

    return year, month, day, hour

def true_anomaly(arr):
    E,e = arr
    return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2.0))

def tle2rv(tle_filename):
    return coes2rv(tle2coes(tle_filename))