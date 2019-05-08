# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:31:41 2019

@author: Ewout van der Velde
"""

import numpy as np
import matplotlib.pyplot as plt

labda = 546.08e-9


class Kromte:
    # Formule voor de kromte straal is a^2/6d + d/2
    Lens = np.array([2.445e-3, 2.440e-3, 2.445e-3, 2.445e-3, 2.445e-3, 2.445e-3])
    Vlak = np.array([2.260e-3, 2.265e-3, 2.265e-3, 2.265e-3, 2.265e-3, 2.265e-3])
    
    a_meet = np.array([28.1e-3, 28.2e-3, 29.1e-3])
    a = np.mean(a_meet)
    a_std = 0.01
    
    d_meet = Lens - Vlak
    d = np.mean(d_meet)
    #d_std = np.std(d_meet, ddof=1)/np.sqrt(len(d_meet))   
    R = (a**2)/(6*d) + d/2

print(Kromte.d)
    
data1 = np.genfromtxt("Metingen1.dat")   
data2 = np.genfromtxt("Metingen2.dat")

def addstraal(data):
    straal = (((data[:,4]+data[:,5]/200) - (data[:,2]+data[:,3]/200))/1000)/2
    straal = np.reshape(straal, [len(straal), 1])
    return np.concatenate((data, straal), axis =1)

data1 = addstraal(data1)
data2 = addstraal(data2)


def cons(c, data):
    mask = (data[:,0] == c)
    return data[mask] 

class meet1:
    data = data1 
    c0 = cons(0, data)
    c25 = cons(25, data)
    c50 = cons(50, data)
    c75 = cons(75, data)
    c100 = cons(100, data)
    c125 = cons(125, data)
    c150 = cons(150, data)
    c175 = cons(175, data)
    c200 = cons(200, data)
    c225 = cons(225, data)
    c250 = cons(250, data)

class meet2:
    data = data2
    c0 = cons(0, data)
    c25 = cons(25, data)
    c50 = cons(50, data)
    c75 = cons(75, data)
    c100 = cons(100, data)
    c125 = cons(125, data)
    c150 = cons(150, data)
    c175 = cons(175, data)
    c200 = cons(200, data)
    c225 = cons(225, data)
    c250 = cons(250, data)

def n_v_r(naam):
    n = (Kromte.R * labda * naam[:,1])/((naam[:,6])**2)
    print(n)
    return n

## Voor bugtesting

plt.figure("test")
plt.title("gemeten Brekingsindex")
plt.plot((n_v_r(meet1.c0)), meet1.c0[:,1], label="C0_1")
plt.plot((n_v_r(meet2.c0)), meet2.c0[:,1], label="C0_2")
plt.plot((n_v_r(meet1.c200)), meet1.c200[:,1], label="C200_1")
plt.plot((n_v_r(meet2.c200)), meet2.c200[:,1], label="C200_2")
plt.xlabel("Brekingsindex")
plt.ylabel("Ring k")
plt.legend()
plt.show("test")

# =============================================================================
# def n_gem(naam1, naam2):
#     return (n_v_r(naam1) + n_v_r(naam2)) / 2 
# 
# plt.figure("gem")
# plt.title("Gemiddelde brekingsindex")
# plt.plot(n_gem(meet1.c0, meet2.c0), meet1.c0[:,1], label=("C0"))
# plt.plot(n_gem(meet1.c200, meet2.c200), meet1.c200[:,1], label=("C200"))
# plt.xlabel("Brekingsindex")
# plt.ylabel("Ring k")
# plt.legend()
# plt.show("gem")
# =============================================================================

plt.figure("Linefitding")
plt.title("Voor linefit")
plt.ylabel("r^2")
plt.xlabel("k")
plt.plot(meet1.c200[:,1],(meet1.c200[:,6])**2)
plt.show("Linefitding")

def gem_std(naam):
    gem = np.mean(n_v_r(naam))
    std = np.std(naam, ddof=1)/np.sqrt(len(naam))
    return np.array([gem, std])

def linefit(naam):
    slope, intercept = np.polyfit(naam[:,1], (naam[:,6])**2, 1)
    return slope

print((linefit(meet1.c0))/(Kromte.R*labda))
print((linefit(meet2.c0))/(Kromte.R*labda))
print((linefit(meet1.c200))/(Kromte.R*labda))
print((linefit(meet2.c200))/(Kromte.R*labda))
