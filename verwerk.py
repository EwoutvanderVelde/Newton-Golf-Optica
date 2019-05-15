# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:31:41 2019

@author: Ewout van der Velde
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sympy import *

labda = 522.08e-9
sig_labda = 10e-9

class Kromte:
    # Formule voor de kromte straal is a^2/6d + d/2
    Lens = np.array([2.445e-3, 2.440e-3, 2.445e-3, 2.445e-3, 2.445e-3, 2.445e-3])
    Vlak = np.array([2.260e-3, 2.265e-3, 2.265e-3, 2.265e-3, 2.265e-3, 2.265e-3])
    
    a_meet = np.array([28.1e-3, 28.2e-3, 29.1e-3])
    a = np.mean(a_meet)
    a_std = 0.0001
    
    d_meet = Lens - Vlak
    d = np.mean(d_meet)
    d_std = np.std(d_meet, ddof=1)/np.sqrt(len(d_meet))   
    R = (a**2)/(6*d) + d/2
    
    
    
data1 = np.genfromtxt("Metingen2.dat")   
#data2 = np.genfromtxt("Metingen2.dat")

def addstraal(data):
    grote_lat = data[:,4]
    kleine_lat = data[:,2]
    grote_wiel = data[:,5]
    kleine_wiel = data[:,3]
    straal = (((grote_lat+grote_wiel/200) - (kleine_lat+kleine_wiel/200))/1000)/2
    straal = np.reshape(straal, [len(straal), 1])
    return np.concatenate((data, straal), axis =1)

data1 = addstraal(data1)
#data2 = addstraal(data2)


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

print(np.shape(meet1.c200))

plt.figure("Meting")
plt.title("Meetresultaten")
plt.ylabel("$r^2$ (mm$^2$)")
plt.xlabel("$k$")
plt.scatter(meet1.c200[:,1],((meet1.c200[:,6])**2)*1000000, label="C200")
plt.scatter(meet1.c150[:,1],((meet1.c150[:,6])**2)*1000000, label="C150")
plt.scatter(meet1.c100[:,1],((meet1.c100[:,6])**2)*1000000, label="C100")
plt.scatter(meet1.c50[:,1],((meet1.c50[:,6])**2)*1000000, label="C50")
plt.scatter(meet1.c0[:,1],((meet1.c0[:,6])**2)*1000000, label="C0")
plt.legend()
plt.show("Meting")


def linefit(naam):
    slope, intercept = np.polyfit(naam[:,1], (naam[:,6])**2, 1)
    return slope

def breking(naam):
    return 1/((linefit(naam))/(Kromte.R*labda))

# =============================================================================
# print("brekingsindex C0 = ",breking(meet1.c0))
# print("brekingsindex C50 = ",breking(meet1.c50))
# print("brekingsindex C100 = ",breking(meet1.c100))
# print("brekingsindex C150 = ",breking(meet1.c150))
# print("brekingsindex C200 = ",breking(meet1.c200))
# =============================================================================

k_linefit = np.linspace(0, 25, num=26)

plt.figure("Linefit")
plt.title("Linefit")
plt.ylabel("$r^2$ (mm$^2$)")
plt.xlabel("$k$")
plt.plot(k_linefit, (k_linefit*linefit(meet1.c200))*1000000, label="C200")
plt.plot(k_linefit, (k_linefit*linefit(meet1.c150))*1000000, label="C150")
plt.plot(k_linefit, (k_linefit*linefit(meet1.c100))*1000000, label="C100")
plt.plot(k_linefit, (k_linefit*linefit(meet1.c50))*1000000, label="C50")
plt.plot(k_linefit, (k_linefit*linefit(meet1.c0))*1000000, label="C0")
plt.legend()
plt.show("Meting")


# =============================================================================
# (m, b) = np.polyfit(meet1.c200[:,1], meet1.c200[:,6]**2, 1)
# yp = np.polyval([m, b], meet1.c200[:,1])
# m, b, r_value, p_value, std_err = stats.linregress(meet1.c200[:,1], yp)
# 
# print("helling volgens polyfit: {:.3e}".format(linefit(meet1.c200)))
# print("helling volgens linregress: {:.3e}".format(m))
# print("std volgens linregress: {:.3e}".format(std_err))
# =============================================================================

def linregresplot(naam, cons):
    x = np.reshape(a = (naam[:,1]), 
                   newshape = [len(naam[:,1]), 1])
    y = np.reshape(a = (naam[:,6]**2), 
                   newshape = [len(naam[:,6]), 1])
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    y_predicted = regression_model.predict(x)
    rmse = mean_squared_error(y, y_predicted)
    r2 = r2_score(y, y_predicted)
    #print("\n{}".format(cons))
    #print("helling volgens sklearn: {}".format(regression_model.coef_))
    #print("rmse volgens sklearn: {}".format(rmse))
    #print("r2 volgens sklearn: {}".format(r2))
    #print("De std van de helling is: {}".format(np.sqrt(rmse)))
    plt.figure()
    plt.errorbar(x = k_linefit, 
                 y = (k_linefit*linefit(naam))*1000000, 
                 yerr = (np.sqrt(rmse))*1000000, 
                 label = "{}".format(cons))
    plt.scatter(x = naam[:,1],
             y = (naam[:,6]**2)*1000000)
    plt.show()
    return 



# =============================================================================
# plt.figure("Errorbars")
# linregresplot(meet1.c0, "C0")
# linregresplot(meet1.c50, "C50")
# linregresplot(meet1.c100, "C100")
# linregresplot(meet1.c150, "C150")
# linregresplot(meet1.c200, "C200")
# plt.legend()
# plt.show("Errorbars")
# =============================================================================

def lingresstd(naam):
    x = np.reshape(a = (naam[:,1]), 
                   newshape = [len(naam[:,1]), 1])
    y = np.reshape(a = (naam[:,6]**2), 
                   newshape = [len(naam[:,6]), 1])
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    y_predicted = regression_model.predict(x)
    rmse = mean_squared_error(y, y_predicted)
    return np.sqrt(rmse)    

def n(naam, cons):
    Hel = Symbol("Hel")
    a = Symbol("a")
    d = Symbol("d")
    lab = Symbol("lab")
    
    n = 1/((Hel)/(((a**2)/(6*d) + d/2)*lab))
    
    Hel_best = linefit(naam)
    sig_Hel = lingresstd(naam)
    
    a_best = Kromte.a
    sig_a = Kromte.a_std
    
    d_best = Kromte.d
    sig_d = Kromte.d_std
    
    lab_best = labda
    sig_lab = sig_labda
    
    n_best = n.subs([(a, a_best), (d, d_best), (Hel, Hel_best), (lab, lab_best)])
    
    n_sig_Hel = sig_Hel*diff(n, Hel).subs([(a, a_best), (d, d_best), (Hel, Hel_best), (lab, lab_best)])
    n_sig_lab = sig_lab*diff(n, lab).subs([(a, a_best), (d, d_best), (Hel, Hel_best), (lab, lab_best)])
    n_sig_d = sig_d*diff(n, d).subs([(a, a_best), (d, d_best), (Hel, Hel_best), (lab, lab_best)])
    n_sig_a = sig_a*diff(n, a).subs([(a, a_best), (d, d_best), (Hel, Hel_best), (lab, lab_best)])
    print(n_sig_Hel, n_sig_lab, n_sig_d, n_sig_a)
    
    n_sig = sqrt(n_sig_Hel**2 + n_sig_lab**2 + n_sig_d**2 + n_sig_a**2)
    print("{} geeft n = {} ± {} ".format(cons, n_best, n_sig))
    return 

n(meet1.c0, "C0")
n(meet1.c50, "C50")
n(meet1.c100, "C100")
n(meet1.c150, "C150")
n(meet1.c200, "C200")


debug1 = np.array([breking(meet1.c0), breking(meet1.c50), breking(meet1.c100), breking(meet1.c150), breking(meet1.c200)])
debug2 = np.array([0, 50, 100, 150, 200])

plt.figure("Debug")
plt.scatter(debug2, debug1)
plt.show("Debug")


import scipy.odr as odr


def odrding(naam):
    # Fictieve dataset x- en y-waarden, beide met onzekerheid
    
    y = (((naam[:,6]).flatten())**2)
    x = (naam[:,1]).flatten()
    #x = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50])
    
    sig_x = np.array([])
    #y = np.array([0.67, 1.36, 1.85, 2.36, 2.60, 3.00, 3.06, 3.08, 3.06, 2.90, 2.68, 2.41, 1.91, 1.42, 0.76])
    sig_yt = (np.zeros(len(naam[:,6])) + 0.000005)
    sig_y = sig_yt * np.mean(y)
    #print(sig_y)
    # =============================================================================
    #  we maken aan deze data een aanpassing met de functie y = a + b x
    #  het is goed om deze dataset te plotten, en op basis hiervan een schatting te
    #  maken voor startwaarden voor de parameters a en b. voor deze specifieke 
    #  dataset kunnen we als schatting voor a de waarde gemeten bij de asafsnede 
    #  nemen (0.5 dus), en voor b, de helling, ongeveer 1.5
    #  met deze opzet kun je overigens eerst een plot maken met data (activeer dan
    #  ax.show), en op een later moment de beste rechte lijn erbij plotten
    #  nb dit is een niet opgemaakte plot
    # =============================================================================
    f,ax = plt.subplots(1)
    ax.errorbar(x,y,yerr=sig_y,fmt='k.')
    #ax.show()
    A_start= 0.0000001
    B_start= 0.0001
    
    # =============================================================================
    #  (1) Definieer een Python-functie die het model bevat, in dit geval een 
    #  rechte lijn
    #  B is een vector met parameters, in dit geval twee (A = B[0], B = B[1])
    #  x is de array met x-waarden
    # =============================================================================
    def f(B, x):
        return B[0] * x + B[1]
    
    
    odr_model = odr.Model(f)
    odr_data  = odr.RealData(x,y,sy=sig_y)          # we hebben geen sx opgegeven aangezien er geen onzekerheid in de tijd aanwezig is
    odr_obj   = odr.ODR(odr_data,odr_model,beta0=[A_start,B_start])
    odr_obj.set_job(fit_type=2)                     # Definieer welke fit-functie je wil gebruiken
    odr_res   = odr_obj.run()                       # Voer de Fit functie uit
    par_best = odr_res.beta                         # vraag de beste schatter op
    par_sig_ext = odr_res.sd_beta                   # de externe onzerheden
    par_cov = odr_res.cov_beta                      # de interne covariantiematrix
    
    print(" De (INTERNE!) covariantiematrix  = \n", par_cov)
    
    # (6d) De chi-kwadraat en de gereduceerde chi-kwadraat van deze aanpassing
    chi2 = odr_res.sum_square
    print("\n Chi-squared         = ", chi2)
    chi2red = odr_res.res_var
    print(" Reduced chi-squared = ", chi2red, "\n")
    
    # (6e) Een compacte weergave van de belangrijkste resultaten als output
    odr_res.pprint()
   # print(output.beta)
    
    # Hier plotten we ter controle de aanpassing met de dataset (niet opgemaakt)
    xplot=np.linspace(0, 25, num=100)
    ax.plot(xplot,(par_best[0] * xplot),'r-')
    return

odrding(meet1.c100)

