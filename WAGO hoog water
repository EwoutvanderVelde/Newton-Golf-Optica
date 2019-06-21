# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:54:40 2019

@author: Ewout van der Velde
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr


# Tijdens het script geeft elke ODR de optie om een grafiek te maken. 
# Mid_plot zijn alle ODR waar frequentie word bepaald
# Final_plot is de ODR waar Labda en Frequentie theoretisch en in praktijk tegen elkaar worden uitgezet
# Om tijd te besparen in het berekenen is het verstandig om Mid_plot op False te zetten
Mid_plot = False
Final_plot = True


#fontsize
fts = 27

# Natuurconstante
g = 9.81


# Startwaarden voor onze bak met water
H = 0.150           # De waterhoogte
Tau = 0.072         # De oppervrakte spanning van water op kamertemperatuur
Rho = 997           # De dichtheid van water


# Een kleine failsafe om veel tijd en pijn te besparen
if Mid_plot == True:
    temp = input("Do you really want to plot all mid_plots? Y/N ")
    if temp == "Y":
        quit
    else:
        Mid_plot = False
        
        
# Inlezen van alle data bestanden            
File_5V = "H5V_1Labda.dat"
File_6V = "H6V_1Labda.dat"
File_7V = "H7V_2Labda.dat"
File_8V = "H8V_2Labda.dat"
File_9V = "H9V_2Labda.dat"
File_10V = "H10V_2Labda.dat"
File_11V = "H11V_2Labda.dat"
File_12V = "H12V_2Labda.dat"
File_13V = "H13V_2Labda.dat"
File_14V = "H14V_2Labda.dat"
File_15V = "H15V_2Labda.dat"
File_16V = "H16V_2Labda.dat"
File_17V = "H17V_2Labda.dat"
File_18V = "H18V_2Labda.dat"
File_19V = "H19V_2Labda.dat"
File_20V = "H20V_2Labda.dat"


# We definieren een functie om de frequentie te bepalen van de waterfolven.
def odrfunc(File, row, label, kx, Freq, Ampl, Offset): 
    Data = np.genfromtxt(File)          # Lees de data in
    
    y = (((Data[:,row]).flatten()))
    x = (Data[:,0]).flatten()
    sig_y = (np.zeros(len(Data[:,1])) + 0.5)
    
    # Deze functie is om onnidige plotjes te voorkomen
    if (Mid_plot == True):
        f,ax = plt.subplots(1)
        ax.set_xlabel("$t$ (s)")
        ax.set_ylabel("$Golfhoogte$ (mm)")
        ax.errorbar(x = x,
                    y = y,
                    yerr = sig_y,
                    fmt = 'k.',
                    label = "Gemeten")
    
    kx = kx
    Freq = Freq
    Ampl = Ampl
    Offset = Offset

    # de onderstaande functie is de golffunctie eta = a*sin(k*x -omega*t) + phi
    def f(B, x):
        return B[2]*np.sin(B[0]-2*np.pi*B[1]*x) + B[3]
    
    odr_model = odr.Model(f)
    odr_data  = odr.RealData(x,y,sy=sig_y)          # we hebben geen sx opgegeven aangezien er geen onzekerheid in de tijd aanwezig is
    odr_obj   = odr.ODR(odr_data,odr_model,beta0=[kx, Freq, Ampl, Offset])
    odr_obj.set_job(fit_type=2)                     # 2 want alleen onzekerheid in x
    odr_res   = odr_obj.run()                       # Voer de Fit functie uit
    par_best = odr_res.beta                         # vraag de beste schatter op
    
    par_sig_ext = odr_res.sd_beta                   # de externe onzerheden

    chi2red = odr_res.res_var
    
    
    print("\n{}: f = {:.3f}±{:.3f} Hz".format(label, par_best[1], par_sig_ext[1]))
    print("Reduced chi-squared = {:.3f}".format(chi2red))
    
    # Hier plotten we ter controle de aanpassing met de dataset
    if (Mid_plot == True):
        xplot=np.linspace(0, (len(Data[:,1]))/50, num=10000)
        ax.plot(xplot, (par_best[2] *np.sin(par_best[0] - 2*np.pi*par_best[1]*xplot) + par_best[3]), 'r-', label="Linefit")
        ax.set_title("Meting bij $U$ = {} \n$Freq$ = {:.3f}±{:.3f} Hz".format(label, par_best[1], par_sig_ext[1]))
        ax.legend()
        plt.savefig("{}.pdf".format(label))
    
    f_list.append(odr_res.beta[1])
    f_sig_list.append(odr_res.sd_beta[1])
    return

# De volgende functie bekend Lambda
def Calc_Lambda(filename, DLambda):
    # We maken gebruik van de afstand tussen de sensoren. 
    # Deze staat in de header van het databestand.
    # Dit betekend dat het databestand moeten inlezen als string.
    
    f = open(filename, "r")                     # Open het databestand
    block = f.readlines()                       # Lees de data in 
    
    par_data = []                               # Maakt een lege Python lijst voor de parameter data
    for line in block:
        par_data.append( line.split() )         # Split de data per regel
    
    param = []
    param.append(par_data[9])                   # Kijk naar regel 9
    param.append(par_data[12])                  # Kijk naar regel 2
    
    para = []                                   # Lege lsit voor de parameters
    for elem in param:
        flt = float(elem[2])                    # Zet getal om van string naar float
        para.append(flt)                        # Zet de parameter in een lijst
    
    Dx = (para[1] - para[0])/100                # Delen door 100 om om te zetten naar meters
    
    Sig_Dx = 1/25                               # Onzekerheid is geschat op 1/25 van de gemeten afstand
        
    Lambda_best = Dx/DLambda                    # Lambda is in veel gevallen 2x zo klein als de gemeten dx
    Sig_Lambda = (Sig_Dx*Lambda_best)/DLambda
    
    Lambda_list.append(Lambda_best)
    Lambda_sig_list.append(Sig_Lambda)
    return 


def Analyse(File, row, label, kx, Freq, Ampl, Offset, NLambda):
    odrfunc(File, row, label, kx, Freq, Ampl, Offset)
    Calc_Lambda(File, NLambda)
    return





f_list = []
f_sig_list = []
Lambda_list = []
Lambda_sig_list = []

print("\n######### Analyse van databestanden #########")
#ODR volg is: bestand, row, label, kx, Freq, Ampl, offset, NLambda
Analyse(File_5V, 2, "5V_HOOG", 1.60, 0.98, 0.5, 0.38, 1)
Analyse(File_6V, 2, "6V_HOOG", 0.76, 1.21, 0.628, -0.355, 1)
Analyse(File_7V, 2, "7V_HOOG", -1.456, 1.44, 0.619, 0.42, 2)
Analyse(File_8V, 2, "8V_HOOG", 0.32, 1.65, -1.7, 0.59, 2)
Analyse(File_9V, 2, "9V_HOOG", 1.69, 1.90, -0.94, 0.4, 2)
Analyse(File_10V, 2, "10V_HOOG", 0.0082, 2.07, 1.41, 0.06, 2)
Analyse(File_11V, 1, "11V_HOOG", -0.23, 2.32, 1.71, -0.19, 2)
Analyse(File_12V, 2, "12V_HOOG", 3.2, 2.58, -1.92, 0.008, 2)
Analyse(File_13V, 2, "13V_HOOG", -0.60, 2.83, 1.62, -0.334, 2)
Analyse(File_14V, 2, "14V_HOOG", 1.45, 3.13, -1.31, 0.174, 2)
Analyse(File_15V, 2, "15V_HOOG", 1.74, 3.44, -2.43, -2, 2)
Analyse(File_16V, 2, "16V_HOOG", 5.99, 3.77, -1.70, -0.36, 2)
Analyse(File_17V, 2, "17V_HOOG", 7.72, 4.12, -1.18, -0.44, 2)
Analyse(File_18V, 2, "18V_HOOG", 6.97, 4.39, 0.678, 0.149, 2)
Analyse(File_19V, 2, "19V_HOOG", 3.11, 4.70, 0.46, 0.063, 2)
Analyse(File_20V, 2, "20V_HOOG", 3.04, 4.99, 0.605, -0.065, 2)

    
f_list = np.array(f_list)
f_sig_list = np.array(f_sig_list)
Lambda_list = np.array(Lambda_list)
Lambda_sig_list = np.array(Lambda_sig_list)


############################### ODR TIJD #######################################

y = f_list
x = Lambda_list
sig_y = f_sig_list 
sig_x = Lambda_sig_list


if (Final_plot == True):
    f,ax = plt.subplots(1, figsize = (12,8))
    ax.set_xlabel("$\lambda$ (m)", fontsize = fts)
    ax.set_ylabel("$f$ (s$^-$$^1$)", fontsize = fts)
    ax.tick_params(axis="both", labelsize=24) 
    ax.errorbar(x = x,
                y = y,
                xerr = sig_x,
                yerr = sig_y,
                fmt = 'k.',
                label = "Meetpunten")



def f(B, x):
    return np.sqrt(( (g / ((2*np.pi)/x)) * np.tanh((2*np.pi / x)*B[0] ) ) / x**2)

odr_model = odr.Model(f)
odr_data  = odr.RealData(x,y,sy=sig_y, sx=sig_x)          
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[H])
odr_obj.set_job(fit_type=0)                     # 0 door onzekerheden in x en y
odr_res   = odr_obj.run()                       

par_best = odr_res.beta                         # vraag de beste schatter op
par_sig_ext = odr_res.sd_beta                   # de externe onzerheden
chi2red = odr_res.res_var

print("\n######### Govonden waarden #########")
print("H = {:.3f}±{:.3f} m".format(par_best[0], par_sig_ext[0]))
#print("\N{greek small letter tau} = {:.3f}±{:.3f} N/m".format(par_best[1], par_sig_ext[1]))
print(" Reduced chi-squared = {:.3f}".format(chi2red))

if (Final_plot == True):
    xplot = np.linspace(0.07, 1, num=100)
    ax.plot(xplot, np.sqrt(( (g / ((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*par_best[0] ) ) / xplot**2 ),'r-', label="Linefit zonder Tau")
    ax.plot(xplot, np.sqrt(( (g / ((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*H ) ) / xplot**2 ),'b:', label="Theorie zonder Tau")
    ax.plot(xplot, np.sqrt( ((g / ((2*np.pi)/xplot) + (Tau/Rho)*((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*H )) / xplot**2 ),'y-', label="Theorie met Tau")
    #ax.set_title("Golflengte uitgezet tegen de Frequentie bij $\\rho$ = {} kg/m$^3$ \n$H$ = {:.3f}±{:.3f} m, \N{greek small letter tau} = {:.3f}±{:.3f} N/m \nReduced chi-squared = {:.3f} ".format(Rho, par_best[0], par_sig_ext[0], par_best[1], par_sig_ext[1], chi2red))
    ax.legend(fontsize = 20)
    
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax, 6, loc=9) # zoom-factor: 2.5, location: upper-left
    axins.plot(xplot, np.sqrt(( (g / ((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*par_best[0] ) ) / xplot**2 ),'r-', label="Linefit zonder Tau")
    axins.plot(xplot, np.sqrt(( (g / ((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*H ) ) / xplot**2 ),'b:', label="Theorie zonder Tau")
    axins.plot(xplot, np.sqrt( ((g / ((2*np.pi)/xplot) + (Tau/Rho)*((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*H )) / xplot**2 ),'y-', label="Theorie met Tau")
    axins.errorbar(x = x,
                   y = y,
                   xerr = sig_x,
                   yerr = sig_y,
                   fmt = 'k.',
                   label = "Meetpunten")   
    x1, x2, y1, y2 = 0.08, 0.105, 4, 4.4 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    plt.savefig("Lambda_Frequentie_Hoog.pdf")

print("\n######### Significantie zonder tau #########")

DeltaH = np.abs(H - par_best[0])
Sig95H = DeltaH/(2*par_sig_ext[0])

if Sig95H > 1:
    print("De gemeten waarde van H wijkt significant af van de theoretische waarde")
else:
    print("H wijkt niet significant af")



########################## 2e ODR

if (Final_plot == True):
    f,ax = plt.subplots(1, figsize = (12,8))
    ax.set_xlabel("$\lambda$ (m)", fontsize = fts)
    ax.set_ylabel("$f$ (s$^-$$^1$)", fontsize = fts)
    ax.tick_params(axis="both", labelsize=24) 
    ax.errorbar(x = x,
                y = y,
                xerr = sig_x,
                yerr = sig_y,
                fmt = 'k.',
                label = "Meetpunten")



def f(B, x):
    return np.sqrt( ((g / ((2*np.pi)/x) + (B[1]/Rho)*((2*np.pi)/x) ) * np.tanh((2*np.pi / x)*B[0] )) / x**2 )

odr_model = odr.Model(f)
odr_data  = odr.RealData(x,y,sy=sig_y, sx=sig_x)          
odr_obj   = odr.ODR(odr_data,odr_model,beta0=[H, Tau])
odr_obj.set_job(fit_type=0)                     # 0 door onzekerheden in x en y
odr_res   = odr_obj.run()                       

par_best = odr_res.beta                         # vraag de beste schatter op
par_sig_ext = odr_res.sd_beta                   # de externe onzerheden
chi2red = odr_res.res_var

print("\n######### Govonden waarden #########")
print("H = {:.3f}±{:.3f} m".format(par_best[0], par_sig_ext[0]))
print("\N{greek small letter tau} = {:.3f}±{:.3f} N/m".format(par_best[1], par_sig_ext[1]))
print(" Reduced chi-squared = {:.3f}".format(chi2red))

if (Final_plot == True):
    xplot = np.linspace(0.07, 1, num=100)
    ax.plot(xplot, np.sqrt( ((g / ((2*np.pi)/xplot) + (par_best[1]/Rho)*((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*par_best[0] )) / xplot**2 ),'r-', label="Linefit met Tau")
    ax.plot(xplot, np.sqrt(( (g / ((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*H ) ) / xplot**2 ),'b:', label="Theorie zonder Tau")
    ax.plot(xplot, np.sqrt(( (g / ((2*np.pi)/xplot) + (Tau/Rho)*((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*H )) / xplot**2 ),'y-', label="Theorie met Tau")
#    ax.set_title("Golflengte uitgezet tegen de Frequentie bij $\\rho$ = {} kg/m$^3$ \n$H$ = {:.3f}±{:.3f} m, \N{greek small letter tau} = {:.3f}±{:.3f} N/m \nReduced chi-squared = {:.3f} ".format(Rho, par_best[0], par_sig_ext[0], par_best[1], par_sig_ext[1], chi2red))
    ax.legend(fontsize = 20)

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax, 6, loc=9) # zoom-factor: 2.5, location: upper-left
    axins.plot(xplot, np.sqrt( ((g / ((2*np.pi)/xplot) + (par_best[1]/Rho)*((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*par_best[0] )) / xplot**2 ),'r-', label="Linefit met Tau")
    axins.plot(xplot, np.sqrt(( (g / ((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*H ) ) / xplot**2 ),'b:', label="Theorie zonder Tau")
    axins.plot(xplot, np.sqrt( ((g / ((2*np.pi)/xplot) + (Tau/Rho)*((2*np.pi)/xplot)) * np.tanh((2*np.pi / xplot)*H )) / xplot**2 ),'y-', label="Theorie met Tau")
    axins.errorbar(x = x,
                   y = y,
                   xerr = sig_x,
                   yerr = sig_y,
                   fmt = 'k.',
                   label = "Meetpunten")   
    x1, x2, y1, y2 = 0.08, 0.105, 4, 4.4 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
    plt.savefig("Lambda_Frequentie_Hoog2.pdf")
    
DeltaTau = np.abs(Tau - par_best[1])
Sig95Tau = DeltaTau/(2*par_sig_ext[1])

DeltaH = np.abs(H - par_best[0])
Sig95H = DeltaH/(2*par_sig_ext[0])


print("\n######### Significantie met Tau #########")

if Sig95Tau > 1:
    print("De gemeten waarde van \N{greek small letter tau} wijkt significant af van de theoretische waarde")
else:
    print("\N{greek small letter tau} wijkt niet significant af")
    
if Sig95H > 1:
    print("De gemeten waarde van H wijkt significant af van de theoretische waarde")
else:
    print("H wijkt niet significant af")
