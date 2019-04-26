import numpy as np

lat1 = np.array([10.5, 10.5, 10.5]) # De aflezing op de mm lat op de microscoopo
wiel = np.array([6.2, 4.0, 7.6]) # De aflezing van het draaiwiel
mm1 = lat1 + wiel/200 # Delen door 200 omdat je het wiel 2 maal honderd stapjes moet verzetten op een mmm te verplaatsen

lat2 = np.array([12.0, 12.0, 12.0])
wiel2 = np.array([31.8, 36.6, 31.2])
mm2 = lat2 + wiel2/200 

dia = mm2 - mm1 # We meten het verschil tussen de randen om diameter te bepalen
schatter = np.mean(dia)
std = np.std(dia, ddof=1)/np.sqrt(len(dia))         # Onzekerheid
print("De diameter = {:.2f} +- {:.2f} mm".format(schatter, std)) 

# Als geheel getal dan is er constuctieve interferentie
n = 1 #brekingsindex stof tussen de lens en glasplaatje
labda = 546.08 # We werkem met het groene filter

fi = (2*n*schatter)/labda + 1/2
print ("fi =",fi) # wanneer fi = 0.5 hebben we uitdoving

temp1 = np.mean([5.21, 5.21, 5.21, 5.21]) # Voorbeeld berekening kromtestraal 
temp2 = np.mean([5.00, 5.00, 5.00, 5.01])

a = 30.1
d = temp1 - temp2

R = (a**2)/(6*d) + d/2
print("Kromtestraal = {:.3f} mm".format(R)) 
