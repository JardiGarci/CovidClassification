#!/usr/bin/env python3
import numpy as np
import csv
#import math 

'''q=[]; h=[]; tau=[]
for i in range(1,11):
	q.append([]); h.append([]); tau.append([])

	with open('MFDFA_BN_130k_'+str(i)+'_sini200_sfin10000/h_tau-spectrum_BN_130k_'+str(i)+'.txt') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=' ')
		for a in readCSV:
			q[i-1].append(float(a[0])); h[i-1].append(float(a[1])); tau[i-1].append(float(a[2]))

for a,b,c in zip(np.mean(q, axis = 0), np.mean(h, axis = 0), np.std(h, axis = 0)):
	print(a, b, c)

print("&")

for a,b,c in zip(np.mean(q, axis = 0), np.mean(tau, axis = 0), np.std(tau, axis = 0)):
	print(a, b, c)'''

M=['R_Normal','R_Non-COVID-19','R_COVID-19']
L = ['izquierdo','derecho']
S = [922,787,1804]       # Número de muestras por clase
Errors=[]

for l in L:
	for m,s in zip(M,S):
		a=[]; f=[]
		for i in range(1,s+1):      # Recorre todas las radiografías de cada clase
			pa=[];pf=[]
			
			with open(m+'/2D_MFDFA_Pulmón_N'+str(i)+'_'+l+'/MF-spectrum.txt') as csvfile:
				readCSV = csv.reader(csvfile, delimiter=' ')
				for A in readCSV:
					pa.append(float(A[0])); pf.append(float(A[1]))

			if np.isnan(np.mean(pa)) == True:
				Errors.append([m,l,i])
			else:
				a.append(pa); f.append(pf)

		np.savetxt('Ave_MF_'+m+'_'+l+'.txt',np.matrix([np.mean(a, axis = 0), np.mean(f, axis = 0), np.std(a, axis = 0), np.std(f, axis = 0)]).transpose(),fmt='%s')

np.savetxt('Errores.txt',np.matrix(Errors),fmt='%s')#.transpose(),fmt='%s')
