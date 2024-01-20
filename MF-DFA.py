#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys, csv, os
from sklearn import linear_model

def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print('Error: Creating directory. ' + directory)

def multidim_cumsum(a):
	out = a[...,:].cumsum(-1)[...,:]
	for i in range(2,a.ndim+1):
		np.cumsum(out, axis=-i, out=out)
	return out

# Define mathematical function for curve fitting 
def func(x, y, a1, a2, a3, a4, a5, c): 
	return a1*x + a2*y + a3*x*y + a4*x*x + a5*y*y + c

deg_DFA = 1; qmin = float(sys.argv[1]); qmax = float(sys.argv[2]); dq = float(sys.argv[3]); num = str(sys.argv[4])

################################################################# LEEMOS BASE DE DATOS
#my_dict_back = np.load('../../../../../../media/carlos/DATA/Jardi/Train/Normal/'+str(num)+'.npz', allow_pickle= True) #922
#my_dict_back = np.load('../../../../../../media/carlos/DATA/Jardi/Train/Non-COVID-19/'+str(num)+'.npz', allow_pickle= True) #787
my_dict_back = np.load('../../../../../../media/carlos/DATA/Jardi/Train/COVID-19/'+str(num)+'.npz', allow_pickle= True) #1804
dic_boxes = my_dict_back['boxes'].item()

###################################################################

for side in dic_boxes.keys(): # Recorre cada lado
	createFolder('2D_MFDFA_Pulmón_N'+str(num)+'_'+str(side)+'/')
	M=[];Q=np.arange(qmin-dq,qmax+2*dq,dq);NQ=len(Q);Mpre=[]  # Crea una lista con los valores de Q en función de qmin y qmax

	for j in range(NQ+1):
		M.append([]);Mpre.append([])

	for len_s in dic_boxes[side]: # Recorre cada tamano de caja
		# tx malla de la caja
		tx = np.array(range(int(len_s))); ty = tx; TX, TY = np.meshgrid(tx, ty) #es necesario para hacer el fit
		F=[]
		for box in range(dic_boxes[side][len_s].shape[0]): # Recorre cada caja de un determinado tamaño
			Sub_IM = dic_boxes[side][len_s][box] #renombro la caja a anlizar

			Sub_IM = Sub_IM - np.mean(Sub_IM) #restamos la media
			IM = multidim_cumsum(Sub_IM) #calculo la integracion de la superficie

			x1, y1, z1 = TX.flatten(), TY.flatten(), IM.flatten()
			x1y1, x1x1, y1y1 = x1*y1, x1*x1, y1*y1
			X_data = np.array([x1, y1, x1y1, x1x1, y1y1]).T  
			Y_data = z1

			reg = linear_model.LinearRegression().fit(X_data, Y_data)
			a1 = reg.coef_[0]; a2 = reg.coef_[1]; a3 = reg.coef_[2]; a4 = reg.coef_[3]; a5 = reg.coef_[4]; c = reg.intercept_

			ZZ = func(TX, TY, a1, a2, a3, a4, a5, c)
			
			F_matrix = IM-ZZ
			F.append(np.mean(F_matrix**2))

		M[0].append(np.log10(int(len_s)));Mpre[0].append(int(len_s))

		for j in range(NQ):
			if j == int(NQ/2):
				M[j+1].append(np.log10( np.exp(0.5*np.mean(np.log(np.array(F)))) ))
				Mpre[j+1].append(np.exp(0.5*np.mean(np.log(np.array(F)))) )
			else:
				M[j+1].append(np.log10( np.mean(np.array(F)**(Q[j]/2.0))**(1.0/Q[j]) ))
				Mpre[j+1].append(np.mean(np.array(F)**(Q[j]/2.0))**(1.0/Q[j]) )

	np.savetxt('2D_MFDFA_Pulmón_N'+str(num)+'_'+str(side)+'/F_qs.txt',np.matrix(Mpre).transpose(),fmt='%s')
	MHT=[[],[],[]]; MHT[0]=Q
	for i in range(NQ):
		h=np.polyfit(M[0],M[i+1],1)[0]
		MHT[1].append(h);MHT[2].append(Q[i]*h-2)

	np.savetxt('2D_MFDFA_Pulmón_N'+str(num)+'_'+str(side)+'/h_tau-spectrum.txt',np.matrix(MHT).transpose(),fmt='%s')

	Maf=[[],[]]
	for k in range(1,NQ-1):
		a=(MHT[2][k+1]-MHT[2][k-1])/(2*dq)
		Maf[0].append(a); Maf[1].append(MHT[0][k]*a-MHT[2][k])
		
	np.savetxt('2D_MFDFA_Pulmón_N'+str(num)+'_'+str(side)+'/MF-spectrum.txt',np.matrix(Maf).transpose(),fmt='%s')
