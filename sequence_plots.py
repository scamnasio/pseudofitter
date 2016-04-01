''' 
Written by Sara Camnasio
CUNY Hunter College class of 2016
sara.camnasio@gmail.com
'''

import numpy as np
import astrotools2 as at
from matplotlib import pyplot as plt
from numpy import *
import utilities as u
import pyspeckit
import StringIO
import corner
import shutil
import os
import decimal
from scipy.stats import norm
from astropy.io import fits


def spt():
	
	'''
	*n*
		int - number of MC iterations
	'''

	# path_input1 = input("Enter last bracket of young obj path:")
	# path_input2 = input("Enter last bracket of field obj path:")
	# path_input3 = input("Enter last bracket of blue obj path:")

	# young:
	path1 = input("Enter last bracket of first spt obj path:")
	name1 = input("Enter name of 1st obj:")
	#standard:
	path2 = input("Enter last bracket of second spt obj path:")
	name2 = input("Enter name of 2nd obj:")
	#blue:
	path3 = input("Enter last bracket of third spt obj path:")
	name3 = input("Enter name of 3rd obj:")

	# path = "/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_Data/{0}".format(path_input)

	

	raw1 = np.genfromtxt(path1, delimiter='', dtype = float)
	raw2 = np.genfromtxt(path2, delimiter='', dtype = float)
	raw3 = np.genfromtxt(path3, delimiter='', dtype = float)

	W1_1 = np.array(raw1[:,0])
	F1_1 = np.array(raw1[:,1])
	U1_1 = np.array(raw1[:,2])

	W2_1 = np.array(raw2[:,0])
	F2_1 = np.array(raw2[:,1])
	U2_1 = np.array(raw2[:,2])

	W3_1 = np.array(raw3[:,0])
	F3_1 = np.array(raw3[:,1])
	U3_1 = np.array(raw3[:,2])

	'''
	If Angstroms:
	'''
	# W1 = W1/10000
	'''
	- If U1 does not exist (data without uncertainty) please comment out line 34 ("U = np.array(raw[:,2])") and uncomment the first line below
	- If U1 is SNR uncomment the second line
	'''
	# U1 = 0.05*F1
	
	# Converting SNR
	# U1_1 = F1_1/U1_1
	U2_1 = F2_1/U2_1
	U3_1 = F3_1/U3_1

	# Trimming the data
	W1_2,F1_2,U1_2 = [i[np.where(np.logical_and(W1_1>1.15, W1_1<1.325))] for i in [W1_1,F1_1,U1_1]]
	W2_2,F2_2,U2_2 = [i[np.where(np.logical_and(W2_1>1.15, W2_1<1.325))] for i in [W2_1,F2_1,U2_1]]
	W3_2,F3_2,U3_2 = [i[np.where(np.logical_and(W3_1>1.15, W3_1<1.325))] for i in [W3_1,F3_1,U3_1]]

	# 1.325
	W1_2[np.isnan(W1_2)] = 0
	F1_2[np.isnan(F1_2)] = 0
	U1_2[np.isnan(U1_2)] = 0

	W2_2[np.isnan(W2_2)] = 0
	F2_2[np.isnan(F2_2)] = 0
	U2_2[np.isnan(U2_2)] = 0

	W3_2[np.isnan(W3_2)] = 0
	F3_2[np.isnan(F3_2)] = 0
	U3_2[np.isnan(U3_2)] = 0

	
	# Recombining the W,F into one array in order to normalize it
	
	band = [1.15, 1.325]
	recombined1 = np.vstack([W1_2,F1_2,U1_2])
	recombined2 = np.vstack([W2_2,F2_2,U2_2])
	recombined3 = np.vstack([W3_2,F3_2,U3_2])

	# data_clean = u.scrub(recombined) 
	# had to remove SCRUB because it fucked up baseline
	data_clean1 = recombined1
	data1 = np.array(at.norm_spec(data_clean1, band))

	data_clean2 = recombined2
	data2 = np.array(at.norm_spec(data_clean2, band))

	data_clean3 = recombined3
	data3 = np.array(at.norm_spec(data_clean3, band))


	W1_3 = data1[:,0]
	F1_3 = data1[:,1]
	U1_3 = data1[:,2]

	W2_3 = data2[:,0]
	F2_3 = data2[:,1]
	U2_3 = data2[:,2]

	W3_3 = data3[:,0]
	F3_3 = data3[:,1]
	U3_3 = data3[:,2]

	# Squeezing into one dimension and shifting the wavelength to the 0 axis
	W1 = np.squeeze(W1_3) 
	W1[:] = [x - 1.15 for x in W1] #original was 1.15, 1.24 is to test Kraus idea  
	F1 = np.squeeze(F1_3)
	F1 = np.array(F1)
	U1 = np.squeeze(U1_3)
	U1 = np.array(U1)

	W2 = np.squeeze(W2_3) 
	W2[:] = [x - 1.15 for x in W2]
	F2 = np.squeeze(F2_3)
	F2 = np.array(F2)
	U2 = np.squeeze(U2_3)
	U2 = np.array(U2)

	W3 = np.squeeze(W3_3) 
	W3[:] = [x - 1.15 for x in W3]
	F3 = np.squeeze(F3_3)
	F3 = np.array(F3)
	U3 = np.squeeze(U3_3)
	U3 = np.array(U3)

	# SNR = input("SNR:")
	# if SNR == "y":
	# 	U = F/U
	# elif SNR == "n/a":
	# 	U = 0.05*F
	# elif SNR == "n":
	# 	U = U
	# Check what the spectrum looks like:
	
	plt.figure()
	
	x1 = linspace(0, 0.175, 200)
	y1 = (3041.96*x1**4 - 1427.6347*x1**3 + 210.0422*x1**2 - 6.8367*x1 + 0.8045)
	plt.errorbar(W1, F1, yerr=U1, color='black')
	plt.plot(x1, y1, color='red', linewidth=2)
	plt.xlim(0, 0.175)
 
	
	x2 = linspace(0, 0.175, 200)
	y2 = (2395.68740381*x2**4 - 1378.5605723*x2**3 + 236.780770491*x2**2 - 11.4938467479*x2**1 + 2.01029264782)
	plt.errorbar(W2, F2+1, yerr=U2, color='black') 
	plt.plot(x2, y2, color='grey', linewidth=2)
	plt.xlim(0,0.175)
	
	
	x3 = linspace(0, 0.175, 200)
	y3 = (3071.37*x3**4 - 1503.4091*x3**3 + 234.6684*x3**2 - 11.2639*x3**1 + 3.0506)
	plt.errorbar(W3, F3+2, yerr=U3, color='black') 
	plt.plot(x3, y3, color='blue', linewidth=2)
	plt.xlim(0,0.175)


	plt.ylabel('Normalized Flux F$_{\lambda}+C $')
	plt.xlabel('Wavelength ($\mu$m) - $W_0$')
	plt.annotate('{0}'.format(name1), xy=(0.11, 0.7), xytext=(0.11, 0.7), color='red', weight='semibold', fontsize=15)
	plt.annotate('{0}'.format(name2), xy=(0.11, 1.7), xytext=(0.11, 1.7), color='grey', weight='semibold', fontsize=15)
	plt.annotate('{0}'.format(name3), xy=(0.11, 2.7), xytext=(0.11, 2.7), color='blue', weight='semibold', fontsize=15)
	plt.annotate('{0}'.format(title), xy=(0.155, 0.3), xytext=(0.155, 0.3), color='black', weight='semibold', fontsize=40)
	plt.ylim(0.15,3.4)




def color_seq():

	
	numbers = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
	colors = [ 'yellow','orange', 'red','orange', 'yellow', 'orange', 'red', 'orange', 'yellow' ]
	plt.figure()
	plt.ylabel('Normalized Flux F$_{\lambda}+C $')
	plt.xlabel('Wavelength ($\mu$m) - $W_0$')
	plt.ylim(0.15,5)
	plt.xlim(0,0.175)

	for n,i in zip(numbers, colors):
		print i
		name = input("Enter the name of object: ")
		spt = input("Enter the spt of object: ")
		path = input("Enter last branch of data path: ")
		color = input ("Enter color")
		snr = input("SNR? y/n: ")
		coeff4 = input("Enter coefficient 4:")
		coeff3 = input("Enter coefficient 3:")
		coeff2 = input("Enter coefficient 2:")
		coeff1 = input("Enter coefficient 1:")
		coeff0 = input("Enter coefficient 0:")

		fullpath = "/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_Data/{0}".format(path)
		raw = np.genfromtxt(fullpath, delimiter='', dtype = float)
		
		W1 = np.array(raw[:,0])
		F1 = np.array(raw[:,1])
		U1 = np.array(raw[:,2])

		# IF SNR:
		if snr == 'y':
			U1 = F1/U1
		elif snr == 'n':
			U1 = U1


		# Trimming the data
		W2,F2,U2 = [i[np.where(np.logical_and(W1>1.15, W1<1.325))] for i in [W1,F1,U1]]
		
		# 1.325
		W2[np.isnan(W2)] = 0
		F2[np.isnan(F2)] = 0
		U2[np.isnan(U2)] = 0
		
		# Recombining the W,F into one array in order to normalize it
		band = [1.15, 1.325]
		recombined = np.vstack([W2,F2,U2])
		data = np.array(at.norm_spec(recombined, band))
		
		W3 = data[:,0]
		F3 = data[:,1]
		U3 = data[:,2]
		
		W1 = np.squeeze(W3) 
		W1[:] = [x - 1.15 for x in W1] #original was 1.15, 1.24 is to test Kraus idea  
		F1 = np.squeeze(F3)
		F1 = np.array(F1)
		U1 = np.squeeze(U3)
		U1 = np.array(U1)

		x1 = linspace(0, 0.175, 200)
		y1 = (coeff4*x1**4 - coeff3*x1**3 + coeff2*x1**2 - coeff1*x1 + coeff0 + n)
		plt.errorbar(W1, F1+n, yerr=U1, color="black")
		plt.annotate('{0}'.format(name), xy=(0.11, 0.7), xytext=(0.11, 0.7), color='black', weight='semibold', fontsize=15)
		plt.plot(x1, y1, color=color, linewidth=2)
		plt.xlim(0, 0.175)
plt.show()
	