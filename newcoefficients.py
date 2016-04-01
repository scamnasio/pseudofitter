''' 
Written by Sara Camnasio
CUNY Hunter College class of 2016
sara.camnasio@gmail.com
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from matplotlib import axis
from matplotlib.pyplot import step, legend, xlim, ylim, show
import collections
import matplotlib.lines as mlines

source = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = float)
source2 = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = str)

name = np.array(source2[:,0])
spt = np.array(source[:,2])
types = np.array(source2[:,4])
JK = np.array(source[:,7])
fourth = np.array(source[:,8])
fourth_err = np.array(source[:,9])
third = np.array(source[:,10])
third_err = np.array(source[:,11])
second = np.array(source[:,12])
second_err = np.array(source[:,13])
first = np.array(source[:,14])
first_err = np.array(source[:,15])
zero = np.array(source[:,16])
zero_err = np.array(source[:,17])

def fits():

	categories = []
	for n in types:
		if n == 'young':
			categories.append("r")
		elif n == "blue":
			categories.append("b")
		elif n == "red":
			categories.append("m")
		elif n == "standard":
			categories.append("k")

	categories = np.array(categories)

	''' Spectral type VS Coeff '''
	
	''' 4th '''
	plt.figure()
	P = plt.subplot(321)
	plt.ylabel("$4^{th} Coefficient$", fontsize=17)
	plt.scatter(spt, fourth, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(spt, fourth, yerr=fourth_err, fmt=None, alpha=0.5, ecolor='k')
	plt.xlim(9,20)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P.set_xticklabels(labels)

	''' 3rd '''
	
	P2 = plt.subplot(322)
	plt.scatter(spt, third, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(spt, third, yerr=third_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylabel("$3^{rd} Coefficient$", fontsize=17)
	plt.xlim(9,20)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P2.set_xticklabels(labels)
	   
	''' 2ND '''

	P3 = plt.subplot(323)
	plt.ylabel("$2^{nd} Coefficient$", fontsize=17)
	plt.scatter(spt, second, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(spt, second, yerr=second_err, fmt=None, alpha=0.5, ecolor='k')
	plt.xlim(9,20)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P3.set_xticklabels(labels)

	
	''' 1ST '''
	P4 = plt.subplot(324)
	#plt.title("J-K deviation vs SpT")
	plt.xlabel("$Spectral Type$", fontsize=13)
	plt.ylabel("$1^{st} Coefficient$", fontsize=17)
	plt.scatter(spt, first, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(spt, first, yerr=first_err, fmt=None, alpha=0.5, ecolor='k')
	plt.xlim(9,20)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P4.set_xticklabels(labels)
	
	''' 0TH '''
	P5 = plt.subplot(325)
	#plt.title("J-K deviation vs SpT")
	plt.xlabel("$Spectral Type$", fontsize=13)
	plt.scatter(spt, zero, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(spt, zero, yerr=zero_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylabel("$0^{th} Coefficient$", fontsize=17)
	plt.xlim(9,20)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P5.set_xticklabels(labels)
	# plt.legend(loc=1, fontsize=10)
	
	# Create fake data (void) to use the linestyles for the legend of the sigmas:
	data1 = mlines.Line2D([], [], color='r', marker='o', label="Red", linestyle='')
	data2 = mlines.Line2D([], [], color='m', marker='o', label="Young", linestyle='')
	data3 = mlines.Line2D([], [], color='b', marker='o', label="Blue", linestyle='')
	plt.legend((data1, data2, data3), ("Red", "Young", "Blue"), loc=1,  numpoints=1)

	
	plt.show()
	
	''' 
	___________________________________
	___________________________________
	___________________________________

	Color VS Coefficients
	
	4TH 

	'''

	plt.figure()
	plt.subplot(321)
	plt.ylabel("$4^{th} Coefficient$", fontsize=17)
	plt.scatter(JK, fourth, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(JK, fourth, yerr=fourth_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylim(-750, 10500)
	plt.xlim(.5, 3)
	
	''' 3RD '''
	
	plt.subplot(322)
	plt.ylabel("$3^{rd} Coefficient$", fontsize=17)
	plt.scatter(JK, third, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(JK, third, yerr=third_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylim(-4500, 0)
	plt.xlim(.8,3)
	   
	''' 2ND '''

	plt.subplot(323)		
	plt.scatter(JK, second, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(JK, second, yerr=second_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylim(50, 650)
	plt.xlim(.8,3)
	plt.ylabel("$2^{nd} Coefficient$", fontsize=17)
	
	''' 1ST '''

	plt.subplot(324)
	plt.xlabel("$J-K$", fontsize=13)
	
	plt.scatter(JK, first, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(JK, first, yerr=first_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylim(-25, 0)
	plt.xlim(.8,3)
	plt.ylabel("$1^{st} Coefficient$", fontsize=17)
	
	''' 0TH '''

	plt.subplot(325)
	plt.xlabel("$J-K$", fontsize=13)
	plt.ylabel("$0^{th} Coefficient$", fontsize=17)
	
	plt.scatter(JK, zero, s=70, marker='o', c=categories, alpha=0.7, label=types)
	plt.errorbar(JK, zero, yerr=zero_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylim(0.65, 1.25)
	plt.xlim(.8,3)
	
	# Create fake data (void) to use the linestyles for the legend of the sigmas:
	data1 = mlines.Line2D([], [], color='r', marker='o', label="Red", linestyle='', markersize=9)
	data2 = mlines.Line2D([], [], color='m', marker='o', label="Young", linestyle='', markersize=9)
	data3 = mlines.Line2D([], [], color='b', marker='o', label="Blue", linestyle='', markersize=9)
	data4 = mlines.Line2D([], [], color='k', marker='o', label="Field Gravity", linestyle='', markersize=9)

	plt.legend((data1, data2, data4, data3), ("Red", "Young", "Field Gravity", "Blue"), loc=1, numpoints=1)
	plt.show()
	