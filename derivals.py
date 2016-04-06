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
JK_err = np.array(source[:,18])

def fits():

	categories = []
	for n in types:
		if n == 'young':
			categories.append("r")
		elif n == "blue":
			categories.append("b")
		elif n == "red":
			categories.append("r")
		elif n == "standard":
			categories.append("k")
		elif n == "subdwarf":
			categories.append("b")

	categories = np.array(categories)

	''' Spectral type VS Coeff '''
	
	''' 4th '''
	plt.figure()
	P = plt.subplot(321)
	plt.ylabel("$4^{th} Coefficient$", fontsize=17)
	plt.scatter(spt, fourth, s=70, marker='o', c=categories, alpha=0.7, label=types)
	par = np.poly1d(np.polyfit(spt, fourth, 1))
	ys = par(spt)
	plt.plot(spt, ys, color='green')
	plt.errorbar(spt, fourth, elinewidth=2, markeredgecolor='None', xerr=0.5, yerr=fourth_err, fmt=None, alpha=0.5, ecolor='k')
	plt.xlim(9,20)
	plt.ylim(-100,10500)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P.set_xticklabels(labels)

	''' 3rd '''
	
	P2 = plt.subplot(322)
	plt.scatter(spt, third, s=70, marker='o', c=categories, alpha=0.7, label=types)
	par = np.poly1d(np.polyfit(spt, third, 1))
	ys = par(spt)
	plt.plot(spt, ys, color='green')
	plt.errorbar(spt, third, elinewidth=2, markeredgecolor='None', xerr=0.5, yerr=third_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylabel("$3^{rd} Coefficient$", fontsize=17)
	plt.xlim(9,20)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P2.set_xticklabels(labels)
	   
	''' 2ND '''

	P3 = plt.subplot(323)
	plt.ylabel("$2^{nd} Coefficient$", fontsize=17)
	plt.scatter(spt, second, s=70, marker='o', c=categories, alpha=0.7, label=types)
	par = np.poly1d(np.polyfit(spt, second, 1))
	ys = par(spt)
	plt.plot(spt, ys, color='green')
	plt.errorbar(spt, second, elinewidth=2, markeredgecolor='None', xerr=0.5, yerr=second_err, fmt=None, alpha=0.5, ecolor='k')
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
	par = np.poly1d(np.polyfit(spt, first, 1))
	ys = par(spt)
	plt.plot(spt, ys, color='green')
	plt.errorbar(spt, first, elinewidth=2, markeredgecolor='None', xerr=0.5, yerr=first_err, fmt=None, alpha=0.5, ecolor='k')
	plt.xlim(9,20)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P4.set_xticklabels(labels)
	
	''' 0TH '''
	P5 = plt.subplot(325)
	#plt.title("J-K deviation vs SpT")
	plt.xlabel("$Spectral Type$", fontsize=13)
	plt.scatter(spt, zero, s=70,  marker='o', c=categories, alpha=0.7, label=types)
	par = np.poly1d(np.polyfit(spt, zero, 1))
	ys = par(spt)
	plt.plot(spt, ys, color='green')
	plt.errorbar(spt, zero, elinewidth=2, markeredgecolor='None', xerr=0.5, yerr=zero_err, fmt=None, alpha=0.5, ecolor='k')
	plt.ylabel("$0^{th} Coefficient$", fontsize=17)
	plt.xlim(9,20)
	plt.xticks(np.arange(9,20,1))
	labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
	P5.set_xticklabels(labels)
	# plt.legend(loc=1, fontsize=10)
	
	# Create fake data (void) to use the linestyles for the legend of the sigmas:
	data1 = mlines.Line2D([], [], color='0.5', marker='o', markeredgecolor='None', label="Non-Young", linestyle='', markersize=10)
	data2 = mlines.Line2D([], [], color='r', marker='^', markeredgecolor='None', label="Young", linestyle='', markersize=10)
	data3 = mlines.Line2D([], [], color='k', marker='*', markeredgecolor='None', label="Standard", linestyle='', markersize=14)
	data4 = mlines.Line2D([], [], color='b', marker='^', markeredgecolor='None', label="Subdwarf", linestyle='', markersize=10)
	data5 = mlines.Line2D([], [], color='b', marker='o', markeredgecolor='None', label="UBL", linestyle='', markersize=10)
	plt.legend((data1, data2, data3, data4, data5), ("Non-Young", "Young", "Standard", "Subdwarf", "UBL"), loc=3, bbox_to_anchor=(1.45, 0),  frameon=False, numpoints=1)
	
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
	par = np.poly1d(np.polyfit(JK, fourth, 1))
	ys = par(JK)
	plt.plot(JK, ys, color='green')
	plt.errorbar(JK, fourth, elinewidth=2, markeredgecolor='None', xerr=JK_err , yerr=fourth_err, fmt=None, alpha=0.5, ecolor='k')
	# plt.ylim(-750, 10500)
	# plt.xlim(.5, 3)
	
	''' 3RD '''
	
	plt.subplot(322)
	plt.ylabel("$3^{rd} Coefficient$", fontsize=17)
	plt.scatter(JK, third, s=70, marker='o', c=categories, alpha=0.7, label=types)
	par = np.poly1d(np.polyfit(JK, third, 1))
	ys = par(JK)
	plt.plot(JK, ys, color='green')
	plt.errorbar(JK, third, elinewidth=2, markeredgecolor='None', xerr=JK_err, yerr=third_err, fmt=None, alpha=0.5, ecolor='k')
	# plt.ylim(-4500, 0)
	# plt.xlim(.8,3)
	   
	''' 2ND '''

	plt.subplot(323)		
	plt.scatter(JK, second, s=70,  marker='o', c=categories, alpha=0.7, label=types)
	par = np.poly1d(np.polyfit(JK, second, 1))
	ys = par(JK)
	plt.plot(JK, ys, color='green')
	plt.errorbar(JK, second, elinewidth=2, markeredgecolor='None', xerr=JK_err, yerr=second_err, fmt=None, alpha=0.5, ecolor='k')
	# plt.ylim(50, 650)
	# plt.xlim(.8,3)
	plt.ylabel("$2^{nd} Coefficient$", fontsize=17)
	
	''' 1ST '''

	plt.subplot(324)
	plt.xlabel("$J-K$", fontsize=13)
	
	plt.scatter(JK, first, s=70,  marker='o', c=categories, alpha=0.7, label=types)
	par = np.poly1d(np.polyfit(JK, first, 1))
	ys = par(JK)
	plt.plot(JK, ys, color='green')
	plt.errorbar(JK, first, elinewidth=2,markeredgecolor='None', xerr=JK_err, yerr=first_err, fmt=None, alpha=0.5, ecolor='k')
	# plt.ylim(-25, 0)
	# plt.xlim(.8,3)
	plt.ylabel("$1^{st} Coefficient$", fontsize=17)
	
	''' 0TH '''

	plt.subplot(325)
	plt.xlabel("$J-K$", fontsize=13)
	plt.ylabel("$0^{th} Coefficient$", fontsize=17)
	
	plt.scatter(JK, zero, s=70,  marker='o', c=categories, alpha=0.7, label=types)
	par = np.poly1d(np.polyfit(JK, zero, 1))
	ys = par(JK)
	plt.plot(JK, ys, color='green')
	plt.errorbar(JK, zero, elinewidth=2, markeredgecolor='None', xerr=JK_err, yerr=zero_err, fmt=None, alpha=0.5, ecolor='k')
	# plt.ylim(0.65, 1.25)
	# plt.xlim(.8,3)
	
	# Create fake data (void) to use the linestyles for the legend of the sigmas:
	data1 = mlines.Line2D([], [], color='0.5', marker='o', markeredgecolor='None', label="Non-Young", linestyle='', markersize=10)
	data2 = mlines.Line2D([], [], color='r', marker='^', markeredgecolor='None', label="Young", linestyle='', markersize=10)
	data3 = mlines.Line2D([], [], color='k', marker='*', markeredgecolor='None', label="Standard", linestyle='', markersize=14)
	data4 = mlines.Line2D([], [], color='b', marker='^', markeredgecolor='None', label="Subdwarf", linestyle='', markersize=10)
	data5 = mlines.Line2D([], [], color='b', marker='o', markeredgecolor='None', label="UBL", linestyle='', markersize=10)
	# data4 = mlines.Line2D([], [], color='k', marker='o', label="Field Gravity", linestyle='', markersize=9)

	plt.legend((data1, data2, data3, data4, data5), ("Non-Young", "Young", "Standard", "Subdwarf", "UBL"), loc=3, bbox_to_anchor=(1.45, 0),  frameon=False, numpoints=1)
	plt.show()

	plt.figure()
	plt.subplot(321)
	plt.acorr(fourth)
	plt.subplot(322)
	plt.acorr(third)
	plt.subplot(323)
	plt.acorr(second)
	plt.subplot(324)
	plt.acorr(first)
	plt.subplot(325)
	plt.acorr(zero)

	# plt.figure()
	# plt.subplot(321)
	# pyplot.acorr(spt)
	# plt.subplot(322)
	# pyplot.acorr(x)
	# plt.subplot(323)
	# pyplot.acorr(x)
	# plt.subplot(324)
	# pyplot.acorr(x)
	# plt.subplot(325)
	# pyplot.acorr(x)

	plt.show()

	