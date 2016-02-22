import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from matplotlib import axis
from matplotlib.pyplot import step, legend, xlim, ylim, show
#from matplotlib.pyplot import *

red = np.genfromtxt('/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_data/Red_new.csv',delimiter=',', dtype = str)
young = np.genfromtxt('/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_data/Young_new.csv',delimiter=',', dtype = str)
blue = np.genfromtxt('/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_data/Blue_new.csv',delimiter=',', dtype = str)
field = np.genfromtxt('/Users/saracamnasio/Research/Projects/UnusuallyRB/Source_data/Field.csv',delimiter=',', dtype = str)


#red

namer = np.array([row[0] for row in red])
ftr = 	np.array([float(row[1]) for row in red])
ftr_unc = 
thr = 	np.array([float(row[2]) for row in red])
twr = 	np.array([float(row[3]) for row in red]) 
oner = 	np.array([float(row[4]) for row in red])
zr = 	np.array([float(row[5]) for row in red])
sptr = 	np.array([float(row[6]) for row in red])
JKr = 	np.array([float(row[7]) for row in red])


#young

namey = [row[0] for row in young]
fty = 	np.array([float(row[1]) for row in young])
thy = 	np.array([float(row[2]) for row in young])
twy = 	np.array([float(row[3]) for row in young])
oney = 	np.array([float(row[4]) for row in young])
zy = 	np.array([float(row[5]) for row in young])
spty = 	np.array([float(row[6]) for row in young])
JKy = 	np.array([float(row[7]) for row in young])



#blue

nameb = [row[0] for row in blue]
ftb = 	np.array([float(row[1]) for row in blue])
thb = 	np.array([float(row[2]) for row in blue])
twb = 	np.array([float(row[3]) for row in blue])
oneb = 	np.array([float(row[4]) for row in blue])
zb = 	np.array([float(row[5]) for row in blue])
sptb = 	np.array([float(row[6]) for row in blue])
JKb = 	np.array([float(row[7]) for row in blue])

#field

namef = [row[0] for row in field]
ftf = 	np.array([float(row[1]) for row in field])
thf = 	np.array([float(row[2]) for row in field])
twf = 	np.array([float(row[3]) for row in field])
onef = 	np.array([float(row[4]) for row in field])
zf = 	np.array([float(row[5]) for row in field])
sptf = 	np.array([float(row[6]) for row in field])
JKf = 	np.array([float(row[7]) for row in field])

names = [namey,nameb, namer, namef]

# def myplot(var1, var2, marker, linestyle, color, label, alpha):
# 
  	# unique = []
  	# for i in names:
  		# if i not in unique: unique.append(i)
  		# for l in unique:
  			# ind = np.where(names == l)
  			# plt.plot(var1[ind], var2[ind], marker=marker, linestyle=linestyle, color=color, label=label, alpha=alpha)
				
# subsitute plt.plot for myplot
def fits():
	# spec_type_fit = np.concatenate((sptr, spty, sptb, sptf))
	# JK_fit = np.concatenate((JKr, JKy, JKb, JKf))
	# fourth_coeff = np.concatenate((ftr, fty, ftb, ftf))
	# third_coeff = np.concatenate((thr, thy, thb, thf))
	# second_coeff = np.concatenate((twr, twy, twb, twf))
	# first_coeff = np.concatenate((oner, oney, oneb, onef))
	# zeroeth_coeff = np.concatenate((zr, zy, zb, zf))
	
	color = '#F26D00'
	
	
	plt.figure()
	plt.subplot(321)
	plt.title("Spectral Type vs Fit Coefficients", fontsize=13)
	#plt.errorbar(sptss, JKss, yerr=errss, fmt=None, ecolor='b')
	# plt.xlabel("Spectral Type", fontsize=13)
	plt.plot(sptr, np.array([float(row[1]) for row in red]), marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(spty, fty, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(sptb, ftb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarfs')
	plt.plot(sptf, ftf, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit1 = scipy.stats.pearsonr(spec_type_fit, fourth_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit1), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	plt.ylabel("4th Coefficient", fontsize=17)
	# xlim(9.5, 20)
	# ylim(-11, 8)
	
	#3rd
	
	plt.subplot(322)
	#plt.errorbar(sptss, JKss, yerr=errss, fmt=None, ecolor='b')
	# plt.xlabel("Spectral Type", fontsize=13)
	plt.plot(sptr, thr, marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(spty, thy, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(sptb, thb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarfs')
	plt.plot(sptf, thf, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit2 = scipy.stats.pearsonr(spec_type_fit, third_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit2), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	# plt.legend(loc=3)
	plt.ylabel("3rd Coefficient", fontsize=17)
	# xlim(9.5, 20)
	# ylim(-11, 8)
	   
	#2nd    
	plt.subplot(323)
	# plt.xlabel("Spectral Type", fontsize=13)
	plt.plot(sptr, twr, marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(spty, twy, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(sptb, twb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarfs')
	plt.plot(sptf, twf, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit3 = scipy.stats.pearsonr(spec_type_fit, second_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit3), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	plt.ylabel("2nd Coefficient", fontsize=17)
	# xlim(9.5, 20)
	#ylim(-1.1, 1)
	
	#1st
	plt.subplot(324)
	#plt.title("J-K deviation vs SpT")
	plt.xlabel("Spectral Type", fontsize=13)
	plt.plot(sptr, oner, marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(spty, oney, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(sptb, oneb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarf')
	plt.plot(sptf, onef, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit4 = scipy.stats.pearsonr(spec_type_fit, first_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit4), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	plt.ylabel("1st Coefficient", fontsize=17)
	
	# xlim(9.5, 20)
	#ylim(-1.1, 1)
	
	#0th
	plt.subplot(325)
	#plt.title("J-K deviation vs SpT")
	plt.xlabel("Spectral Type", fontsize=13)
	plt.plot(sptr, zr, marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(spty, zy, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(sptb, zb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarf')
	plt.plot(sptf, zf, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit5 = scipy.stats.pearsonr(spec_type_fit, zeroeth_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit5), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	plt.ylabel("0th Coefficient", fontsize=17)
	plt.legend(loc=1, fontsize=10)
	# xlim(9.5, 20)
	#ylim(0, 3.5)
	
	plt.show()
	
	# color vs coefficie
	
	#4th
	plt.figure()
	plt.subplot(321)
	plt.title("J-K Color vs Fit Coefficients", fontsize=13)
	#plt.errorbar(sptss, JKss, yerr=errss, fmt=None, ecolor='b')
	# plt.xlabel("J-K", fontsize=13)
	plt.plot(JKr, np.array([float(row[1]) for row in red]), marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(JKy, fty, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(JKb, ftb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarfs')
	plt.plot(JKf, ftf, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit6 = scipy.stats.pearsonr(JK_fit, fourth_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit6), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	plt.ylabel("4th Coefficient", fontsize=17)
	# xlim(9.5, 20)
	# ylim(-11, 8)
	
	#3rd
	
	plt.subplot(322)
	#plt.errorbar(sptss, JKss, yerr=errss, fmt=None, ecolor='b')
	# plt.xlabel("J-K", fontsize=13)
	plt.plot(JKr, thr, marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(JKy, thy, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(JKb, thb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarf')
	plt.plot(JKf, thf, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit7 = scipy.stats.pearsonr(JK_fit, third_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit7), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	# plt.legend(loc=3)
	plt.ylabel("3rd Coefficient", fontsize=17)
	# xlim(.9, 2.5)
	# ylim(-11, 8)
	   
	#2nd    
	plt.subplot(323)
	# plt.xlabel("J-K", fontsize=13)
	plt.plot(JKr, twr, marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(JKy, twy, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(JKb, twb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarf')
	plt.plot(JKf, twf, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit8 = scipy.stats.pearsonr(JK_fit, second_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit8), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	plt.ylabel("2nd Coefficient", fontsize=17)
	# xlim(.9, 2.5)
	#ylim(-1.1, 1)
	
	#1st
	plt.subplot(324)
	#plt.title("J-K deviation vs SpT")
	plt.xlabel("J-K", fontsize=13)
	plt.plot(JKr, oner, marker='o', linestyle='None', color=color, alpha=0.7,  label='Red Non-Young')
	plt.plot(JKy, oney, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(JKb, oneb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarf')
	plt.plot(JKf, onef, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit9 = scipy.stats.pearsonr(JK_fit, first_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit9), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	plt.ylabel("1st Coefficient", fontsize=17)
	
	# xlim(.9, 2.5)
	#xlim(-1.1, 1)
	
	#0th
	plt.subplot(325)
	#plt.title("J-K deviation vs SpT")
	plt.xlabel("J-K", fontsize=13)
	plt.plot(JKr, zr, marker='o', linestyle='None', color=color, alpha=0.7, label='Red Non-Young')
	plt.plot(JKy, zy, marker='o', linestyle='None', color='r', alpha=0.7, label='Red Young')
	plt.plot(JKb, zb, marker='o', linestyle='None', color='b',alpha=0.7, label='Blue Non-Subdwarf')
	plt.plot(JKf, zf, marker='*', linestyle='None', color='k',alpha=0.7, label='Field Gravity Objects')
	# linear_fit10 = scipy.stats.pearsonr(JK_fit, zeroeth_coeff)
	# plt.annotate('Linear Fit Index: {0}'.format(linear_fit10), xy=(1.225, 1), xytext=(1.225, 1), color='black', weight='semibold', fontsize=8)
	plt.ylabel("0th Coefficient", fontsize=17)
	plt.legend(loc=1, fontsize=10)
	
	# xlim(.9, 2.5)
	#xlim(0, 3.5)
	
	
	plt.show()
	