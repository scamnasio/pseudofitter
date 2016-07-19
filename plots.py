''' 
Written by Sara Camnasio
CUNY Hunter College class of 2016
sara.camnasio@gmail.com

Current to do list 04/20/2016:
- Add uncertainties 
'''

import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn.apionly as sns
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
sns.set(color_codes=True)



# Grabbing the coefficients from the analysis results table
source = np.genfromtxt('/Users/saracamnasio/Dropbox/Research/Projects/UnusuallyRB/2016_Analysis/plotting/CP_results.csv',delimiter=',', skip_header=1, dtype = float)
source1 = np.genfromtxt('/Users/saracamnasio/Dropbox/Research/Projects/UnusuallyRB/2016_Analysis/plotting/CP_results.csv',delimiter=',', skip_header=1, dtype = str)

# Importing the names from the same table, same order, just as strings
# source_name = np.genfromtxt('/Users/saracamnasio/Desktop/Master_Sheet.csv',delimiter=',', skip_header=1, dtype = str)

# Grabbing only the coefficients from the table
names = source1[:,0]
IP = source[:,18]
IP_unc = source[:,19]
lmax = source[:,20]
lmax_unc = source[:,21]
lmin = source[:,22]
lmin_unc = source[:,23]
obj_type = source1[:,17]
JK = source[:,5]
JK_unc = source[:,6]
HK = source[:,7]
HK_unc = source[:,8]
JH = source[:,9]
JH_unc = source[:,10]
JK_dev = source[:,11]
JK_dev_unc = source[:,12]
HK_dev = source[:,13]
HK_dev_unc = source[:,14]
JH_dev = source[:,15]
JH_dev_unc = source[:,16]
color_value = source1[:,24]
marker_value = source1[:,25]
opt_spt = source[:,2]
opt_spt_unc = source[:,3]
marker_size = source[:,27]

#toggle this to see double objects:
#color_value = source1[:,26]

def plots():


	''' This function outputs local max, min, and inflection points in terms of wavelength '''

	sns.reset_orig()

	data1 = mlines.Line2D([], [], color='r', marker='o', label="Non-Young", linestyle='', markersize=8)
	data2 = mlines.Line2D([], [], color='r', marker='^', label="Young", linestyle='', markersize=8)
	data3 = mlines.Line2D([], [], color='k', marker='*', label="Standard", linestyle='', markersize=12)
	data6 = mlines.Line2D([], [], color='grey', marker='*', label="Field", linestyle='', markersize=12)
	data4 = mlines.Line2D([], [], color='b', marker='^', label="Subdwarf", linestyle='', markersize=8)
	data5 = mlines.Line2D([], [], color='b', marker='o', label="Non-Subdwarf", linestyle='', markersize=8)

	#LMIN/LMAX vs J-K
	plt.figure() 
	for n in range(len(names)):

		plt.subplots_adjust(hspace=0.001)
		ax1 = plt.subplot(211)
		plt.errorbar(JK[n], lmin[n], xerr=JK_unc[n], yerr=lmin_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor=color_value[n], elinewidth=2)
		plt.scatter(JK[n], lmin[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n])
		plt.xlabel("$J-K_{s}$ Color")
		plt.ylabel("Local Mininimum ($\lambda$)")
		plt.xlim(0.7,3.2)
	
	
		ax2 = plt.subplot(212, sharex=ax1)
		plt.errorbar(JK[n], lmax[n], xerr=JK_unc[n], yerr=lmax_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor=color_value[n], elinewidth=2)
		plt.scatter(JK[n], lmax[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n])
		plt.xlabel("$J-K_{s}$ Color")
		plt.ylabel("Local Maximum ($\lambda$)")
		plt.xlim(.7,3.2)
		plt.ylim(1.23,1.33)

		plt.setp(ax1.get_xticklabels(), visible=False)
		plt.legend((data6, data3, data5, data4, data1, data2), ("Field", "Standard", "Non-Subdwarf", "Subdwarf", "Non-Young", "Young"), fontsize= 13,  loc=2, numpoints=1, bbox_to_anchor=(.9, 1.25))

	#LMIN/LMAX vs H-K
	plt.figure()
	for n in range(len(names)):
		plt.subplots_adjust(hspace=0.001)
		ax3 = plt.subplot(211)
		plt.errorbar(HK[n], lmin[n], xerr=HK_unc[n], yerr=lmin_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor=color_value[n], elinewidth=2, zorder=-1)
		plt.scatter(HK[n], lmin[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("$H-K_{s}$ Color")
		plt.ylabel("Local Mininimum ($\lambda$)")
		plt.ylim(1.145,1.195)
	
	
		ax4 = plt.subplot(212, sharex=ax3)
		plt.errorbar(HK[n], lmax[n], xerr=HK_unc[n], yerr=lmax_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor=color_value[n], elinewidth=2, zorder=-1)
		plt.scatter(HK[n], lmax[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("$H-K_{s}$ Color")
		plt.ylabel("Local Maximum ($\lambda$)")
		plt.ylim(1.24,1.315)
		plt.xlim(.3,1.4)
		
		plt.setp(ax3.get_xticklabels(), visible=False)
		plt.legend((data6, data3, data5, data4, data1, data2), ("Field", "Standard", "Non-Subdwarf", "Subdwarf", "Non-Young", "Young"), fontsize= 13,  loc=2, numpoints=1, bbox_to_anchor=(.9, 1.25))

	#LMIN/LMAX vs J-H
	plt.figure()
	for n in range(len(names)):
		plt.subplots_adjust(hspace=0.001)
		ax5 = plt.subplot(211)
		plt.errorbar(JH[n], lmin[n], xerr=JH_unc[n], yerr=lmin_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor=color_value[n], elinewidth=2, zorder=-1)
		plt.scatter(JH[n], lmin[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("$J-H_{s}$ Color")
		plt.ylabel("Local Mininimum ($\lambda$)")
		plt.ylim(1.145,1.195)
	
	
		ax6 = plt.subplot(212, sharex=ax5)
		plt.errorbar(JH[n], lmax[n], xerr=JH_unc[n], yerr=lmax_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor=color_value[n], elinewidth=2, zorder=-1)
		plt.scatter(JH[n], lmax[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("$J-H_{s}$ Color")
		plt.ylabel("Local Maximum ($\lambda$)")
		plt.ylim(1.24,1.325)
		plt.xlim(.2,1.9)
		
		plt.setp(ax5.get_xticklabels(), visible=False)
		plt.legend((data6, data3, data5, data4, data1, data2), ("Field", "Standard", "Non-Subdwarf", "Subdwarf", "Non-Young", "Young"), fontsize= 13,  loc=2, numpoints=1, bbox_to_anchor=(.9, 1.25))

	plt.figure()
	for n in range(len(names)):
		plt.subplots_adjust(hspace=0.001)
		P1 = plt.subplot(211)
		plt.errorbar(opt_spt[n], lmin[n], yerr=lmin_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor=color_value[n], elinewidth=2, zorder=-1)
		plt.scatter(opt_spt[n], lmin[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("Spectral Type")
		plt.ylabel("Local Mininimum ($\lambda$)")
		plt.xticks(np.arange(9,20,1))
		labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
		P1.set_xticklabels(labels)
		plt.ylim(1.14,1.2)
	
	
		P2 = plt.subplot(212, sharex=P1)
		plt.errorbar(opt_spt[n], lmax[n], yerr=lmax_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor=color_value[n], elinewidth=2, zorder=-1)
		plt.scatter(opt_spt[n], lmax[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("Spectral Type")
		plt.ylabel("Local Maximum ($\lambda$)")
		plt.xticks(np.arange(9,20,1))
		labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
		P2.set_xticklabels(labels)
		plt.ylim(1.23,1.325)
		
		plt.setp(P1.get_xticklabels(), visible=False)
		plt.legend((data6, data3, data5, data4, data1, data2), ("Field", "Standard", "Non-Subdwarf", "Subdwarf", "Non-Young", "Young"), fontsize= 13,  loc=2, numpoints=1, bbox_to_anchor=(.9, 1.25))
	
def sea():
	red = []
	blue = []

	for n in range(len(names)):
		if obj_type[n] == 'red' or 'young':
			red.append(source[n,:])

		elif obj_type[n] == 'blue':
			blue.append(source[n,:])
		else:
			pass
	


	plt.figure()
	sns.jointplot(red[:,5], red[:,20], kind="reg", color="#cc3300")
	sns.jointplot(blue[:,5], blue[:,20], kind="reg", color="#cc3300")

	# fig.set(xlabel='J-K Color', ylabel='Local maximum')
	plt.xlabel('J-K Color')
	plt.ylabel('Local Max')
	
	# sns.jointplot(x=JH, y=lmin, kind="reg")
	# plt.xlabel('J-H Color')
	# plt.ylabel('Local Min')
	# 
# 
	# sns.jointplot(x=JH, y=lmax, kind="reg")
	# plt.xlabel('J-H Color')
	# plt.ylabel('Local Max')
	# 
# 
	# sns.jointplot(x=HK, y=lmin, kind="reg")
	# plt.xlabel('H-K Color')
	# plt.ylabel('Local Min')
	# 
# 
	# sns.jointplot(x=HK, y=lmax, kind="reg")
	# plt.xlabel('H-K Color')
	# plt.ylabel('Local Max')
# 
# 
	# sns.jointplot(x=JK_dev, y=lmin, kind="reg")
	# plt.xlabel('J-K Dev')
	# plt.ylabel('Local Min')
	# 
# 
	# sns.jointplot(x=JK_dev, y=lmax, kind="reg")
	# plt.xlabel('J-K Dev')
	# plt.ylabel('Local Max')
# 
# 
	# sns.jointplot(x=opt_spt, y=lmin, kind="reg")
	# plt.xlabel('Optical Spt')
	# plt.ylabel('Local Min')
	# 
# 
	# sns.jointplot(x=opt_spt, y=lmax, kind="reg")
	# plt.xlabel('Opt spt')
	# plt.ylabel('Local Max')
	# 
# 



	# np.savetxt()

	# to use when saving plotting result or plotting table:
	#date_format = time.strftime("%d/%m/%Y")

