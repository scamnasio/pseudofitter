''' 
Written by Sara Camnasio
sara.camnasio@gmail.com

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

# Importing results from table
source = np.genfromtxt('/Users/saracamnasio/Dropbox/Research/Projects/UnusuallyRB/2016_Analysis/plotting/CP_results.csv',delimiter=',', skip_header=1, dtype = float)
source1 = np.genfromtxt('/Users/saracamnasio/Dropbox/Research/Projects/UnusuallyRB/2016_Analysis/plotting/CP_results.csv',delimiter=',', skip_header=1, dtype = str)

# Naming values for easier plotting adjustments:
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
#color_value = source1[:,24]
marker_value = source1[:,25]
opt_spt = source[:,2]
opt_spt_unc = source[:,3]
marker_size = source[:,27]

#toggle this to see double objects:
color_value = source1[:,26]

#testing hex colors:
#color_value = ["#0080ff", "k", "k", "#b30000", "#b30000", "#b30000", "#b30000", "grey", "#0080ff", "k", "k", "#b30000", "grey", "#0080ff", "#0080ff", "#0080ff", "#0080ff", "#0080ff", "#b30000", "#0080ff", "#0000b3", "#0080ff", "grey", "grey", "grey", "grey", "#0080ff", "#0080ff", "#0080ff", "#0080ff", "#0080ff", "#b30000", "#0000b3", "#0080ff", "#ff5600", "#ff5600", "#ff5600", "#0080ff", "k", "k", "k", "#b30000", "#b30000", "#b30000", "#b30000", "#0080ff", "grey", "grey", "#b30000", "#b30000", "#b30000", "#b30000", "#b30000", "k", "k", "k", "#0000b3", "#b30000", "#b30000"]

def plots():

	''' Plots results from csv table into 4 figures, 1x2 subplot each'''

	sns.reset_orig()

	data1 = mlines.Line2D([], [], color='grey', marker='*', label="Field", linestyle='', markersize=12)
	data2 = mlines.Line2D([], [], color='k', marker='*', label="Standard", linestyle='', markersize=12)
	data3 = mlines.Line2D([], [], color='#0000b3', marker='o', label="Extremely Blue $(\Delta J-K_{s})\geq 2 \sigma$", linestyle='', markersize=8)
	data4 = mlines.Line2D([], [], color='#0080ff', marker='o', label="Bluer than avg $(\Delta J-K_{s})< 2 \sigma$", linestyle='', markersize=8)
	data5 = mlines.Line2D([], [], color='#b30000', marker='o', label="Extremely Red $(\Delta J-K_{s})\geq 2 \sigma$", linestyle='', markersize=8)
	data6 = mlines.Line2D([], [], color='#ff5600', marker='o', label="Redder than avg $(\Delta J-K_{s})< 2 \sigma$", linestyle='', markersize=8)

	#LMIN/LMAX vs J-K
	plt.figure(figsize=(15, 9)) 
	for n in range(len(names)):
		
		plt.subplots_adjust(hspace=0.001)
		ax1 = plt.subplot(211)
		plt.errorbar(JK_dev[n], lmin[n], xerr=JK_dev_unc[n], yerr=lmin_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor='k', elinewidth=2)
		plt.scatter(JK_dev[n], lmin[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n])
		plt.xlabel("$J-K-(J-K_{s})_{avg}$")
		plt.ylabel("Local Mininimum ($\lambda$)")
		plt.legend((data1, data2, data3, data4, data5, data6), ("Field", "Standard", "Extremely Blue $(\Delta J-K_{s})\geq 2 \sigma$", "Bluer than avg $(\Delta J-K_{s})< 2 \sigma$", "Extremely Red $(\Delta J-K_{s})\geq 2 \sigma$", "Redder than avg $(\Delta J-K_{s})< 2 \sigma$"), fontsize= 9,  loc=3, numpoints=1) #bbox_to_anchor=(.9, 1.25)

	
		ax2 = plt.subplot(212, sharex=ax1)
		plt.errorbar(JK_dev[n], lmax[n], xerr=JK_dev_unc[n], yerr=lmax_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor='k', elinewidth=2)
		plt.scatter(JK_dev[n], lmax[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n])
		plt.xlabel("$J-K-(J-K_{s})_{avg}$")
		plt.ylabel("Local Maximum ($\lambda$)")
		plt.xlim(-1,.8)
		plt.ylim(1.23,1.33)

		plt.setp(ax1.get_xticklabels(), visible=False)
		
	#LMIN/LMAX vs H-K
	plt.figure(figsize=(15, 9))
	for n in range(len(names)):
		plt.subplots_adjust(hspace=0.001)
		ax3 = plt.subplot(211)
		plt.errorbar(HK_dev[n], lmin[n], xerr=HK_dev_unc[n], yerr=lmin_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor='k', elinewidth=2, zorder=-1)
		plt.scatter(HK_dev[n], lmin[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("$H-K-(H-K_{s})_{avg}$")
		plt.ylabel("Local Mininimum ($\lambda$)")
		plt.ylim(1.145,1.195)
		
	
		ax4 = plt.subplot(212, sharex=ax3)
		plt.errorbar(HK_dev[n], lmax[n], xerr=HK_dev_unc[n], yerr=lmax_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor='k', elinewidth=2, zorder=-1)
		plt.scatter(HK_dev[n], lmax[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("$H-K-(H-K_{s})_{avg}$")
		plt.ylabel("Local Maximum ($\lambda$)")
		plt.ylim(1.24,1.315)
		plt.xlim(-.4,.7)
		plt.legend((data1, data2, data3, data4, data5, data6), ("Field", "Standard", "Extremely Blue $(\Delta J-K_{s})\geq 2 \sigma$", "Bluer than avg $(\Delta J-K_{s})< 2 \sigma$", "Extremely Red $(\Delta J-K_{s})\geq 2 \sigma$", "Redder than avg $(\Delta J-K_{s})< 2 \sigma$"), fontsize= 9,  loc=3, numpoints=1)
		plt.setp(ax3.get_xticklabels(), visible=False)
		
	#LMIN/LMAX vs J-H
	plt.figure(figsize=(15, 9))
	for n in range(len(names)):
		plt.subplots_adjust(hspace=0.001)
		ax5 = plt.subplot(211)
		plt.errorbar(JH_dev[n], lmin[n], xerr=JH_dev_unc[n], yerr=lmin_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor='k', elinewidth=2, zorder=-1)
		plt.scatter(JH_dev[n], lmin[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("$J-H-(J-H)_{avg}$")
		plt.ylabel("Local Mininimum ($\lambda$)")
		plt.ylim(1.145,1.195)
		plt.legend((data1, data2, data3, data4, data5, data6), ("Field", "Standard", "Extremely Blue $(\Delta J-K_{s})\geq 2 \sigma$", "Bluer than avg $(\Delta J-K_{s})< 2 \sigma$", "Extremely Red $(\Delta J-K_{s})\geq 2 \sigma$", "Redder than avg $(\Delta J-K_{s})< 2 \sigma$"), fontsize= 9,  loc=4, numpoints=1)

	
		ax6 = plt.subplot(212, sharex=ax5)
		plt.errorbar(JH_dev[n], lmax[n], xerr=JH_dev_unc[n], yerr=lmax_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor='k', elinewidth=2, zorder=-1)
		plt.scatter(JH_dev[n], lmax[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("$J-H-(J-H)_{avg}$")
		plt.ylabel("Local Maximum ($\lambda$)")
		plt.ylim(1.24,1.325)
		plt.xlim(-1,1.5)
		
		plt.setp(ax5.get_xticklabels(), visible=False)
		
	plt.figure(figsize=(15, 9))
	for n in range(len(names)):
		plt.subplots_adjust(hspace=0.001)
		P1 = plt.subplot(211)
		plt.errorbar(opt_spt[n], lmin[n], yerr=lmin_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor='k', elinewidth=2, zorder=-1)
		plt.scatter(opt_spt[n], lmin[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("Spectral Type")
		plt.ylabel("Local Mininimum ($\lambda$)")
		plt.xticks(np.arange(9,20,1))
		labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
		P1.set_xticklabels(labels)
		plt.ylim(1.14,1.2)
		plt.legend((data1, data2, data3, data4, data5, data6), ("Field", "Standard", "Extremely Blue $(\Delta J-K_{s})\geq 2 \sigma$", "Bluer than avg $(\Delta J-K_{s})< 2 \sigma$", "Extremely Red $(\Delta J-K_{s})\geq 2 \sigma$", "Redder than avg $(\Delta J-K_{s})< 2 \sigma$"), fontsize= 9,  loc=3, numpoints=1)
	
	
	
		P2 = plt.subplot(212, sharex=P1)
		plt.errorbar(opt_spt[n], lmax[n], yerr=lmax_unc[n], fmt='none', alpha=0.5, linestyle='None', ecolor='k', elinewidth=2, zorder=-1)
		plt.scatter(opt_spt[n], lmax[n], alpha = 0.9, s=marker_size[n], c=color_value[n], marker=marker_value[n], zorder=1)
		plt.xlabel("Spectral Type")
		plt.ylabel("Local Maximum ($\lambda$)")
		#plt.xticks(np.arange(9,20,1))
		labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']
		P2.set_xticklabels(labels)
		plt.ylim(1.23,1.325)
		plt.setp(P1.get_xticklabels(), visible=False)
		
def sea():

	source_data = pd.read_csv('/Users/saracamnasio/Dropbox/Research/Projects/UnusuallyRB/2016_Analysis/plotting/CP_results.csv')
	sns.pairplot(source_data, hue="Classification", vars=["Opt_SpT", "Lmin_mu"], palette="Set2", diag_kind="kde", size=2.5)
	sns.pairplot(source_data, hue="Classification", vars=["Opt_SpT", "Lmax_mu"], palette="Set2", diag_kind="kde", size=2.5)

	sns.pairplot(source_data, hue="Classification", vars=["JK_Dev", "Lmin_mu"], palette="Set2", diag_kind="kde", size=2.5)
	sns.pairplot(source_data, hue="Classification", vars=["HK_Dev", "Lmin_mu"], palette="Set2", diag_kind="kde", size=2.5)
	sns.pairplot(source_data, hue="Classification", vars=["JH_Dev", "Lmin_mu"], palette="Set2", diag_kind="kde", size=2.5)

	sns.pairplot(source_data, hue="Classification", vars=["JK_Dev", "Lmax_mu"], palette="Set2", diag_kind="kde", size=2.5)
	sns.pairplot(source_data, hue="Classification", vars=["HK_Dev", "Lmax_mu"], palette="Set2", diag_kind="kde", size=2.5)
	sns.pairplot(source_data, hue="Classification", vars=["JH_Dev", "Lmax_mu"], palette="Set2", diag_kind="kde", size=2.5)

	# sns.jointplot(x="Opt_SpT", y="Lmin_mu", hue="Classification", data=tips2, kind="kde")
	
