import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
sns.reset_orig()

source = np.genfromtxt('/Users/saracamnasio/Dropbox/Research/Projects/UnusuallyRB/2016_Analysis/input/Final_sample.csv', delimiter=',', skip_header=1, dtype = float)
source1 = np.genfromtxt('/Users/saracamnasio/Dropbox/Research/Projects/UnusuallyRB/2016_Analysis/input/Final_sample.csv', delimiter=',', skip_header=1, dtype = str)
mclean = np.genfromtxt('/Users/saracamnasio/Dropbox/Research/Projects/UnusuallyRB/2016_Analysis/input/Mclean.csv', delimiter=',', skip_header=1, dtype = float)


marker_style = source1[:,6]
marker_size = source[:,7]
spt = source[:,1]
JK_dev=source[:,14] # max min: 1.049, -1.219
JK_dev_unc = source[:,15]
JH_dev = source[:,16]
JH_dev_unc = source[:,17]
HK_dev = source[:,18]
HK_dev_unc = source[:,19]

mean_dev=np.mean(JK_dev)
std=np.std(JK_dev)

FirstEW = source[:,20]
FirstEW_err = source[:,21]
SecondEW = source[:,22]
SecondEW_err = source[:,23]
ThirdEW = source[:,24]
ThirdEW_err = source[:,25]
FourthEW = source[:,26]
FourthEW_err = source[:,27]
fillstyle=source1[:,29]


FirstEW_ML = mclean[:,2]
FirstEW_err_ML = mclean[:,3]
SecondEW_ML = mclean[:,4]
SecondEW_err_ML = mclean[:,5]
ThirdEW_ML = mclean[:,6]
ThirdEW_err_ML = mclean[:,7]
FourthEW_ML = mclean[:,8]
FourthEW_err_ML = mclean[:,9]
spt_ML = mclean[:,0]

markers_array = ['o','*','s', 'D']

data1 = mlines.Line2D([], [], color='white', marker='*', label="Spectrally & Photometrically Peculiar", linestyle='', markersize=12)
data2 = mlines.Line2D([], [], color='white', marker='o', label="Spectrally Peculiar", linestyle='', markersize=8)
data3 = mlines.Line2D([], [], color='white', marker='s', label="Photometrically Peculiar", linestyle='', markersize=8)
data4 = mlines.Line2D([], [], color='white', marker='D', label="Not Peculiar", linestyle='', markersize=8)
data5 = mlines.Line2D([], [], color='grey', marker='^', label="McLean et al. 2003", linestyle='', markersize=8)
#data6 = mlines.Line2D([], [], color='red', marker='o', label="This work", linestyle='', markersize=8)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
data7 = mlines.Line2D([], [], color='red', marker='*', label="Spectrally & Photometrically Peculiar", linestyle='', markersize=12)
data8 = mlines.Line2D([], [], color='red', marker='o', label="Spectrally Peculiar", linestyle='', markersize=8)
data9 = mlines.Line2D([], [], color='red', marker='s', label="Photometrically Peculiar", linestyle='', markersize=8)
data10 = mlines.Line2D([], [], color='red', marker='D', label="Not Peculiar", linestyle='', markersize=8)

labels = ['','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9']


def plot():

	#EW vs SPT:
	
	fig=plt.figure(figsize=(12,8))
	plt.subplots_adjust(hspace=0.001)
	P1=plt.subplot(221)
	plt.ylim(0,12)
	plt.xlim(9.5,19.5)
	plt.ylabel("1.1693 $\mu$m EW")
	plt.xlabel("Spectral Type")
	for n in range(len(spt)):
		if FirstEW[n] != 0: 
			plt.errorbar(spt[n], FirstEW[n], yerr=FirstEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			sc1= plt.scatter(spt[n], FirstEW[n], alpha = 0.9, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	#plt.colorbar(sc1)
	plt.errorbar(spt_ML, FirstEW_ML, yerr=FirstEW_err_ML, fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
	plt.scatter(spt_ML, FirstEW_ML, alpha = 0.4, s=120, c='k',zorder=1, marker= '^')
	plt.xticks(np.arange(9,20,1))
	P1.set_xticklabels(labels)
	plt.setp(P1.get_xticklabels(), visible=False)


	P2=plt.subplot(222, sharex=P1)
	plt.ylim(2,16)
	plt.xlim(9.5,19.5)
	plt.ylabel("1.1773 $\mu$m EW")
	plt.xlabel("Spectral Type")
	
	for n in range(len(spt)):
		if SecondEW[n] != 0: 
			plt.errorbar(spt[n], SecondEW[n], yerr=SecondEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			sc2= plt.scatter(spt[n], SecondEW[n], alpha = 0.9, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	#plt.colorbar(sc2)
	plt.errorbar(spt_ML, SecondEW_ML, yerr=SecondEW_err_ML, fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
	plt.scatter(spt_ML, SecondEW_ML, alpha = 0.4, s=120, c='k',zorder=1, marker= '^')
	plt.xticks(np.arange(9,20,1))
	P2.set_xticklabels(labels)
	plt.setp(P2.get_xticklabels(), visible=False)


	P3=plt.subplot(223, sharex=P1)
	plt.ylim(0,10)
	plt.xlim(9.5,19.5)
	plt.ylabel("1.2436 $\mu$m EW")
	plt.xlabel("Spectral Type")
	for n in range(len(spt)):
		if ThirdEW[n] != 0: 
			plt.errorbar(spt[n], ThirdEW[n], yerr=ThirdEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			sc3= plt.scatter(spt[n], ThirdEW[n], alpha = 0.9, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	#plt.colorbar(sc3)
	plt.errorbar(spt_ML, ThirdEW_ML, yerr=ThirdEW_err_ML, fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
	plt.scatter(spt_ML, ThirdEW_ML, alpha = 0.4, s=120, c='k',zorder=1, marker= '^')
	plt.xticks(np.arange(9,20,1))
	P3.set_xticklabels(labels)

	plt.legend((data1, data2, data3, data4, data5), ("Spec & Photo Pec", "Spectrally Pec", "Photometrically Pec", "Not Pec", "McLean et al. 2003"), fontsize= 11,  loc='lower left', numpoints=1)


	P4=plt.subplot(224, sharex=P1)
	#plt.ylim(1.24,1.315)
	plt.xlim(9.5,19.5)
	plt.ylabel("1.2525 $\mu$m EW")
	plt.xlabel("Spectral Type")
	for n in range(len(spt)):
		if FourthEW[n] != 0: 
			plt.errorbar(spt[n], FourthEW[n], yerr=FourthEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			sc4= plt.scatter(spt[n], FourthEW[n], alpha = 0.9, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	#plt.colorbar(sc4)
	plt.errorbar(spt_ML, FourthEW_ML, yerr=FourthEW_err_ML, fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
	plt.scatter(spt_ML, FourthEW_ML, alpha = 0.4, s=120, c='k',zorder=1, marker= '^')
	plt.xticks(np.arange(9,20,1))
	P4.set_xticklabels(labels)

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	cbar=fig.colorbar(sc4, cax=cbar_ax)
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel('$\Delta J-K_{s}$', rotation=270)


	#Subplots:


	# SPT
	fig1=plt.figure(figsize=(12,8))
	plt.subplots_adjust(hspace=0.001)
	ax1=plt.subplot(221)
	plt.ylim(0,12)
	plt.xlim(10,19.5)
	plt.ylabel("1.1693 $\mu$m EW")
	plt.xlabel("Spectral Type")
	for n in range(len(spt)):
		if FirstEW[n] != 0:
			
			plt.errorbar(spt[n], FirstEW[n], yerr=FirstEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(spt[n], FirstEW[n], alpha = 0.9, s=marker_size[n], c='red',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.errorbar(spt_ML, FirstEW_ML, yerr=FirstEW_err_ML, fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
	plt.scatter(spt_ML, FirstEW_ML, alpha = 0.4, s=120, c='k',zorder=1, marker= '^')
	plt.xticks(np.arange(9,20,1))
	ax1.set_xticklabels(labels)
	plt.setp(ax1.get_xticklabels(), visible=False)


	ax2=plt.subplot(222)
	plt.ylim(2,16)
	plt.xlim(10,19.5)
	plt.ylabel("1.1773 $\mu$m EW")
	plt.xlabel("Spectral Type")
	for n in range(len(spt)):
		if SecondEW[n] != 0:
			plt.errorbar(spt[n], SecondEW[n], yerr=SecondEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(spt[n], SecondEW[n], alpha = 0.9, s=marker_size[n], c='red', cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.errorbar(spt_ML, SecondEW_ML, yerr=SecondEW_err_ML, fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
	plt.scatter(spt_ML, SecondEW_ML, alpha = 0.4, s=120, c='k',zorder=1, marker= '^')
	plt.xticks(np.arange(9,20,1))
	ax2.set_xticklabels(labels)
	plt.setp(ax2.get_xticklabels(), visible=False)
#
	ax3=plt.subplot(223)
	plt.ylim(0.5,11)
	plt.xlim(10,19.5)
	plt.ylabel("1.2436 $\mu$m EW")
	plt.xlabel("Spectral Type")
	for n in range(len(spt)):
		if ThirdEW[n] != 0:
			
			plt.errorbar(spt[n], ThirdEW[n], yerr=ThirdEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(spt[n], ThirdEW[n], alpha = 0.9, s=marker_size[n], c='red', cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.errorbar(spt_ML, ThirdEW_ML, yerr=ThirdEW_err_ML, fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
	plt.scatter(spt_ML, ThirdEW_ML, alpha = 0.4, s=120, c='k',zorder=1, marker= '^')
	plt.xticks(np.arange(9,20,1))
	ax3.set_xticklabels(labels)
	
	plt.legend((extra, data7, data8, data9, data10, data5), ("This work:", "Spec & Photo Pec", "Spectrally Pec", "Photometrically Pec", "Not Pec", "McLean et al. 2003"), fontsize= 11,  loc='lower left', numpoints=1)
	


	ax4=plt.subplot(224)
	plt.ylim(0,12)
	plt.xlim(10,19.5)
	plt.ylabel("1.2525 $\mu$m EW")
	plt.xlabel("Spectral Type")
	for n in range(len(spt)):
		if FirstEW[n] != 0:
			plt.errorbar(spt[n], FourthEW[n], yerr=FourthEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(spt[n], FourthEW[n], alpha = 0.9, s=marker_size[n], c='red', cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.errorbar(spt_ML, FourthEW_ML, yerr=FourthEW_err_ML, fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
	plt.scatter(spt_ML, FourthEW_ML, alpha = 0.4, s=120, c='k',zorder=1, marker= '^')
	plt.xticks(np.arange(9,20,1))
	ax4.set_xticklabels(labels)


	# JK
	fig2=plt.figure(figsize=(12,8))
	ax5=plt.subplot(221)
	plt.subplots_adjust(hspace=0.001)
	plt.ylabel("1.1693 $\mu$m EW")
	plt.xlabel("$\Delta J-K_{s}$")
	for n in range(len(spt)):
		if FirstEW[n] != 0:
			plt.errorbar(JK_dev[n], FirstEW[n], yerr=FirstEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(JK_dev[n], FirstEW[n], alpha = 0.75, s=marker_size[n], c='white',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.setp(ax5.get_xticklabels(), visible=False)
	plt.legend((data1, data2, data3, data4), ("Spec & Photo Pec", "Spectrally Pec", "Photometrically Pec", "Not Pec"), fontsize= 11,  loc='lower left', numpoints=1)
	

	ax6=plt.subplot(222)
	plt.ylabel("1.1773 $\mu$m EW")
	plt.xlabel("$\Delta J-K_{s}$")
	for n in range(len(spt)):
		if SecondEW[n] != 0:
			plt.errorbar(JK_dev[n], SecondEW[n], yerr=SecondEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(JK_dev[n], SecondEW[n], alpha = 0.75, s=marker_size[n], c='white', cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.setp(ax6.get_xticklabels(), visible=False)
#
	ax7=plt.subplot(223)
	plt.ylabel("1.2436 $\mu$m EW")
	plt.xlabel("$\Delta J-K_{s}$")
	for n in range(len(spt)):
		if ThirdEW[n] != 0:
			plt.errorbar(JK_dev[n], ThirdEW[n], yerr=ThirdEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(JK_dev[n], ThirdEW[n], alpha = 0.75, s=marker_size[n],zorder=1,c='white', marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass

	ax8=plt.subplot(224)
	plt.ylabel("1.2525 $\mu$m EW")
	plt.xlabel("$\Delta J-K_{s}$")
	for n in range(len(spt)):
		if FirstEW[n] != 0:
			plt.errorbar(JK_dev[n], FourthEW[n], yerr=FourthEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(JK_dev[n], FourthEW[n], alpha = 0.75, s=marker_size[n], c='white',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass

	# JH
	fig3=plt.figure(figsize=(12,8))
	ax9=plt.subplot(221)
	plt.subplots_adjust(hspace=0.001)
	plt.ylabel("1.1693 $\mu$m EW")
	plt.xlabel("$\Delta J-H$")
	for n in range(len(spt)):
		if FirstEW[n] != 0:
			plt.errorbar(JH_dev[n], FirstEW[n], yerr=FirstEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(JH_dev[n], FirstEW[n], alpha = 0.75, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.setp(ax9.get_xticklabels(), visible=False)
	plt.legend((data1, data2, data3, data4), ("Spec & Photo Pec", "Spectrally Pec", "Photometrically Pec", "Not Pec"), fontsize= 11,  loc='lower left', numpoints=1)
	


	ax10=plt.subplot(222)
	plt.ylabel("1.1773 $\mu$m EW")
	plt.xlabel("$\Delta J-H$")
	for n in range(len(spt)):
		if SecondEW[n] != 0:
			plt.errorbar(JH_dev[n], SecondEW[n], yerr=SecondEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(JH_dev[n], SecondEW[n], alpha = 0.75, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.setp(ax10.get_xticklabels(), visible=False)
#
	ax11=plt.subplot(223)
	plt.ylabel("1.2436 $\mu$m EW")
	plt.xlabel("$\Delta J-H$")
	for n in range(len(spt)):
		if ThirdEW[n] != 0:
			plt.errorbar(JH_dev[n], ThirdEW[n], yerr=ThirdEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(JH_dev[n], ThirdEW[n], alpha = 0.75, s=marker_size[n],zorder=1,c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm', marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
#
	ax12=plt.subplot(224)
	plt.ylabel("1.2525 $\mu$m EW")
	plt.xlabel("$\Delta J-H$")
	for n in range(len(spt)):
		if FirstEW[n] != 0:
			plt.errorbar(JH_dev[n], FourthEW[n], yerr=FourthEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			df1=plt.scatter(JH_dev[n], FourthEW[n], alpha = 0.75, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	fig3.subplots_adjust(right=0.8)
	cbar_ax = fig3.add_axes([0.85, 0.15, 0.05, 0.7])
	cbar=fig3.colorbar(df1, cax=cbar_ax)
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel('$\Delta J-K_{s}$', rotation=270)

	# HK
	fig4=plt.figure(figsize=(12,8))
	ax13=plt.subplot(221)
	plt.subplots_adjust(hspace=0.001)
	plt.ylabel("1.1693 $\mu$m EW")
	plt.xlabel("$\Delta H-K_{s}$")
	for n in range(len(spt)):
		if FirstEW[n] != 0:
			plt.errorbar(HK_dev[n], FirstEW[n], yerr=FirstEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(HK_dev[n], FirstEW[n], alpha = 0.75, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.setp(ax13.get_xticklabels(), visible=False)
	plt.legend((data1, data2, data3, data4), ("Spec & Photo Pec", "Spectrally Pec", "Photometrically Pec", "Not Pec"), fontsize= 11,  loc='lower left', numpoints=1)
	

	ax14=plt.subplot(222)
	plt.ylabel("1.1773 $\mu$m EW")
	plt.xlabel("$\Delta H-K_{s}$")
	for n in range(len(spt)):
		if SecondEW[n] != 0:
			plt.errorbar(HK_dev[n], SecondEW[n], yerr=SecondEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(HK_dev[n], SecondEW[n], alpha = 0.75, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	plt.setp(ax14.get_xticklabels(), visible=False)
#
	ax15=plt.subplot(223)
	plt.ylabel("1.2436 $\mu$m EW")
	plt.xlabel("$\Delta H-K_{s}$")
	for n in range(len(spt)):
		if ThirdEW[n] != 0:
			plt.errorbar(HK_dev[n], ThirdEW[n], yerr=ThirdEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			plt.scatter(HK_dev[n], ThirdEW[n], alpha = 0.75, s=marker_size[n],zorder=1,c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm', marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
#
	ax16=plt.subplot(224)
	plt.ylabel("1.2525 $\mu$m EW")
	plt.xlabel("$\Delta H-K_{s}$")
	for n in range(len(spt)):
		if FirstEW[n] != 0:
			plt.errorbar(JK_dev[n], FourthEW[n], yerr=FourthEW_err[n], fmt='none', alpha=.9, zorder=-1, linestyle='None', ecolor='grey', elinewidth=2)
			df2 = plt.scatter(JK_dev[n], FourthEW[n], alpha = 0.75, s=marker_size[n], c=JK_dev[n], vmin=-std, vmax=std, cmap='coolwarm',zorder=1, marker= MarkerStyle(marker_style[n], fillstyle = 'full'))
		else:
			pass
	fig4.subplots_adjust(right=0.8)
	cbar_ax = fig4.add_axes([0.85, 0.15, 0.05, 0.7])
	cbar=fig4.colorbar(df2, cax=cbar_ax)
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel('$\Delta J-K_{s}$', rotation=270)



