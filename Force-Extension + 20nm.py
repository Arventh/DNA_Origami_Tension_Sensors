#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:09:07 2021
Updated 2022 July 17
@author: arventh
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI']
#Inputfile wtih time, energy, HBs, strand length and x, y, z co-ords for 2 the two traps, i.e. Zf[4:7], Zf[7:10] 
folder = '/home/Data/....' #raw_input directory
txt = '3-0-1_out_Observables_1.dat'
os.chdir(folder)
data = open(txt, 'r')
i, x, y, y_peak, err = 0,[],[],[],[]

#Constants. Change these parameters to simulation conditons.
dt = 0.005 #simulation time units (Output has time in MD units where MD units = steps * dt)
F_units = 48.63 #units in pN 
l_units = 0.8518 #units in nm 
t_units = 3.03e-12 #seconds
k1,k2 = 0.2,0.2 #simulation units
m_avg = 400 #exponential moving average smoothing of n-points
ext_rate = 0.5e-7 #rate of increase length (simulation units) per simulation step
F_dir = np.array([0,0,-1]) #direction of trap movement
t1_init = np.array([38.014832769026, 6.7136758901042, 4.8958553069682]) #moving trap
t2_init = np.array([31.9006098158952, 15.7306077056064, 17.8350084890777]) #fixed trap
UnitF_dir = F_dir/np.linalg.norm(F_dir) #unit vector
init_dist = np.dot((t1_init - t2_init), UnitF_dir) #initial distance between two traps along force dir
keff = ((k1*k2)/(k1+k2)) #1 unit of force constant (1 unit force/1 unit length) - 57.09 pN/nm

fig = plt.figure() 
ax = fig.add_subplot(1,1,1)
for line in data:
    Zf = np.array(line.split(), dtype=float)
    l_dna = Zf[3]*l_units #end to end strand length
    trap1_current = t1_init + (Zf[0]/dt)*ext_rate*UnitF_dir #trap1 displacement
    trap_ext1 = trap1_current -  Zf[7:10]  #vector subtraction of moving trap1 and its attached nucleotide
    trap_ext2 = Zf[4:7] - t2_init #vector subtraction of fixed trap2 and its attached nucleotide
    force = (np.dot((trap_ext1+trap_ext2),UnitF_dir))*keff*F_units
    overall_ext = trap1_current - t2_init
    trap_ext = (np.dot(overall_ext,UnitF_dir)-init_dist)*l_units
    x.append(trap_ext) #this is extension of traps along force-axis not the DNA strand extension.
    y.append(force)
    if Zf[2] <= 75   and i == 0: #Zf[2] is HB count
        Rupture_F = force
        Rupture_ext = trap_ext
        y_peak.append(force)
        i = 1

y_peak = pd.Series(y).rolling(window=m_avg).mean().iloc[m_avg-1:].values # EMA calculation
y_pk = y_peak.tolist()
for loop in range(m_avg-1):
    y_pk.insert(0, 0)

#graph plotting 
ax.set(ylabel='force (pN)', xlabel='trap extension (nm)')
ax.set(xlim=[0,35],ylim=[-2,40])
plt.plot(x,y, color='#f9f871',linewidth=1, alpha = 0.75)
Loadrate = (ext_rate*l_units)/t_units
Loadrate = format(Loadrate, ".2e") + ' nm/s'
Force_label = format(force, ".2f") + ' pN'
plt.plot(x,y_pk, color='#c34a36',linewidth=0.55, alpha = 0.70)
ax.legend([Loadrate, 'EMA ('+ str(m_avg)+ ')'], loc=0, frameon=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#D94862 #344973
#90BF2A #144C59
#f9f871 #c34a36
peaks, ht = find_peaks(y_pk, height=[13.2,13.7], prominence=1) #finds a list of peaks satisfying the criteria
for peakapeaka in peaks:
    if x[int(peakapeaka)] > Rupture_ext: #Range in which peaks are needed
        anno = str(round(y_pk[int(peakapeaka)], 2)) + ' pN' # peak annotation
        plt.plot(x[int(peakapeaka)],y_pk[int(peakapeaka)],"o", color='#000000', alpha=0.6, fillstyle = 'none')
        ax.annotate(anno,xy=(x[int(peakapeaka)], y_pk[int(peakapeaka)]), 
            xycoords='data', xytext=(5,5), #offset the rupture force annotation
            textcoords='offset points', horizontalalignment='left', verticalalignment='bottom')

for pntr in range(1, -400, -1):
    m = peakapeaka+pntr
    err.append(y[m])
break_force = np.mean(err)
error = np.std(err, ddof=1) / np.sqrt(np.size(err))
Force_value = 'S.E.M ' + '\u2213 ' + format(error, ".2f") + ' pN' #format(break_force, ".2f") + 
ax.text(26.5, -0.9, Force_value)

plt.savefig('Force vs trap Extension_1 for ' + txt.rstrip('.dat') + '.png', transparent=True)
plt.show()

for oolala in range(len(x)):
    csv = open("Force_ext_graph_plot.csv", "a")
    csv.write(str(x[oolala]) + ',' + str(y[oolala]) + ',' + str(y_pk[oolala]) + '\n')

csv.close()
data.close()

