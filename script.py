#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 09:32:22 2021

@author: phoudayer
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt


#============================================================================#
#                                   USETEX                                   #
#============================================================================#
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
#============================================================================#


#============================================================================#
#                             DIAGNOSTIC ANALYSIS                            #
#============================================================================#
def dk(f, f_cov, k) :
    '''Return the k-th differences (and the associated covariance matrix) 
       of the frequencies f'''
    if k == 0 :
        return f, f_cov
    if k == 1 :
        a = [1,0,-1]       # Here we consider centered differences
    if k == 2 :
        a = [1,-2,1]
    q = len(a)//2
    dimA = len(f)
    A = np.sum(np.diag(np.flip(a)[i]*np.ones(dimA-np.abs(i-q))/2**k,i-q) for i in range(len(a)))
    dk_f     = pd.DataFrame(np.dot(A, f)[q:-q], index=f.index[q:-q], columns=f.columns)
    dk_f_cov = np.array([np.dot(np.dot(A,cov_l),A.T)[q:-q,q:-q] for cov_l in f_cov])
    return dk_f, dk_f_cov

def plot_diagnostic(f, dk_f, dk_f_cov, show=False) : 
    '''Plot the k-th differences as a function of frequency'''
    colors = ['k', 'r', 'b']
    markers = ['v', 'd', '*']
    for f_l, dk_f_l, dk_f_cov_l, m_l, c_l in zip(f, dk_f, dk_f_cov, markers, colors) : 
        plt.errorbar(f.loc[dk_f[dk_f_l].index, f_l], dk_f[dk_f_l], yerr=np.diag(dk_f_cov_l)**0.5,
                    fmt=m_l, color=c_l, alpha=0.3, label=dk_f_l)
    plt.legend()
    if show: plt.show()
#============================================================================#



#============================================================================#
#                                DIAGNOSTIC FIT                              #
#============================================================================#
...
#============================================================================#



#============================================================================#
#                                 MAIN SCRIPT                                #
#============================================================================#
if __name__=='__main__':
    # Name of the file containing the frequencies
    """Y_char = 'Y253'
    M_char = '1.00Msun'
    fname  = 'Yveline/' + M_char + '_' + Y_char + '_FREQ_SCALED.csv'"""
    fname  = 'freq1.csv'
    
    # Maximum degree (<4) and radial orders considered
    l_max = 2
    n_min = 10
    n_max = 30          
    freq = pd.read_csv(fname, index_col=0, usecols=range(l_max+2)).loc[n_min:n_max]
    
    # We define a covariance matrix 
    # We naively asssume here that all degrees l share the same covariance matrix
    sig = 1e-3    # This means sigma(\nu/\Delta\nu) ~ 0.001 for all frequencies
    freq_cov = np.array([sig**2*np.eye(freq.shape[0]) for l in range(l_max+1)])
    
    
    # Diagnostic considered (d^k nu = k^th differences)
    k = 2
    dk_freq, dk_freq_cov = dk(freq, freq_cov, k)
    plot_diagnostic(freq, dk_freq, dk_freq_cov, show=True)
#============================================================================#
