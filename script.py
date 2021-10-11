#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 09:32:22 2021

@author: phoudayer
"""

import types
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import minimize, curve_fit
from scipy.linalg import block_diag


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
def dk(f, f_cov, k):
    '''Return the k-th differences (and the associated covariance matrix) 
    of the frequencies f'''
    if k == 0:
        return f, f_cov
    if k == 1:
        a = [1,0,-1]       # Here we consider centered differences
    if k == 2:
        a = [1,-2,1]
        
    q = len(a)//2
    dimA = len(f)
    A = sum(np.diag(np.flip(a)[i]*np.ones(dimA-np.abs(i-q))/2**k,i-q) for i in range(len(a)))
    dk_f     = pd.DataFrame(np.dot(A, f)[q:-q], index=f.index[q:-q], columns=f.columns)
    dk_f_cov = np.array([np.dot(np.dot(A,f_cov_l),A.T)[q:-q,q:-q] for f_cov_l in f_cov])
    return dk_f, dk_f_cov

def plot_diagnostic(f, dk_f, dk_f_cov, dk_f_mod=None, show=False): 
    '''Plot the k-th differences as a function of frequency'''
    colors = ['k', 'r', 'b']
    markers = ['v', 'd', '*']
    for l_name, dk_f_cov_l, m_l, c_l in zip(f, dk_f_cov, markers, colors): 
        plt.errorbar(f.loc[dk_f[l_name].index, l_name], dk_f[l_name], yerr=np.diag(dk_f_cov_l)**0.5,
                    fmt=m_l, color=c_l, alpha=0.3, label=r"$\ell = "+l_name[-1]+"$")
    if dk_f_mod is not None:
        all_f = np.array(f).flatten()
        ordered = np.argsort(all_f)
        all_mod = np.array(dk_f_mod).flatten()
        plt.plot(all_f[ordered], all_mod[ordered], 'g-', alpha = 0.5, label="Model")
    plt.legend()
    if show: plt.show()
#============================================================================#



#============================================================================#
#                               FITTING FUNCTIONS                            #
#============================================================================#
def M94a(f, a, τ, φ):
    '''Expression of the frequency shift caused by the base of the convection
    zone (BCZ) in an overshoot model derived in Monteiro et al. (1994) and 
    reused in Vrard et al. (2015)'''
    return (a/f) * np.cos(2*np.pi*f*τ+φ)

def M94b(f, a, τ, φ):
    '''Expression of the frequency shift caused by the base of the convection
    zone (BCZ) in a non-overshoot model derived in Monteiro et al. (1994) and 
    reused in Vrard et al. (2015) and Dréau et al. (2021)'''
    return (a/f**2) * np.cos(2*np.pi*f*τ+φ)

def MT98(f, a, b, τ, φ):
    '''Expression of the frequency shift caused by the 2nd He ionisation
    derived in Monteiro & Thompson (1998) and used in Verma et al. (2014a)'''
    return a * np.sin(np.pi*f*b)**2/(np.pi*f*b) * np.cos(2*np.pi*f*τ+φ)

def HG07(f, a, b, τ, φ):
    '''Expression of the frequency shift caused by the 2nd He ionisation
    derived in Houdek & Gough (2007)
    This expression were also applied to the second differences fit in 
    Verma et al. (2014a, 2019) and adapted in Farnir et al. (2019)'''
    return (a*f) * np.exp(-(b*f)**2) * np.cos(2*np.pi*f*τ+φ)

def V15(f, a, τ, φ):
    '''Expression of the frequency shift caused by the 2nd He ionisation
    derived in Vrard et al. (2015)'''
    return a * np.cos(2*np.pi*f*τ+φ)

def Inverse_polynomial(f, a, b, c, d):
    '''Expression of the smooth component introduced in Houdek & Gough (2007)'''
    return a + b*f**-1 + c*f**-2 + d*f**-3

def Asymptotic(k, f, a, b):
    '''Expression of the smooth component derived of the asymptotic 
    expansion presented in Houdek & Gough (2011). 
    For each k-th differences, we kept only the most important terms.'''
    if k == 0:
        return f + a*f**-1 + b*f**-3
    if k == 1:
        return 1 + a*f**-2 + b*f**-4
    if k == 2:
        return 0 + a*f**-3 + b*f**-5

def arg_types(func):
    'Argument types of func for prior construction'
    if func == M94a:
        return ['scale', 'depth', 'phase']
    if func == M94b:
        return ['scale', 'depth', 'phase']
    if func == MT98:
        return ['scale', 'depth', 'depth', 'phase']
    if func == HG07:
        return ['scale', 'scale', 'depth', 'phase']
    if func == V15:
        return ['scale', 'depth', 'phase']
    if func == Inverse_polynomial:
        return ['scale', 'scale', 'scale', 'scale']
    if (func == Asymptotic or type(func) is types.LambdaType):
        return ['scale', 'scale']
#============================================================================#



#============================================================================#
#                              POSTERIOR DEFINITION                          #
#============================================================================#    
def log_prior_on_scale(par):
    '''Return Jeffrey's prior on scale'''
    # return -np.log(np.abs(par))
    return 0.0
    

def log_prior_on_depth(par, m, std):
    '''Return the Beta districution of mean m and standard deviation std'''
    a =     m*(m*(1-m)/std**2-1)
    b = (1-m)*(m*(1-m)/std**2-1)
    return st.beta(a, b).logpdf(par)   

def log_prior_on_phase(par):
    '''Return Jeffrey's prior on phase'''
    return 0.0

def first_guess(func):
    types = arg_types(func)
    x0 = []
    for kind in types:
        if kind == 'scale' :
            x0.append(0.1)
        if kind == 'depth' :
            x0.append(0.25)
        if kind == 'phase' :
            x0.append(0.0)
    return np.array(x0)
        
    
def log_prior(func, *args):
    types = arg_types(func)
    logp = []
    for kind, par in zip(types, args):
        if kind == 'scale' :
            logp.append(log_prior_on_scale(par))
        if kind == 'depth' :
            logp.append(log_prior_on_depth(par, 0.25, 0.2))
        if kind == 'phase' :
            logp.append(log_prior_on_phase(par))
    logp = np.sum(logp)
    print(logp)
    return logp

def log_likelihood(f, dk_f, dk_f_cov, func, *args):
    '''The likelihood function is calculated assuming a 
    Gaussian distribution on the diagnostic dk_f with
    covariance matrix dk_f_cov'''
    logl = 0.0
    for l_name, dk_f_cov_l in zip(f, dk_f_cov): 
        dk_f_mod_l = func(f.loc[dk_f[l_name].index, l_name], *args)
        logl += st.multivariate_normal(dk_f_mod_l, dk_f_cov_l).logpdf(dk_f[l_name])
    print(logl)
    print('')
    return logl

def log_posterior(f, dk_f, dk_f_cov, func, *args):
    logp = log_prior(func, *args)
    logl = log_likelihood(f, dk_f, dk_f_cov, func, *args)
    return logp + logl
#============================================================================#


#============================================================================#
#                                 MAIN SCRIPT                                #
#============================================================================#
if __name__=='__main__':
    # Name of the file containing the frequencies
    Y_char = 'Y253'
    M_char = '1.00Msun'
    fname  = 'Yveline/' + M_char + '_' + Y_char + '_FREQ_SCALED.csv'
    # fname = 'freq1.csv'
    
    # Maximum degree (<4) and radial orders considered
    l_max = 2
    n_min = 10
    n_max = 40         
    freq = pd.read_csv(fname, index_col=0, usecols=range(l_max+2)).loc[n_min:n_max]
    
    # We define a covariance matrix 
    # We naively asssume here that all degrees l share the same covariance matrix
    sig = 1e-3    # This means sigma(\nu/\Delta\nu) ~ 0.001 for all frequencies
    freq_cov = np.array([sig**2*np.eye(freq.shape[0]) for l in range(l_max+1)])
    
    
    # Diagnostic considered (d^k nu = k^th differences)
    k = 2
    dk_freq, dk_freq_cov = dk(freq, freq_cov, k)
    
    # Definition of a glitch model
    glitch = HG07
    smooth = Inverse_polynomial
    # smooth = lambda f, a, b: Asymptotic(k, f, a, b)
    model  = lambda f, *params: glitch(f, *params[:glitch.__code__.co_argcount-1]) \
                              + smooth(f, *params[glitch.__code__.co_argcount-1:])
    
    # Initial guess for our model
    x0 = np.concatenate((first_guess(glitch), first_guess(smooth)))
    
    # Definition of a cost function
    cost_func = lambda params: - log_prior(glitch, *params[:glitch.__code__.co_argcount-1]) \
                                - log_prior(smooth, *params[glitch.__code__.co_argcount-1:]) \
                                - log_likelihood(freq, dk_freq, dk_freq_cov, model, *params)
    
    # # Minimisation
    # res = minimize(cost_func, x0, method='Powell', tol=1e-4, options={'maxiter': 1000, 'apdative':True})
    # print('Best parameter set : ', *zip(np.concatenate((glitch.__code__.co_varnames[1:],
    #                                                     smooth.__code__.co_varnames[1:])),
    #                                     np.round(res.x, 3)))
    # dk_freq_mod = model(freq, *res.x)
    # plot_diagnostic(freq, dk_freq, dk_freq_cov, dk_freq_mod, show=True)
    
    # Minimisation using curve_fit
    Two_fits = True
    all_freq = np.array(freq.loc[dk_freq.index]).flatten()
    all_dk_freq = np.array(dk_freq).flatten()
    all_dk_freq_cov = block_diag(*dk_freq_cov)
    ordered = np.argsort(all_freq)
    
    if Two_fits:
        p_smooth, p_smooth_cov = curve_fit(smooth, all_freq[ordered], all_dk_freq[ordered], 
                                            p0=first_guess(smooth), sigma=all_dk_freq_cov[ordered], 
                                            absolute_sigma=True)
        residuals = all_dk_freq[ordered] - smooth(all_freq[ordered], *p_smooth)
        p_glitch, p_glitch_cov = curve_fit(glitch, all_freq[ordered], residuals, 
                                            p0=first_guess(glitch), sigma=all_dk_freq_cov[ordered], 
                                            absolute_sigma=True)
        x0 = np.concatenate((p_glitch, p_smooth))
        # dk_freq_glitch = np.array(glitch(freq, *p_glitch)).flatten()
        # dk_freq_smooth = np.array(smooth(freq, *p_smooth)).flatten()
        # dk_freq_model  = np.array(model(freq, *p_mod)).flatten()
        # print('Best parameter set : ', *zip(np.concatenate((glitch.__code__.co_varnames[1:],
        #                                                     smooth.__code__.co_varnames[1:])),
        #                                     np.round(p_mod, 5)))
        # plot_diagnostic(freq, dk_freq, dk_freq_cov, show=False)
        # for fit, style, label in zip([dk_freq_glitch, dk_freq_smooth, dk_freq_model],
        #                               ['k--', 'k:', 'k-'], ['Glitch', 'Smooth', 'Model']):
        #     plt.plot(all_freq[ordered], fit[ordered], style, label=label)
        # plt.show()
    
    
    p_mod, p_mod_cov = curve_fit(model, all_freq[ordered], all_dk_freq[ordered], 
                                 p0=x0, sigma=all_dk_freq_cov[ordered], absolute_sigma=True)
    print('Best parameter set : ', *zip(np.concatenate((glitch.__code__.co_varnames[1:],
                                                        smooth.__code__.co_varnames[1:])),
                                        np.round(p_mod, 5)))
    dk_freq_mod = model(freq, *p_mod)
    plot_diagnostic(freq, dk_freq, dk_freq_cov, dk_freq_mod, show=True)
    
#============================================================================#
