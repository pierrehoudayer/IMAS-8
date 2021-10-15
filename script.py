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
#                             DIAGNOSTIC DEFINITION                          #
#============================================================================#
def read_freq(fname, l_max, n_min, n_max):
    '''Return the frequencies contained in fname with 
    l <= l_max, n_min <= n <= n_max'''
    return pd.read_csv(fname, index_col=0, usecols=range(l_max+2), na_values=-99.9999).loc[n_min:n_max]

def read_freq_cov(fname, l_max, n_min, n_max):
    '''Return the covariance matrices (1 per degree)
    contained in fname with l <= l_max, n_min <= n <= n_max'''
    return np.array([
        np.diag(pd.read_csv(fname, index_col=0, usecols=[0,4+l], na_values=-9.9999)
            .loc[n_min:n_max].values.reshape((n_max-n_min+1))**2) for l in range(l_max+1)
        ])

def generate_freq_cov(sig, l_max, n_min, n_max):
    '''Genrerate the covariance matrices (1 per degree)
    using a single dispersion value (sig) with l <= l_max, n_min <= n <= n_max'''
    return np.array([sig**2*np.eye(n_max-n_min+1) for l in range(l_max+1)])


def dk(f, f_cov, k):
    '''Return the k-th differences (and the corresponding covariance matrix) 
    of the frequencies f'''
    if k == 0:
        dk_f = f.copy()
        dk_f_cov = f_cov.copy()
        for l in range(f.shape[1]):
            dk_f['l = {}'.format(l)] -= f['l = {}'.format(l)].index.values + l/2
        return dk_f, dk_f_cov
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
        resolution = 1000
        all_f = np.array(f).flatten()
        ordered = np.argsort(all_f)
        f_long = np.linspace(all_f[ordered][0], all_f[ordered][-1], resolution)
        plt.plot(f_long, dk_f_mod, 'g-', alpha = 0.5, label="Model")
    plt.legend()
    if show: plt.show()
#============================================================================#



#============================================================================#
#                               FITTING FUNCTIONS                            #
#============================================================================#
def M94a(f, a, tau, phi):
    '''Expression of the frequency shift caused by the base of the convection
    zone (BCZ) in an overshoot model derived in Monteiro et al. (1994) and 
    reused in Vrard et al. (2015)'''
    return (a/f) * np.cos(2*np.pi*f*tau+phi)

def M94b(f, a, tau, phi):
    '''Expression of the frequency shift caused by the base of the convection
    zone (BCZ) in a non-overshoot model derived in Monteiro et al. (1994) and 
    reused in Vrard et al. (2015) and Dréau et al. (2021)'''
    return (a/f**2) * np.cos(2*np.pi*f*tau+phi)

def MT98(f, a, b, tau, phi):
    '''Expression of the frequency shift caused by the 2nd He ionisation
    derived in Monteiro & Thompson (1998) and used in Verma et al. (2014a)'''
    return a * np.sin(np.pi*f*b)**2/(np.pi*f*b) * np.cos(2*np.pi*f*tau+phi)

def HG07(f, a, b, tau, phi):
    '''Expression of the frequency shift caused by the 2nd He ionisation
    derived in Houdek & Gough (2007)
    This expression were also applied to the second differences fit in 
    Verma et al. (2014a, 2019) and adapted in Farnir et al. (2019)'''
    return (a*f) * np.exp(-(b*f)**2) * np.cos(2*np.pi*f*tau+phi)

def V15(f, a, tau, phi):
    '''Expression of the frequency shift caused by the 2nd He ionisation
    used in Vrard et al. (2015)'''
    return a * np.cos(2*np.pi*f*tau+phi)

def Offset(f, a0):
    '''Just an offset to model the smooth component'''
    return a0

def Inverse_polynomial(f, a0, a1, a2, a3):
    '''Expression of the smooth component introduced in Houdek & Gough (2007)'''
    return a0 + a1*f**-1 + a2*f**-2 + a3*f**-3

def Polynomial(f, a0, a1, a2):
    '''Expression of the smooth component introduced in Verma et al. (2014)'''
    return a0 + a1*f**1 + a2*f**2

def Asymptotic(k, f, a0, a1):
    '''Expression of the smooth component derived of the asymptotic 
    expansion presented in Houdek & Gough (2011). 
    For each k-th differences, we kept only the most important terms.'''
    if k == 1:
        return 1 + a0*f**-2 + a1*f**-4
    if k == 2:
        return 0 + a0*f**-3 + a1*f**-5

def arg_types(func):
    'Argument types of func for prior construction'
    if func == M94a:
        return ['scale', 'depth', 'phase']
    if func == M94b:
        return ['scale', 'depth', 'phase']
    if func == MT98:
        return ['scale', 'scale', 'depth', 'phase']
    if func == HG07:
        return ['scale', 'scale', 'depth', 'phase']
    if func == V15:
        return ['scale', 'depth', 'phase']
    if func == Offset:
        return ['scale']
    if func == Inverse_polynomial:
        return ['scale', 'scale', 'scale', 'scale']
    if func == Polynomial:
        return ['scale', 'scale', 'scale']
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
            if fname == 'freq_model_RGB.csv' or fname == 'freq_KIC_6277741_RGB.csv' :
                x0.append(0.5)
            elif fname == 'freq_model_MS.csv' or fname == 'freq_KIC_8379927_MS.csv' :
                x0.append(0.1)
            else :
                x0.append(0.1)
        if kind == 'depth' :
            if fname == 'freq_model_RGB.csv' or fname == 'freq_KIC_6277741_RGB.csv' :
                x0.append(0.5)
            elif fname == 'freq_model_MS.csv' or fname == 'freq_KIC_8379927_MS.csv' :
                x0.append(0.3)
            else :
                x0.append(0.3)
        if kind == 'phase' :
            if fname == 'freq_model_RGB.csv' or fname == 'freq_KIC_6277741_RGB.csv' :
                x0.append(0.0)
            elif fname == 'freq_model_MS.csv' or fname == 'freq_KIC_8379927_MS.csv' :
                x0.append(0.0)
            else :
                x0.append(0.0)
    return np.array(x0)
        
    
def log_prior(func, *args):
    types = arg_types(func)
    logp = []
    for kind, par in zip(types, args):
        if kind == 'scale' :
            logp.append(log_prior_on_scale(par))
        if kind == 'depth' :
            logp.append(log_prior_on_depth(par, 0.25, 0.1))
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
    fname = 'freq_KIC_8379927_MS.csv'
    
    # Maximum degree (<4) and radial orders considered
    l_max = 2
    n_min = 2
    n_max = 14
    
    # We consider a particular set of radial order to avoid NaNs
    # in the specific case of frequencies derived from data
    if fname == 'freq_KIC_6277741_RGB.csv':
        n_min = 5
        n_max = 9 
    if fname == 'freq_KIC_8379927_MS.csv':
        n_min = 16
        n_max = 26
    
    # We read the frequencies in fname
    freq = read_freq(fname, l_max, n_min, n_max)
    
    # We define a covariance matrix 
    if 'KIC' in fname:
        freq_cov = read_freq_cov(fname, l_max, n_min, n_max)
    else:
        # We naively asssume here that all degrees l share the same covariance matrix
        sig = 1e-3    # This means sigma(\nu)/\Delta\nu ~ 0.001 for all frequencies
        freq_cov = generate_freq_cov(sig, l_max, n_min, n_max)
    
    
    # Diagnostic considered (d^k nu = k^th differences)
    k = 1                          # Different options: 0, 1, 2
    dk_freq, dk_freq_cov = dk(freq, freq_cov, k)
    
    
    # Definition of a model
    glitch = V15                   # Different options: M94a, M94b, MT98, HG07, V15
    smooth = Inverse_polynomial    # Different options: Offset, Polynomial, Inverse_polynomial
    # smooth = lambda f, a, b: Asymptotic(k, f, a, b)
    model  = lambda f, *params: glitch(f, *params[:glitch.__code__.co_argcount-1]) \
                              + smooth(f, *params[glitch.__code__.co_argcount-1:])
    
    # Minimisation using curve_fit
    all_freq = np.array(freq.loc[dk_freq.index]).flatten()
    all_dk_freq = np.array(dk_freq).flatten()
    all_dk_freq_cov = block_diag(*dk_freq_cov)
    ordered = np.argsort(all_freq)
    
    # Initial guess for our model
    x0 = np.concatenate((first_guess(glitch), first_guess(smooth)))
    maxfev = int(1e5)    
    Two_fits = True     # This option allows a first fit to determine a first guess
    
    if Two_fits:
        # Fit of the smooth component alone
        p_smooth, p_smooth_cov = curve_fit(smooth, all_freq[ordered], all_dk_freq[ordered], 
                                            p0=first_guess(smooth), sigma=all_dk_freq_cov[ordered], 
                                            absolute_sigma=True, maxfev=maxfev)
        residuals = all_dk_freq[ordered] - smooth(all_freq[ordered], *p_smooth)
        
        # Fit of the residuals using the glitch model
        p_glitch, p_glitch_cov = curve_fit(glitch, all_freq[ordered], residuals, 
                                            p0=first_guess(glitch), sigma=all_dk_freq_cov[ordered], 
                                            absolute_sigma=True, maxfev=maxfev)
        
        # The best solution is then used as a first guess for the complete fit (smooth + glitch)
        x0 = np.concatenate((p_glitch, p_smooth))
    
    # Fit of the complete model
    p_mod, p_mod_cov = curve_fit(model, all_freq[ordered], all_dk_freq[ordered], 
                                 p0=x0, sigma=all_dk_freq_cov[ordered],
                                 absolute_sigma=True, maxfev=maxfev)
    
    print('Best parameter set : ', *zip(np.concatenate((glitch.__code__.co_varnames[1:],
                                                        smooth.__code__.co_varnames[1:])),
                                        np.round(p_mod, 5)))
    resolution = 1000
    freq_long = np.linspace(all_freq[ordered][0], all_freq[ordered][-1], resolution)
    dk_freq_mod = model(freq_long, *p_mod)
    
    # Comparison of the fitted expression with the data
    plot_diagnostic(freq, dk_freq, dk_freq_cov, dk_freq_mod, show=True)
    
    
    # # Definition of a cost function
    # cost_func = lambda params: - log_prior(glitch, *params[:glitch.__code__.co_argcount-1]) \
    #                             - log_prior(smooth, *params[glitch.__code__.co_argcount-1:]) \
    #                             - log_likelihood(freq, dk_freq, dk_freq_cov, model, *params)
    
    # # Minimisation using a log posterior function
    # res = minimize(cost_func, x0, method='Powell', tol=1e-4, options={'maxiter': 1000, 'apdative':True})
    # print('Best parameter set : ', *zip(np.concatenate((glitch.__code__.co_varnames[1:],
    #                                                     smooth.__code__.co_varnames[1:])),
    #                                     np.round(res.x, 3)))
    # dk_freq_mod = model(freq, *res.x)
    # plot_diagnostic(freq, dk_freq, dk_freq_cov, dk_freq_mod, show=True)
    
#============================================================================#
