# ********************************************************************************
# ******************* 8th Iberian Meeting on AsteroSeismology ********************
# ************* On determining accurate proxy for helium abundance ***************
# ********************************************************************************

# ***************************** Sets of frequencies ******************************

    In this folder, you will find several sets of p-mode frequencies derived 
from observations and stellar models:


- [freq_KIC_6277741_RGB.csv] : dimensionless p-mode frequencies (nunl / Dnu) of 
the RGB star KIC 6277741 (Dnu = 3.0 muHz, M = 1.36 M_sun), derived 
from Dréau et al. (2021). -99.9999 and -9.9999 means no available data.

- [freq_model_RGB.csv] : dimensionless p-mode frequencies (nunl / Dnu) of a RGB 
model (Dnu = 3.0 muHz, M = 1.36 M_sun) computed from the stellar evolution code 
MESA and oscillation code ADIPLS, with the method of Ball et al. (2018). 

- [freq_KIC_8379927_MS.csv] : dimensionless p-mode frequencies (nunl / Dnu) 
of the MS star KIC 8379927 (Dnu = 120.4 muHz, M = 1.17 M_sun), derived from 
Appourchaux et al. (2012). -99.9999 and -9.9999 means no available data.

- [freq_model_MS.csv] : dimensionless p-mode frequencies (nunl / Dnu) of a MS 
model (Dnu = 120.9 muHz, M = 1.2 M_sun) computed from the stellar evolution code 
MESA and oscillation code ADIPLS. 

- [freq1.csv], [freq2.csv], [freq3.csv] : dimensionless p-mode frequencies 
(nunl / Dnu) of three MS model that includes the He ionisation zones only 
(Houdayer et al., 2021). Between those models, three different tuple (Ys, psi) 
have been chosen. Ys is the mass fraction of helium while psi is the electron 
degeneracy at the Helium ionisation zones. The valies of the tuples (Ys, psi) are 
hidden on purpose. 

# ************************************ Script ************************************

    We provide a script in python called [script.py] to analyse the 
modulation induced by the helium ionisation zones in mode frequencies. It is not 
meant to be optimal, so do not hesitate to use your own code if you have any. 
In order to run this python code, you need to install the following modules:
- [types, pandas, numpy, matplotlib, scipy]

In the framework of this hands-on project, we will be playing with the following
options:

- l253     [fname]       : filename (one of the files presented above)
- l257-258 [nmin, nmax]  : minimum and maximum radial order to consider
- l282     [k]           : d^k nu = k^th differences in mode frequencies
- l287     [glitch]      : expression of the glitch signature (all the options are
described in the section [Fitting functions]. Do not hesitate to adopt other 
expressions.
- l288     [smooth]      : expression of the smooth function (all the options are
described in the section [Fitting functions]. Do not hesitate to adopt other 
expressions.
- l187     [first_guess] : function used to define the first guesses of the free 
parameters. There are three types of options: [depth] is for the period of the
modulation, [phase] is for the phase, and [scale] is for the remaining parameters.

	If you have any difficulty to run this code, please do not hesitate to
contact us.

# ********************************* @authors *************************************
[Guillaume Dréau] & [Pierre Houdayer]
[guillaume.dreau@obspm.fr], [pierre.houdayer@obspm.fr]
# ********************************************************************************
