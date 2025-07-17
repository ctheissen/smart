import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
import matplotlib.gridspec as gridspec
from astropy.io import fits
import emcee
#from schwimmbad import MPIPool
from multiprocessing import Pool
from multiprocessing import set_start_method
import smart
import model_fit
import mcmc_utils
import corner
import os
import sys
import time
import copy
import argparse
import json
import ast
import warnings
from datetime import date, datetime
warnings.filterwarnings("ignore")
import astropy.units as u
from astropy.units import cds
cds.enable()
from astropy.constants import c as speedoflight
import splat

##############################################################################################
## This is the script to make the code multiprocessing, using arcparse to pass the arguments
## The code is run with 8 parameters, including Teff, logg, RV, vsini, telluric alpha, and 
## nuisance parameters for wavelength, flux and noise.
##############################################################################################

parser = argparse.ArgumentParser(description="Run the forward-modeling routine for science files",
	usage="run_mcmc_science.py order date_obs sci_data_name data_path save_to_path lsf priors limits")

#parser.add_argument("source",metavar='src',type=str,
#   default=None, help="source name", nargs="+")

parser.add_argument("order",metavar='o',type=str,
    default=None, help="order", nargs="+")

parser.add_argument("date_obs",metavar='dobs',type=str,
    default=None, help="source name", nargs="+")

parser.add_argument("sci_data_name",metavar='sci',type=str,
    default=None, help="science data name", nargs="+")

#parser.add_argument("tell_data_name",metavar='tell',type=str,
#    default=None, help="telluric data name", nargs="+")

parser.add_argument("data_path",type=str,
    default=None, help="science data path", nargs="+")

#parser.add_argument("tell_path",type=str,
#    default=None, help="telluric data path", nargs="+")

parser.add_argument("save_to_path",type=str,
    default=None, help="output path", nargs="+")

parser.add_argument("lsf",type=float,
    default=None, help="line spread function", nargs="+")

parser.add_argument("-instrument",metavar='--inst',type=str,
    default='nirspec', help="spectrometer name of the instrument; default nirspec")

parser.add_argument("-outlier_rejection",metavar='--rej',type=float,
    default=3.0, help="outlier rejection based on the multiple of standard deviation of the residual; default 3.0")

parser.add_argument("-ndim",type=int,
    default=8, help="number of dimension; default 8")

parser.add_argument("-nwalkers",type=int,
    default=50, help="number of walkers of MCMC; default 50")

parser.add_argument("-step",type=int,
    default=600, help="number of steps of MCMC; default 600")

parser.add_argument("-burn",type=int,
    default=500, help="burn of MCMC; default 500")

parser.add_argument("-moves",type=float,
    default=2.0, help="moves of MCMC; default 2.0")

parser.add_argument("-pixel_start",type=int,
    default=10, help="starting pixel index for the science data; default 10")

parser.add_argument("-pixel_end",type=int,
    default=-40, help="ending pixel index for the science data; default -40")

#parser.add_argument("-pwv",type=float,
#    default=0.5, help="precipitable water vapor for telluric profile; default 0.5 mm")

#parser.add_argument("-alpha_tell",type=float,
#    default=1.0, help="telluric alpha; default 1.0")

parser.add_argument("-applymask",type=bool,
    default=False, help="apply a simple mask based on the STD of the average flux; default is False")

parser.add_argument("-plot_show",type=bool,
    default=False, help="show the MCMC plots; default is False")

parser.add_argument("-coadd",type=bool,
    default=False, help="coadd the spectra; default is False")

parser.add_argument("-coadd_sp_name",type=str,
    default=None, help="name of the coadded spectra")

parser.add_argument("-modelset",type=str,
    default='btsettl08', help="model set; default is btsettl08")

parser.add_argument("-final_mcmc", action='store_true', help="run final mcmc; default False")

parser.add_argument("-include_fringe_model", action='store_true', help="model the fringe pattern; default False")

args = parser.parse_args()

######################################################################################################

#source                 = str(args.source[0])
order                  = str(args.order[0])
instrument             = str(args.instrument)
date_obs               = str(args.date_obs[0])
sci_data_name          = str(args.sci_data_name[0])
data_path              = str(args.data_path[0])
#tell_data_name         = str(args.tell_data_name[0])
#tell_path              = str(args.tell_path[0])
save_to_path_base      = str(args.save_to_path[0])
lsf0                   = float(args.lsf[0])
ndim, nwalkers, step   = int(args.ndim), int(args.nwalkers), int(args.step)
burn                   = int(args.burn)
moves                  = float(args.moves)
applymask              = args.applymask
pixel_start, pixel_end = int(args.pixel_start), int(args.pixel_end)
#pwv                    = float(args.pwv)
#alpha_tell             = float(args.alpha_tell[0])
plot_show              = args.plot_show
coadd                  = args.coadd
outlier_rejection      = float(args.outlier_rejection)
modelset               = str(args.modelset)
final_mcmc             = args.final_mcmc
include_fringe_model   = args.include_fringe_model

if final_mcmc:
	#save_to_path1  = save_to_path_base + '/init_mcmc'
	save_to_path   = save_to_path_base + '/final_mcmc'

else:
	save_to_path   = save_to_path_base + '/init_mcmc'

# date
today     = date.today()
now       = datetime.now()
dt_string = now.strftime("%H:%M:%S")	

#####################################

print('MASK', applymask)
print(sci_data_name)
print(data_path)
print(data_path + sci_data_name + '.fits')
#sys.exit()

spectrum = splat.Spectrum(file='nires_J1259+0651A_20250516.fits', instrument='NIRES')
#spectrum = splat.Spectrum(file='nires_J1259+0651B_20250516.fits', instrument='NIRES')

wave  = spectrum.wave.value * 10000 # convert microns to angstroms
flux  = spectrum.flux.value  ### NEED TO CONVERT THESE UNITS!
noise = spectrum.noise.value  ### NEED TO CONVERT THESE UNITS!

mask   = np.where( (np.isnan(flux) ) )[0]
wave   = wave  * u.angstrom
flux   = flux  * u.erg/u.cm**2/u.micron/u.s 
noise  = noise * u.erg/u.cm**2/u.micron/u.s 

# Convert to correct units
flux  = flux.to(u.erg/u.s/u.cm**2/u.angstrom)
noise = noise.to(u.erg/u.s/u.cm**2/u.angstrom)

data          = smart.Spectrum(flux=flux.value, wave=wave.value, noise=noise.value, order=order, instrument=instrument)
data.wave     = data.wave.flatten()
data.oriWave  = data.wave.flatten()
data.flux     = data.flux.flatten()
data.oriFlux  = data.flux.flatten()
data.noise    = data.noise.flatten()
data.oriNoise = data.noise.flatten()

sci_data  = data



"""
MCMC run for the science spectra. See the parameters in the makeModel function.

Parameters
----------

sci_data  	: 	sepctrum object
				science data

tell_data 	: 	spectrum object
				telluric data for calibrating the science spectra

priors   	: 	dic
				keys are teff_min, teff_max, logg_min, logg_max, vsini_min, vsini_max, rv_min, rv_max, 
				am_min, am_max, pwv_min, pwv_max, A_min, A_max, B_min, B_max

Optional Parameters
-------------------

limits 		:	dic
					mcmc limits with the same format as the input priors

ndim 		:	int
				mcmc dimension

nwalkers 	:	int
					number of walkers

	step 		: 	int
					number of steps

	burn 		:	int
					burn for the mcmc

	moves 		: 	float
					stretch parameter for the mcmc. The default is 2.0 based on the emcee package

	pixel_start	:	int
					starting pixel number for the spectra in the MCMC

	pixel_end	:	int
					ending pixel number for the spectra in the MCMC

	alpha_tell	:	float
					power of telluric spectra for estimating the line spread function of the NIRSPEC instrument

	modelset 	:	str
					'btsettl08' or 'phoenixaces' model sets

	save_to_path: 	str
					path to savr the MCMC output				

"""

if save_to_path is not None:
	if not os.path.exists(save_to_path):
		os.makedirs(save_to_path)
else:
	save_to_path = '.'

#if limits is None: limits = priors

data          = copy.deepcopy(sci_data)


# barycentric corrction
barycorr    = smart.barycorr(spectrum.header, instrument=instrument).value
print("BARYCORR:",barycorr)

## read the input custom mask and priors
lines          = open(save_to_path+'/mcmc_parameters.txt').read().splitlines()
custom_mask    = json.loads(lines[3].split('custom_mask')[1])
priors         = ast.literal_eval(lines[4].split('priors ')[1])
barycorr       = json.loads(lines[11].split('barycorr')[1])


# no logg 5.5 for teff lower than 900
if 'btsettl08' in modelset.lower() and priors['teff_min'] < 900: logg_max = 5.0
else: logg_max = 5.5

# limit of the flux nuisance parameter: 5 percent of the median flux
A_const       = 0.05 * abs(np.median(data.flux))
A_const       = 100

if 'btsettl08' in modelset.lower():
	limits         = { 
						'teff_min':max(priors['teff_min']-300,500), 'teff_max':min(priors['teff_max']+300,3500),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':10.0,
						'rv_min':-1000.0,                           'rv_max':1000.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':20.0,
						'A_min':-50,							    'A_max':50,
						'B_min':-50,                              	'B_max':50,
						'N_min':0.10,                               'N_max':10.0,
						'lsf_min':1,                                'lsf_max':200 				
					}

elif modelset.lower() == 'phoenix-btsettl-cifist2011-2015':
	limits         = { 
						'teff_min':max(priors['teff_min']-300,1200),'teff_max':min(priors['teff_max']+300,3500),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':10.0,
						'rv_min':-1000.0,                           'rv_max':1000.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':20.0,
						'A_min':-50,							    'A_max':50,
						'B_min':-50,                              	'B_max':50,
						'N_min':0.10,                               'N_max':10.0,
						'lsf_min':1,                                'lsf_max':200 				
					}


elif modelset.lower() == 'sonora-2024':
	limits         = { 
						'teff_min':max(priors['teff_min']-300,900), 'teff_max':min(priors['teff_max']+300,2400),
						'logg_min':3.5,                             'logg_max':5.5,
						'metal_min':-0.5,                           'metal_max':0.5,
						'fsed_min':1,                               'fsed_max':20,
						'vsini_min':0.0,                            'vsini_max':100.0,
						'rv_min':-1000.0,                           'rv_max':1000.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':20.0,
						'A_min':-50,								'A_max':50,
						'B_min':-0.6,								'B_max':0.6,
						'N_min':0.10,                               'N_max':2.0, 	
				        'lsf_min':5,                                'lsf_max':400, # see JWST/NIRSpec resolving power plot			
					}

elif 'sonora' in modelset.lower():
	limits         = { 
						'teff_min':max(priors['teff_min']-300,200), 'teff_max':min(priors['teff_max']+300,2400),
						'logg_min':3.5,                             'logg_max':logg_max,
						'vsini_min':0.0,                            'vsini_max':10.0,
						'rv_min':-1000.0,                           'rv_max':1000.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':20.0,
						'A_min':-50,								'A_max':50,
						'B_min':-50,                              	'B_max':50,
						'N_min':0.10,                               'N_max':10.0,
						'lsf_min':1,                                'lsf_max':200 				
					}

elif modelset.lower() == 'phoenix-aces-agss-cond-2011':
	limits         = { 
						'teff_min':max(priors['teff_min']-300,2300), 'teff_max':min(priors['teff_max']+300,10000),
						'logg_min':3.5,                             'logg_max':logg_max,
						'metal_min':-4,                             'metal_max':1.,
						'vsini_min':0.0,                            'vsini_max':100.0,
						'rv_min':-200.0,                            'rv_max':200.0,
						'am_min':1.0,                               'am_max':3.0,
						'pwv_min':0.5,                            	'pwv_max':5.0,
						'A_min':-50,								'A_max':50,
						'B_min':-0.6,								'B_max':0.6,
						'N_min':0.10,                               'N_max':10.0, 	
						'lsf_min':1,                                'lsf_max':200 				
					}



# HIRES wavelength calibration is not that precise, release the constraint for the wavelength offset nuisance parameter
if data.instrument == 'hires':
	limits['B_min'] = -3.0 # Angstrom
	limits['B_max'] = +3.0 # Angstrom

if final_mcmc:
	limits['rv_min'] = priors['rv_min'] - 10
	limits['rv_max'] = priors['rv_max'] + 10

### mask the end pixels
data.wave      = data.wave[pixel_start:pixel_end]
data.flux      = data.flux[pixel_start:pixel_end]
data.noise     = data.noise[pixel_start:pixel_end]
data.oriWave   = data.oriWave[pixel_start:pixel_end]
data.oriFlux   = data.oriFlux[pixel_start:pixel_end]
data.oriNoise  = data.oriNoise[pixel_start:pixel_end]

## add a pixel label for plotting
length1     = len(data.wave)
pixel       = np.delete(np.arange(length1),custom_mask)
print('MASK:', data.mask, len(data.mask))
print('PIXELS:', pixel_start, pixel_end)


## apply a custom mask
print('CUSTOM_MASK', custom_mask)
data.mask_custom(custom_mask=custom_mask)


# log file
log_path = save_to_path + '/mcmc_parameters.txt'


#########################################################################################
## for multiprocessing
#########################################################################################
print('Modelset:', modelset)
print('Instrument:', instrument)
print('include_fringe_model', include_fringe_model)
print(lsf0)

def lnlike(theta, data, lsf0):
	"""
	Log-likelihood, computed from chi-squared.

	Parameters
	----------
	theta
	lsf
	data

	Returns
	-------
	-0.5 * chi-square + sum of the log of the noise

	"""

	## Parameters MCMC
	#print('THETA:', theta)
	teff, logg, metal, rv, N, lsf = theta #N noise prefactor
	#teff, logg, vsini, rv, , am, pwv, A, B, freq, amp, phase = theta
	
	#print('DATA')
	#print(data.wave)
	#print(data.flux)
	#print(data.wave.shape)
	#print(mask)
	#print('1')
	model = smart.makeModel(teff=teff, logg=logg, metal=metal, rv=rv, #flux_mult=A, #wave_offset=B, 
		                        lsf=lsf, order=str(data.order), data=data, modelset=modelset,
		                        include_fringe_model=include_fringe_model, instrument=instrument, tell=False)
	#print('2')
	#print('MODEL')
	#print(model.wave)
	#print(model.flux)
	#print(model.wave.shape)

	#plt.figure(1, figsize=(10,5))
	#plt.plot(data.wave, data.flux, alpha=0.5, label='data')
	#plt.plot(model.wave, model.flux, alpha=0.5, label='model')
	#plt.plot(model.wave, data.flux-model.flux, alpha=0.5, label='O-C')
	#plt.legend()
	
	#print('CHI0')
	chisquare = smart.chisquare(data, model)/N**2
	#print('3')
	#print(-0.5 * (chisquare + np.sum(np.log(2*np.pi*(data.noise)**2))))
	#plt.pause(0.2)
	#plt.close(1)
	#chisquare = smart.chisquare(data, model)

	return -0.5 * (chisquare + np.sum(np.log(2*np.pi*(data.noise)**2)))
	

def lnprior(theta, limits=limits):
	"""
	Specifies a flat prior
	"""
	## Parameters for theta
	teff, logg, metal, rv, N, lsf = theta

	if  limits['teff_min']  < teff  < limits['teff_max'] \
	and limits['logg_min']  < logg  < limits['logg_max'] \
	and limits['metal_min'] < metal < limits['metal_max'] \
	and limits['rv_min']    < rv    < limits['rv_max']   \
	and limits['N_min']     < N     < limits['N_max'] \
	and limits['lsf_min']   < lsf   < limits['lsf_max']:
		return 0.0

	return -np.inf

def lnprob(theta, data, lsf0):
	#print('THETA0:', theta)
		
	lnp = lnprior(theta)
		
	if not np.isfinite(lnp):
		return -np.inf
		
	return lnp + lnlike(theta, data, lsf0)

pos = [np.array([	priors['teff_min']  + (priors['teff_max']   - priors['teff_min'] ) * np.random.uniform(), 
					priors['logg_min']  + (priors['logg_max']   - priors['logg_min'] ) * np.random.uniform(), 
					priors['metal_min'] + (priors['metal_max']  - priors['metal_min'] ) * np.random.uniform(),
					#priors['vsini_min'] + (priors['vsini_max']  - priors['vsini_min']) * np.random.uniform(),
					priors['rv_min']    + (priors['rv_max']     - priors['rv_min']   ) * np.random.uniform(), 
					#priors['A_min']     + (priors['A_max']      - priors['A_min'])     * np.random.uniform(),
					priors['N_min']     + (priors['N_max']      - priors['N_min'])     * np.random.uniform(),
					#priors['B_min']     + (priors['B_max']      - priors['B_min'])     * np.random.uniform(),
					priors['lsf_min']   + (priors['lsf_max']    - priors['lsf_min'])   * np.random.uniform()
					]) for i in range(nwalkers)]


print('Priors:',priors)
print('Limits:',limits)

## single processing

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf0), a=moves, moves=emcee.moves.KDEMove())
time1 = time.time()
sampler.run_mcmc(pos, step, progress=True)
time2 = time.time()
'''
## multiprocessing

set_start_method('fork')
with Pool() as pool:
	#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf, pwv), a=moves, pool=pool)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data, lsf0), a=moves, pool=pool,
			moves=emcee.moves.KDEMove()
			)
	time1 = time.time()
	sampler.run_mcmc(pos, step, progress=True)
	time2 = time.time()
'''

np.save(save_to_path + '/sampler_chain', sampler.chain[:, :, :])

samples = sampler.chain[:, :, :].reshape((-1, ndim))

np.save(save_to_path + '/samples', samples)

print('total time: ',(time2-time1)/60,' min.')
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
print(sampler.acceptance_fraction)
autocorr_time = sampler.get_autocorr_time(discard=burn, quiet=True)
print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(autocorr_time)))
print(autocorr_time)

# create plots
print('SAVE PATH:', save_to_path)
sampler_chain = np.load(save_to_path + '/sampler_chain.npy')
samples       = np.load(save_to_path + '/samples.npy')

ylabels = ["$T_{\mathrm{eff}} (K)$",
           "$\log{g}$(dex)",
           "[M/H]",
           #"vsini",
           "$RV(km/s)$",
           #"$C_{F_{\lambda}}$ (cnt/s)",
           "$C_\mathrm{Noise}$",
           #"PWV",
           #"AM",
           "LSF"
           ]


## create walker plots
print('Creating walker plot')
plt.rc('font', family='sans-serif')
plt.tick_params(labelsize=10)
fig = plt.figure(tight_layout=True)
gs  = gridspec.GridSpec(ndim, 1)
gs.update(hspace=0.1)

for i in range(ndim):
	ax = fig.add_subplot(gs[i, :])
	for j in range(nwalkers):
		ax.plot(np.arange(1,int(step+1)), sampler_chain[j,:,i],'k',alpha=0.2)
		ax.set_ylabel(ylabels[i])
fig.align_labels()
plt.minorticks_on()
plt.xlabel('nstep')
plt.savefig(save_to_path+'/walker.png', dpi=300, bbox_inches='tight')
if plot_show:
	plt.show()
plt.close()

# create array triangle plots
triangle_samples = sampler_chain[:, burn:, :].reshape((-1, ndim))
#print(triangle_samples.shape)

# create the final spectra comparison
teff_mcmc, logg_mcmc, metal_mcmc, rv_mcmc, N_mcmc, lsf_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
	zip(*np.percentile(triangle_samples, [16, 50, 84], axis=0)))


print(teff_mcmc, logg_mcmc, metal_mcmc, rv_mcmc, N_mcmc, lsf_mcmc)

# add the summary to the txt file

# log file
log_path2 = save_to_path + '/mcmc_result.txt'

file_log2 = open(log_path2,"w+")
file_log2.write("teff_mcmc {}\n".format(str(teff_mcmc[0])))
file_log2.write("logg_mcmc {}\n".format(str(logg_mcmc[0])))
file_log2.write("metal_mcmc {}\n".format(str(metal_mcmc[0])))
#file_log2.write("fsed_mcmc {}\n".format(str(fsed_mcmc[0])))
#file_log2.write("vsini_mcmc {}\n".format(str(vsini_mcmc[0])))
file_log2.write("rv_mcmc {}\n".format(str(rv_mcmc[0]+barycorr)))
file_log2.write("N_mcmc {}\n".format(str(N_mcmc[0])))
#file_log2.write("pwv_mcmc {}\n".format(str(pwv_mcmc[0])))
#file_log2.write("am_mcmc {}\n".format(str(am_mcmc[0])))
#file_log2.write("A_mcmc {}\n".format(str(A_mcmc[0])))
#file_log2.write("B_mcmc {}\n".format(str(B_mcmc[0])))
file_log2.write("lsf_mcmc {}\n".format(str(lsf_mcmc[0])))
file_log2.write("teff_mcmc_e {}\n".format(str(max(abs(teff_mcmc[1]), abs(teff_mcmc[2])))))
file_log2.write("logg_mcmc_e {}\n".format(str(max(abs(logg_mcmc[1]), abs(logg_mcmc[2])))))
file_log2.write("metal_mcmc_e {}\n".format(str(max(abs(metal_mcmc[1]), abs(metal_mcmc[2])))))
#file_log2.write("fsed_mcmc_e {}\n".format(str(max(abs(fsed_mcmc[1]), abs(fsed_mcmc[2])))))
#file_log2.write("vsini_mcmc_e {}\n".format(str(max(abs(vsini_mcmc[1]), abs(vsini_mcmc[2])))))
file_log2.write("rv_mcmc_e {}\n".format(str(max(abs(rv_mcmc[1]), abs(rv_mcmc[2])))))
file_log2.write("N_mcmc_e {}\n".format(str(max(abs(N_mcmc[1]), abs(N_mcmc[2])))))
#file_log2.write("pwv_mcmc_e {}\n".format(str(max(abs(pwv_mcmc[1]), abs(pwv_mcmc[2])))))
#file_log2.write("am_mcmc_e {}\n".format(str(max(abs(am_mcmc[1]), abs(am_mcmc[2])))))
#file_log2.write("A_mcmc_e {}\n".format(str(max(abs(A_mcmc[1]), abs(A_mcmc[2])))))
#file_log2.write("B_mcmc_e {}\n".format(str(max(abs(B_mcmc[1]), abs(B_mcmc[2])))))
file_log2.write("lsf_mcmc_e {}\n".format(str(max(abs(lsf_mcmc[1]), abs(lsf_mcmc[2])))))
# upper and lower uncertainties
# upper uncertainties
file_log2.write("teff_mcmc_ue {}\n".format(str(abs(teff_mcmc[1]))))
file_log2.write("logg_mcmc_ue {}\n".format(str(abs(logg_mcmc[1]))))
file_log2.write("metal_mcmc_ue {}\n".format(str(abs(metal_mcmc[1]))))
#file_log2.write("fsed_mcmc_ue {}\n".format(str(abs(fsed_mcmc[1]))))
#file_log2.write("vsini_mcmc_ue {}\n".format(str(abs(vsini_mcmc[1]))))
file_log2.write("rv_mcmc_ue {}\n".format(str(abs(rv_mcmc[1]))))
file_log2.write("N_mcmc_ue {}\n".format(str(abs(N_mcmc[1]))))
#file_log2.write("pwv_mcmc_ue {}\n".format(str(abs(pwv_mcmc[1]))))
#file_log2.write("am_mcmc_ue {}\n".format(str(abs(am_mcmc[1]))))
#file_log2.write("A_mcmc_ue {}\n".format(str(abs(A_mcmc[1]))))
#file_log2.write("B_mcmc_ue {}\n".format(str(abs(B_mcmc[1]))))
file_log2.write("lsf_mcmc_ue {}\n".format(str(abs(lsf_mcmc[1]))))
# lower uncertainties
file_log2.write("teff_mcmc_le {}\n".format(str(abs(teff_mcmc[2]))))
file_log2.write("logg_mcmc_le {}\n".format(str(abs(logg_mcmc[2]))))
file_log2.write("metal_mcmc_le {}\n".format(str(abs(metal_mcmc[2]))))
#file_log2.write("fsed_mcmc_le {}\n".format(str(abs(fsed_mcmc[2]))))
#file_log2.write("vsini_mcmc_le {}\n".format(str(abs(vsini_mcmc[2]))))
file_log2.write("rv_mcmc_le {}\n".format(str(abs(rv_mcmc[2]))))
file_log2.write("N_mcmc_le {}\n".format(str(abs(N_mcmc[2]))))
#file_log2.write("pwv_mcmc_le {}\n".format(str(abs(pwv_mcmc[2]))))
#file_log2.write("am_mcmc_le {}\n".format(str(abs(am_mcmc[2]))))
#file_log2.write("A_mcmc_le {}\n".format(str(abs(A_mcmc[2]))))
#file_log2.write("B_mcmc_le {}\n".format(str(abs(B_mcmc[2]))))
file_log2.write("lsf_mcmc_le {}\n".format(str(abs(lsf_mcmc[2]))))
file_log2.close()

# Correct the walkers for barycentric motion
triangle_samples[:,3] += barycorr

## triangular plots
print('Creating corner plot')
plt.rc('font', family='sans-serif')
fig = corner.corner(triangle_samples, 
	labels=ylabels,
	truths=[teff_mcmc[0], 
	logg_mcmc[0], 
	metal_mcmc[0], 
	#vsini_mcmc[0], 
	rv_mcmc[0]+barycorr, 
	#A_mcmc[0],
	N_mcmc[0],
	#pwv_mcmc[0],
	#am_mcmc[0],
	lsf_mcmc[0]
	],
	quantiles=[0.16, 0.84],
	label_kwargs={"fontsize": 20})
plt.minorticks_on()
fig.savefig(save_to_path+'/triangle.png', dpi=300, bbox_inches='tight')
if plot_show:
	plt.show()
plt.close()

teff  = teff_mcmc[0]
logg  = logg_mcmc[0]
metal = metal_mcmc[0]
#fsed  = fsed_mcmc[0]
#vsini = vsini_mcmc[0]
rv    = rv_mcmc[0]
#A     = A_mcmc[0]
#B     = B_mcmc[0]
N     = N_mcmc[0]
#pwv   = pwv_mcmc[0]
#am    = am_mcmc[0]
lsf   = lsf_mcmc[0]

print('Creating model and data plot')
model = model_fit.makeModel(teff=teff, logg=logg, metal=metal, rv=rv, #flux_mult=A,
	lsf=lsf, order=str(data.order), data=data, modelset=modelset, 
	include_fringe_model=False, instrument=instrument, tell=False)


fig = plt.figure(figsize=(16,6))
gs  = gridspec.GridSpec(2,1, height_ratios=[3,1])
#ax1 = fig.add_subplot(211)
#ax3 = fig.add_subplot(212)
ax1 = fig.add_subplot(gs[0])
ax3 = fig.add_subplot(gs[1])
plt.rc('font', family='sans-serif')
plt.tick_params(labelsize=15)
diff = data.flux-model.flux
ax1.plot(model.wave, model.flux, color='C3', linestyle='-', label='model',alpha=0.8)
#ax1.plot(model_notell.wave,model_notell.flux, color='C0', linestyle='-', label='model no telluric',alpha=0.8)
ax1.plot(data.wave, data.flux,'k-', label='data',alpha=0.5)
#N=1
ax3.fill_between(data.wave,-data.noise*N,data.noise*N,facecolor='C0',alpha=0.5)
ax3.fill_between(data.wave,-data.noise,data.noise,facecolor='C1',alpha=0.5)
ax3.axhline(y=0, color='k', linestyle=':',linewidth=0.5)
#plt.ylim(-np.max(np.append(np.abs(data.noise),np.abs(data.flux-model.flux)))*1.2,np.max(data.flux)*1.2)
ax1.set_ylabel("Flux ($cnts/s$)",fontsize=15)
ax3.set_ylabel("Residuals ($cnts/s$)",fontsize=15)
ax3.set_xlabel("$\lambda$ ($\AA$)",fontsize=15)
#plt.figtext(0.89,0.85,str(data.header['OBJECT'])+' '+data.name+' O'+str(data.order),
#	color='k',
#	horizontalalignment='right',
#	verticalalignment='center',
#	fontsize=15)
plt.figtext(0.89,0.82,"$Teff \, {0}^{{+{1}}}_{{-{2}}}/ logg \, {3}^{{+{4}}}_{{-{5}}}/ [M/H] \, {6}^{{+{7}}}_{{-{8}}}/ RV \, {9}^{{+{10}}}_{{-{11}}}$".format(\
	round(teff_mcmc[0]),
	round(teff_mcmc[1]),
	round(teff_mcmc[2]),
	round(logg_mcmc[0],1),
	round(logg_mcmc[1],3),
	round(logg_mcmc[2],3),
	round(metal_mcmc[0],1),
	round(metal_mcmc[1],3),
	round(metal_mcmc[2],3),
	round(rv_mcmc[0]+barycorr,2),
	round(rv_mcmc[1],2),
	round(rv_mcmc[2],2)),
	color='C0',
	horizontalalignment='right',
	verticalalignment='center',
	fontsize=12)
plt.figtext(0.89,0.79,r"$\chi^2$ = {}, DOF = {}".format(\
	round(smart.chisquare(data,model)), round(len(data.wave-ndim)/3)),
color='k',
horizontalalignment='right',
verticalalignment='center',
fontsize=12)
ax3.plot(data.wave, diff,'k-',alpha=0.8, label='residual')
ax1.legend(loc=4)
ax1.minorticks_on()
ax3.minorticks_on()

ax2 = ax1.twiny()
ax2.plot(pixel, data.flux, color='w', alpha=0)
ax2.set_xlabel('Pixel',fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xlim(pixel[0], pixel[-1])
ax2.minorticks_on()
	
#plt.legend()
plt.savefig(save_to_path + '/spectrum.png', dpi=300, bbox_inches='tight')
plt.savefig(save_to_path + '/spectrum.pdf', bbox_inches='tight')
if plot_show:
	plt.show()
plt.close()

# chi2 and dof in the log
log_path = save_to_path + '/mcmc_parameters.txt'
file_log = open(log_path,"a")
file_log.write("*** Below is the summary *** \n")
file_log.write("total_time {} min\n".format(str((time2-time1)/60)))
file_log.write("mean_acceptance_fraction {0:.3f} \n".format(np.mean(sampler.acceptance_fraction)))
file_log.write("mean_autocorrelation_time {0:.3f} \n".format(np.mean(autocorr_time)))
file_log.write("chi2 {} \n".format(round(smart.chisquare(data,model))))
file_log.write("dof {} \n".format(round(len(data.wave-ndim)/3)))
file_log.write("teff_mcmc {} K\n".format(str(teff_mcmc)))
file_log.write("logg_mcmc {} dex (cgs)\n".format(str(logg_mcmc)))
file_log.write("metal_mcmc {} dex\n".format(str(metal_mcmc)))
#file_log.write("fsed_mcmc {} dex\n".format(str(fsed_mcmc)))
#file_log.write("vsini_mcmc {} km/s\n".format(str(vsini_mcmc)))
file_log.write("rv_mcmc {} km/s\n".format(str(rv_mcmc)))
file_log.write("N_mcmc {}\n".format(str(N_mcmc)))
#file_log.write("am_mcmc {}\n".format(str(am_mcmc)))
#file_log.write("pwv_mcmc {}\n".format(str(pwv_mcmc)))
#file_log.write("A_mcmc {}\n".format(str(A_mcmc)))
#file_log.write("B_mcmc {}\n".format(str(B_mcmc)))
file_log.write("lsf_mcmc {}\n".format(str(lsf_mcmc)))
file_log.close()


med_snr      = np.nanmedian(data.flux/data.noise)
if instrument == 'nirspec':
	wave_cal_err = tell_sp.header['STD']
else:
	wave_cal_err = np.nan

