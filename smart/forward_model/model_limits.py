"""
Model Limits Module

This module provides model parameter limits for different modelsets.
The limits are used in MCMC fitting routines to constrain the parameter space.

The modelset name is checked against the lowercase version for case-insensitive matching.
"""


def get_model_limits(modelset, priors, logg_max, A_const=100, N_max=20.0):
    """
    Get model parameter limits based on the modelset name.
    
    The function checks the modelset name against its lowercase version for matching.
    
    Parameters
    ----------
    modelset : str
        Name of the model set (checked case-insensitively against lowercase)
    priors : dict
        Dictionary containing prior values with keys like 'teff_min', 'teff_max', etc.
    logg_max : float
        Maximum log(g) value to use
    A_const : float, optional
        Constant for flux nuisance parameter limits (default: 100)
    N_max : float, optional
        Maximum noise scaling parameter (default: 20.0)
    
    Returns
    -------
    limits : dict
        Dictionary containing parameter limits with keys like 'teff_min', 'teff_max', etc.
        Returns None if modelset is not recognized.
    
    Examples
    --------
    >>> priors = {'teff_min': 2500, 'teff_max': 3000}
    >>> limits = get_model_limits('btsettl08', priors, logg_max=5.5)
    >>> limits = get_model_limits('BTSETTL08', priors, logg_max=5.5)  # case-insensitive
    """
    
    # Normalize modelset name to lowercase for comparison
    modelset_lower = modelset.lower()
    modelset_upper = modelset.upper()
    
    limits = None
    
    # BT-Settl 2008 models
    if 'btsettl08' in modelset_lower:
        limits = {
            'teff_min': max(priors['teff_min'] - 300, 500),
            'teff_max': min(priors['teff_max'] + 300, 6000),
            'logg_min': 3.5,
            'logg_max': logg_max,
            'vsini_min': 0.0,
            'vsini_max': 100.0,
            'rv_min': -200.0,
            'rv_max': 200.0,
            'am_min': 1.0,
            'am_max': 3.0,
            'pwv_min': 0.5,
            'pwv_max': 20.0,
            'A_min': -5,
            'A_max': 5,
            'B_min': -0.6,
            'B_max': 0.6,
            'N_min': 0.10,
            'N_max': N_max
        }
    
    # Sonora models (general)
    elif modelset_lower == 'sonora':
        limits = {
            'teff_min': max(priors['teff_min'] - 300, 200),
            'teff_max': min(priors['teff_max'] + 300, 2400),
            'logg_min': 3.5,
            'logg_max': logg_max,
            'vsini_min': 0.0,
            'vsini_max': 100.0,
            'rv_min': -200.0,
            'rv_max': 200.0,
            'am_min': 1.0,
            'am_max': 3.0,
            'pwv_min': 0.5,
            'pwv_max': 20.0,
            'A_min': -A_const,
            'A_max': A_const,
            'B_min': -0.6,
            'B_max': 0.6,
            'N_min': 0.10,
            'N_max': N_max
        }
    
    # Sonora 2024 models
    elif modelset_lower == 'sonora-2024':
        limits = {
            'teff_min': max(priors['teff_min'] - 300, 900),
            'teff_max': min(priors['teff_max'] + 300, 2400),
            'logg_min': 3.5,
            'logg_max': 5.5,
            'metal_min': -0.5,
            'metal_max': 0.5,
            'fsed_min': 1,
            'fsed_max': 20,
            'vsini_min': 0.0,
            'vsini_max': 100.0,
            'rv_min': -1000.0,
            'rv_max': 1000.0,
            'am_min': 1.0,
            'am_max': 3.0,
            'pwv_min': 0.5,
            'pwv_max': 20.0,
            'A_min': -50,
            'A_max': 50,
            'B_min': -0.6,
            'B_max': 0.6,
            'N_min': 0.10,
            'N_max': 2.0,
            'lsf_min': 5,
            'lsf_max': 400  # see JWST/NIRSpec resolving power plot
        }
    
    # PHOENIX ACES models
    elif modelset_lower == 'phoenixaces':
        limits = {
            'teff_min': max(priors['teff_min'] - 300, 2300),
            'teff_max': min(priors['teff_max'] + 300, 10000),
            'logg_min': 3.5,
            'logg_max': logg_max,
            'vsini_min': 0.0,
            'vsini_max': 100.0,
            'rv_min': -200.0,
            'rv_max': 200.0,
            'am_min': 1.0,
            'am_max': 3.0,
            'pwv_min': 0.5,
            'pwv_max': 20.0,
            'A_min': -A_const,
            'A_max': A_const,
            'B_min': -0.6,
            'B_max': 0.6,
            'N_min': 0.10,
            'N_max': N_max
        }
    
    # PHOENIX BT-SETTL CIFIST 2011-2015 models
    elif (modelset_lower == 'phoenix-btsettl-cifist2011-2015' or 
          modelset_upper == 'PHOENIX-BTSETTL-CIFIST2011-2015' or
          modelset_upper == 'PHOENIX_BTSETTL_CIFIST2011_2015'):
        limits = {
            'teff_min': max(priors['teff_min'] - 300, 2300),
            'teff_max': min(priors['teff_max'] + 300, 7000),
            'logg_min': 3.5,
            'logg_max': logg_max,
            'vsini_min': 0.0,
            'vsini_max': 100.0,
            'rv_min': -200.0,
            'rv_max': 200.0,
            'am_min': 1.0,
            'am_max': 3.0,
            'pwv_min': 0.5,
            'pwv_max': 20.0,
            'A_min': -A_const,
            'A_max': A_const,
            'B_min': -0.6,
            'B_max': 0.6,
            'N_min': 0.10,
            'N_max': N_max
        }
    
    # PHOENIX ACES AGSS COND 2011 models
    elif modelset_lower == 'phoenix-aces-agss-cond-2011':
        limits = {
            'teff_min': max(priors['teff_min'] - 300, 2300),
            'teff_max': min(priors['teff_max'] + 300, 10000),
            'logg_min': 3.5,
            'logg_max': logg_max,
            'metal_min': -4,
            'metal_max': 1.0,
            'vsini_min': 0.0,
            'vsini_max': 100.0,
            'rv_min': -200.0,
            'rv_max': 200.0,
            'am_min': 1.0,
            'am_max': 3.0,
            'pwv_min': 0.5,
            'pwv_max': 20.0,
            'A_min': -50,
            'A_max': 50,
            'B_min': -0.6,
            'B_max': 0.6,
            'N_min': 0.10,
            'N_max': 2.0,
            'lsf_min': 1,
            'lsf_max': 200
        }
    
    # HD13724B G395H specific model (case-sensitive match)
    elif modelset_upper == 'HD13724B_G395H':
        limits = {
            'teff_min': 1000,
            'teff_max': 1000,
            'logg_min': 5,
            'logg_max': 5,
            'metal_min': 0,
            'metal_max': 0.5,
            'co_min': 0.5,
            'co_max': 2.5,
            'kzz_min': 10**2,
            'kzz_max': 10**9,
            'vsini_min': 0.0,
            'vsini_max': 100.0,
            'rv_min': -1000.0,
            'rv_max': 1000.0,
            'am_min': 1.0,
            'am_max': 3.0,
            'pwv_min': 0.5,
            'pwv_max': 20.0,
            'A_min': -22,
            'A_max': -17,
            'B_min': -0.6,
            'B_max': 0.6,
            'N_min': 0.10,
            'N_max': 2.0,
            'lsf_min': 5,
            'lsf_max': 200
        }
    
    # Catch-all for 'sonora' in name but not exact match (for backward compatibility)
    elif 'sonora' in modelset_lower and limits is None:
        limits = {
            'teff_min': max(priors['teff_min'] - 300, 200),
            'teff_max': min(priors['teff_max'] + 300, 2400),
            'logg_min': 3.5,
            'logg_max': logg_max,
            'vsini_min': 0.0,
            'vsini_max': 100.0,
            'rv_min': -200.0,
            'rv_max': 200.0,
            'am_min': 1.0,
            'am_max': 3.0,
            'pwv_min': 0.5,
            'pwv_max': 20.0,
            'A_min': -A_const,
            'A_max': A_const,
            'B_min': -0.6,
            'B_max': 0.6,
            'N_min': 0.10,
            'N_max': N_max
        }
    
    return limits


def update_limits_for_instrument(limits, instrument):
    """
    Update limits based on instrument-specific requirements.
    
    Parameters
    ----------
    limits : dict
        Dictionary of parameter limits
    instrument : str or object
        Instrument name (string) or data object with instrument attribute
        
    Returns
    -------
    limits : dict
        Updated limits dictionary
    """
    if limits is None:
        return limits
    
    # Handle both string and object with instrument attribute
    instrument_name = instrument
    if hasattr(instrument, 'lower'):
        instrument_name = instrument.lower()
    elif hasattr(instrument, 'instrument'):
        instrument_name = instrument.instrument.lower() if hasattr(instrument.instrument, 'lower') else str(instrument.instrument).lower()
    else:
        instrument_name = str(instrument).lower()
    
    # HIRES wavelength calibration is not that precise, 
    # release the constraint for the wavelength offset nuisance parameter
    if instrument_name == 'hires':
        limits['B_min'] = -3.0  # Angstrom
        limits['B_max'] = 3.0   # Angstrom
    
    return limits


def update_limits_for_final_mcmc(limits, priors):
    """
    Update RV limits for final MCMC run.
    
    Parameters
    ----------
    limits : dict
        Dictionary of parameter limits
    priors : dict
        Dictionary of prior values
        
    Returns
    -------
    limits : dict
        Updated limits dictionary
    """
    if limits is None:
        return limits
    
    if 'rv_min' in priors and 'rv_max' in priors:
        limits['rv_min'] = priors['rv_min'] - 10
        limits['rv_max'] = priors['rv_max'] + 10
    
    return limits
