import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from forward_model.model_limits import get_model_limits, update_limits_for_instrument, update_limits_for_final_mcmc


def test_btsettl08_limits():
    """Test BT-Settl 2008 model limits"""
    priors = {'teff_min': 2500, 'teff_max': 3000}
    logg_max = 5.5
    
    limits = get_model_limits('btsettl08', priors, logg_max)
    
    assert limits is not None
    assert limits['teff_min'] == max(2500 - 300, 500)
    assert limits['teff_max'] == min(3000 + 300, 6000)
    assert limits['logg_min'] == 3.5
    assert limits['logg_max'] == 5.5
    assert limits['vsini_min'] == 0.0
    assert limits['vsini_max'] == 100.0
    assert limits['rv_min'] == -200.0
    assert limits['rv_max'] == 200.0


def test_btsettl08_case_insensitive():
    """Test that BT-Settl matching is case-insensitive"""
    priors = {'teff_min': 2500, 'teff_max': 3000}
    logg_max = 5.5
    
    limits_lower = get_model_limits('btsettl08', priors, logg_max)
    limits_upper = get_model_limits('BTSETTL08', priors, logg_max)
    limits_mixed = get_model_limits('BtSettl08', priors, logg_max)
    
    assert limits_lower == limits_upper == limits_mixed


def test_sonora_limits():
    """Test Sonora model limits"""
    priors = {'teff_min': 1000, 'teff_max': 1500}
    logg_max = 5.0
    A_const = 50
    
    limits = get_model_limits('sonora', priors, logg_max, A_const=A_const)
    
    assert limits is not None
    assert limits['teff_min'] == max(1000 - 300, 200)
    assert limits['teff_max'] == min(1500 + 300, 2400)
    assert limits['A_min'] == -A_const
    assert limits['A_max'] == A_const


def test_sonora_2024_limits():
    """Test Sonora 2024 model limits with additional parameters"""
    priors = {'teff_min': 1200, 'teff_max': 1800}
    logg_max = 5.5
    
    limits = get_model_limits('sonora-2024', priors, logg_max)
    
    assert limits is not None
    assert limits['teff_min'] == max(1200 - 300, 900)
    assert limits['teff_max'] == min(1800 + 300, 2400)
    assert limits['logg_max'] == 5.5  # Fixed for sonora-2024
    assert 'metal_min' in limits
    assert 'metal_max' in limits
    assert 'fsed_min' in limits
    assert 'fsed_max' in limits
    assert 'lsf_min' in limits
    assert 'lsf_max' in limits


def test_phoenixaces_limits():
    """Test PHOENIX ACES model limits"""
    priors = {'teff_min': 3000, 'teff_max': 4000}
    logg_max = 5.0
    A_const = 75
    
    limits = get_model_limits('phoenixaces', priors, logg_max, A_const=A_const)
    
    assert limits is not None
    assert limits['teff_min'] == max(3000 - 300, 2300)
    assert limits['teff_max'] == min(4000 + 300, 10000)
    assert limits['A_min'] == -A_const
    assert limits['A_max'] == A_const


def test_phoenix_btsettl_cifist_limits():
    """Test PHOENIX BT-SETTL CIFIST 2011-2015 model limits"""
    priors = {'teff_min': 3000, 'teff_max': 4000}
    logg_max = 5.0
    
    # Test different name formats
    limits1 = get_model_limits('phoenix-btsettl-cifist2011-2015', priors, logg_max)
    limits2 = get_model_limits('PHOENIX-BTSETTL-CIFIST2011-2015', priors, logg_max)
    limits3 = get_model_limits('PHOENIX_BTSETTL_CIFIST2011_2015', priors, logg_max)
    
    assert limits1 is not None
    assert limits2 is not None
    assert limits3 is not None
    assert limits1['teff_max'] == min(4000 + 300, 7000)


def test_phoenix_aces_agss_cond_limits():
    """Test PHOENIX ACES AGSS COND 2011 model limits"""
    priors = {'teff_min': 3000, 'teff_max': 4000}
    logg_max = 5.0
    
    limits = get_model_limits('phoenix-aces-agss-cond-2011', priors, logg_max)
    
    assert limits is not None
    assert 'metal_min' in limits
    assert 'metal_max' in limits
    assert limits['metal_min'] == -4
    assert limits['metal_max'] == 1.0
    assert 'lsf_min' in limits
    assert 'lsf_max' in limits


def test_unknown_modelset():
    """Test that unknown modelset returns None"""
    priors = {'teff_min': 2500, 'teff_max': 3000}
    logg_max = 5.5
    
    limits = get_model_limits('unknown_modelset', priors, logg_max)
    
    assert limits is None


def test_update_limits_for_hires():
    """Test HIRES instrument-specific limit updates"""
    limits = {
        'B_min': -0.6,
        'B_max': 0.6,
        'teff_min': 2000
    }
    
    updated_limits = update_limits_for_instrument(limits, 'hires')
    
    assert updated_limits['B_min'] == -3.0
    assert updated_limits['B_max'] == 3.0
    assert updated_limits['teff_min'] == 2000  # Other limits unchanged


def test_update_limits_for_other_instrument():
    """Test that other instruments don't modify limits"""
    limits = {
        'B_min': -0.6,
        'B_max': 0.6,
        'teff_min': 2000
    }
    
    updated_limits = update_limits_for_instrument(limits, 'nirspec')
    
    assert updated_limits['B_min'] == -0.6
    assert updated_limits['B_max'] == 0.6


def test_update_limits_for_final_mcmc():
    """Test RV limit updates for final MCMC"""
    limits = {
        'rv_min': -200.0,
        'rv_max': 200.0,
        'teff_min': 2000
    }
    priors = {
        'rv_min': -50.0,
        'rv_max': 50.0
    }
    
    updated_limits = update_limits_for_final_mcmc(limits, priors)
    
    assert updated_limits['rv_min'] == -60.0
    assert updated_limits['rv_max'] == 60.0
    assert updated_limits['teff_min'] == 2000  # Other limits unchanged


def test_n_max_parameter():
    """Test that N_max parameter is properly used"""
    priors = {'teff_min': 2500, 'teff_max': 3000}
    logg_max = 5.5
    N_max = 10.0
    
    limits = get_model_limits('btsettl08', priors, logg_max, N_max=N_max)
    
    assert limits['N_max'] == 10.0


def test_sonora_catch_all():
    """Test that sonora catch-all works for sonora variants"""
    priors = {'teff_min': 1000, 'teff_max': 1500}
    logg_max = 5.0
    
    # Test a sonora variant that's not explicitly defined
    limits = get_model_limits('sonora-mini', priors, logg_max)
    
    assert limits is not None
    assert limits['teff_min'] == max(1000 - 300, 200)
    assert limits['teff_max'] == min(1500 + 300, 2400)


if __name__ == '__main__':
    # Run all tests
    test_btsettl08_limits()
    test_btsettl08_case_insensitive()
    test_sonora_limits()
    test_sonora_2024_limits()
    test_phoenixaces_limits()
    test_phoenix_btsettl_cifist_limits()
    test_phoenix_aces_agss_cond_limits()
    test_unknown_modelset()
    test_update_limits_for_hires()
    test_update_limits_for_other_instrument()
    test_update_limits_for_final_mcmc()
    test_n_max_parameter()
    test_sonora_catch_all()
    print("All tests passed!")
