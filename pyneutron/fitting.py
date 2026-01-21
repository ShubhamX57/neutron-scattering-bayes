"""
Fitting functions and utilities for neutron data analysis
"""

import numpy as np
from scipy.optimize import curve_fit

def lorentzian(x, A, x0, gamma, C):
    """Single Lorentzian function"""
    return A * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def double_lorentzian(x, A1, x01, gamma1, A2, x02, gamma2, C):
    """Double Lorentzian function"""
    lorentz1 = A1 * (gamma1/2)**2 / ((x - x01)**2 + (gamma1/2)**2)
    lorentz2 = A2 * (gamma2/2)**2 / ((x - x02)**2 + (gamma2/2)**2)
    return lorentz1 + lorentz2 + C

def gaussian(x, A, x0, sigma, C):
    """Gaussian function"""
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + C

def fit_spectrum(x_data, y_data, y_errors, func, p0, bounds, maxfev=5000):
    """
    Fit a spectrum with given function
    
    Parameters:
    -----------
    x_data : array
        X data (omega values)
    y_data : array
        Y data (intensity values)
    y_errors : array
        Error values
    func : callable
        Fitting function
    p0 : list
        Initial parameters
    bounds : tuple
        Parameter bounds
    maxfev : int
        Maximum number of function evaluations
        
    Returns:
    --------
    popt : array
        Optimized parameters
    perr : array
        Parameter errors
    """
    # Remove NaN and infinite values
    mask = np.isfinite(x_data) & np.isfinite(y_data) & (y_errors > 0)
    x_fit = x_data[mask]
    y_fit = y_data[mask]
    errors_fit = y_errors[mask]
    
    if len(x_fit) < 4:
        raise ValueError("Not enough valid data points for fitting")
    
    # Perform fit
    popt, pcov = curve_fit(func, x_fit, y_fit, 
                          p0=p0, sigma=errors_fit, 
                          bounds=bounds, maxfev=maxfev)
    
    # Calculate errors
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
    
    return popt, perr