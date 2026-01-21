"""
Utility functions for PyNeutron
"""

import pandas as pd
import numpy as np

def export_results(filename, data, fit_results):
    """
    Export fit results to CSV file
    
    Parameters:
    -----------
    filename : str
        Output filename
    data : dict
        Data dictionary
    fit_results : list
        List of fit results
    """
    # Prepare data for export
    q_values = []
    fit_params = []
    fit_errors = []
    
    for i, result in enumerate(fit_results):
        if result is not None:
            q_values.append(data['q'][i])
            popt, perr = result
            fit_params.append(popt)
            fit_errors.append(perr)
    
    if not q_values:
        raise ValueError("No valid fit results to export")
    
    # Create DataFrame
    num_params = len(fit_params[0])
    columns = ['Q']
    
    # Add parameter names
    param_names = ['Amplitude', 'Center', 'Width', 'Background']
    for i in range(num_params):
        name = param_names[i] if i < len(param_names) else f'Param_{i}'
        columns.append(name)
        columns.append(f'{name}_error')
    
    data_rows = []
    for q, params, errors in zip(q_values, fit_params, fit_errors):
        row = [q]
        for i in range(num_params):
            row.append(params[i] if i < len(params) else np.nan)
            row.append(errors[i] if i < len(errors) else np.nan)
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(filename, index=False, float_format='%.6f')