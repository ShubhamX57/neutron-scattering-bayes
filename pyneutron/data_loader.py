"""
Data loading utilities for NeXus/HDF5 files
"""

import h5py
import numpy as np

def load_nexus_file(filename):
    """
    Load data from NeXus/HDF5 file
    
    Parameters:
    -----------
    filename : str
        Path to the NeXus/HDF5 file
        
    Returns:
    --------
    dict
        Dictionary containing data arrays
    """
    with h5py.File(filename, 'r') as f:
        # Common NeXus/MANTID naming patterns
        possible_q_names = ['q', 'Q', 'momentum', 'momentum_transfer', 'x', 'qx']
        possible_omega_names = ['omega', 'energy', 'Energy', 'E', 'y', 'energy_transfer']
        possible_data_names = ['S_data', 'data', 'intensity', 'counts', 'signal', 'z']
        possible_error_names = ['errors', 'S_errors', 'error', 'variance', 'sigma']
        
        # Search through the file structure
        def find_dataset(f, possible_names, default=None):
            for name in possible_names:
                if name in f:
                    return f[name][:]
            # Try recursive search
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    for possible in possible_names:
                        if possible.lower() in key.lower():
                            return f[key][:]
            return default
        
        # Try to load data
        q_data = find_dataset(f, possible_q_names)
        omega_data = find_dataset(f, possible_omega_names)
        s_data = find_dataset(f, possible_data_names)
        
        if q_data is None or omega_data is None or s_data is None:
            # Try to find data in entry structure
            if 'entry' in f:
                entry = f['entry']
                q_data = find_dataset(entry, possible_q_names)
                omega_data = find_dataset(entry, possible_omega_names)
                s_data = find_dataset(entry, possible_data_names)
        
        if q_data is None or omega_data is None or s_data is None:
            raise ValueError("Could not find required datasets in file")
        
        # Try to find errors
        errors_data = find_dataset(f, possible_error_names, None)
        if errors_data is None and 'entry' in f:
            errors_data = find_dataset(f['entry'], possible_error_names, None)
        
        # Ensure data has correct shape
        if len(s_data.shape) == 2:
            # Check if dimensions match
            if s_data.shape[0] != len(q_data):
                # Transpose if necessary
                s_data = s_data.T
                if errors_data is not None and errors_data.shape == s_data.shape:
                    errors_data = errors_data.T
        
        # Create data dictionary
        data = {
            'q': q_data.flatten() if len(q_data.shape) > 1 else q_data,
            'omega': omega_data.flatten() if len(omega_data.shape) > 1 else omega_data,
            'S_data': s_data,
            'S_errors': errors_data if errors_data is not None else np.ones_like(s_data) * np.sqrt(np.abs(s_data) + 0.01),
            'filename': filename
        }
        
        # Ensure arrays are 1D
        if len(data['q'].shape) > 1:
            data['q'] = data['q'].flatten()
        if len(data['omega'].shape) > 1:
            data['omega'] = data['omega'].flatten()
        
        # Validate dimensions
        if len(data['S_data'].shape) != 2:
            raise ValueError(f"Expected 2D data, got shape {data['S_data'].shape}")
        
        if data['S_data'].shape[0] != len(data['q']):
            # Try to fix shape mismatch
            if data['S_data'].shape[1] == len(data['q']):
                data['S_data'] = data['S_data'].T
                data['S_errors'] = data['S_errors'].T
            else:
                raise ValueError(f"Data shape mismatch: S_data.shape={data['S_data'].shape}, q length={len(data['q'])}")
        
        return data

def create_sample_data():
    """Create sample data for testing"""
    q = np.linspace(0.1, 3.0, 50)
    omega = np.linspace(-5, 5, 200)
    
    # Create synthetic S(Q,ω) data with dispersion relation
    S_data = np.zeros((len(q), len(omega)))
    for i, q_val in enumerate(q):
        # Dispersion relation: ω = c * q
        omega0 = 2.0 * q_val  # Linear dispersion
        gamma = 0.5 + 0.1 * q_val  # Width increases with Q
        
        # Create Lorentzian peaks
        lorentzian = 5.0 * (gamma/2)**2 / ((omega - omega0)**2 + (gamma/2)**2)
        lorentzian += 0.5 * (0.6/2)**2 / ((omega + omega0)**2 + (0.6/2)**2)  # Anti-Stokes
        
        # Add background and noise
        background = 0.1 * np.exp(-omega**2/10)
        S_data[i, :] = lorentzian + background + 0.05 * np.random.randn(len(omega))
    
    # Ensure no negative values
    S_data = np.abs(S_data)
    S_errors = 0.1 * np.sqrt(S_data + 0.01)  # Simulated errors
    
    return {
        'q': q,
        'omega': omega,
        'S_data': S_data,
        'S_errors': S_errors,
        'filename': 'Sample Data'
    }