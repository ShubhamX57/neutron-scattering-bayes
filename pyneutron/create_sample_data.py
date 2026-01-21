"""
Create sample HDF5 file for testing
"""

import h5py
import numpy as np

# Create sample data
q = np.linspace(0.1, 3.0, 50)
omega = np.linspace(-5, 5, 200)

# Create synthetic S(Q,ω) data
S_data = np.zeros((len(q), len(omega)))
for i, q_val in enumerate(q):
    omega0 = 2.0 * q_val
    gamma = 0.5 + 0.1 * q_val
    
    lorentzian = 5.0 * (gamma/2)**2 / ((omega - omega0)**2 + (gamma/2)**2)
    lorentzian += 0.5 * (0.6/2)**2 / ((omega + omega0)**2 + (0.6/2)**2)
    
    background = 0.1 * np.exp(-omega**2/10)
    S_data[i, :] = lorentzian + background + 0.05 * np.random.randn(len(omega))

S_data = np.abs(S_data)
S_errors = 0.1 * np.sqrt(S_data + 0.01)

# Save to HDF5 file
with h5py.File('sample_data.h5', 'w') as f:
    f.create_dataset('q', data=q)
    f.create_dataset('omega', data=omega)
    f.create_dataset('S_data', data=S_data)
    f.create_dataset('S_errors', data=S_errors)
    
print("Sample data saved to sample_data.h5")