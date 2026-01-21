"""
PyNeutron - Neutron Data Analysis Tool
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .main import main
from .app import NeutronAnalysisApp
from .data_loader import load_nexus_file
from .fitting import lorentzian, double_lorentzian, gaussian

__all__ = [
    "main",
    "NeutronAnalysisApp",
    "load_nexus_file",
    "lorentzian",
    "double_lorentzian",
    "gaussian",
]