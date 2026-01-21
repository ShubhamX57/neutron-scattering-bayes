#!/usr/bin/env python3
"""
Main entry point for PyNeutron application
"""

import sys
import traceback

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['numpy', 'h5py', 'matplotlib', 'scipy', 'pandas']
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def main():
    """Main function to run the application"""
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print("Missing packages:", missing_packages)
        print("Please install them using:")
        print("pip install", " ".join(missing_packages))
        
        # Try to install automatically
        response = input("Attempt to install missing packages? (y/n): ")
        if response.lower() == 'y':
            import subprocess
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print("Installation successful! Restarting application...")
                # Restart the application
                subprocess.Popen([sys.executable, __file__])
                sys.exit(0)
            except:
                print("Failed to install packages. Please install manually.")
                input("Press Enter to exit...")
                sys.exit(1)
        else:
            input("Press Enter to exit...")
            sys.exit(1)
    
    # Import and run the application
    try:
        import tkinter as tk
        from .app import NeutronAnalysisApp
        
        root = tk.Tk()
        app = NeutronAnalysisApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()