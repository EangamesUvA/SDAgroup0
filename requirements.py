# Installing required pip packages
import os
os.system("pip install pipreqs")
os.system("pipreqs --force")
os.system("pip install -r requirements.txt")
del os

# Packages needed by all files
import numpy as np

# Setup for all files (for example the dataset)
