# Installing required pip packages
import os
os.system("python3 -m pip install --upgrade pip")
os.system("pip install pipreqs")
os.system("pipreqs --force")
os.system("pip install -r requirements.txt")
del os
