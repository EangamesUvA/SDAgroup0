# Installing required pip packages
import os
os.system("python3 -m pip install -q --upgrade pip")
os.system("pip install -q pipreqs")
os.system("pipreqs --force")
os.system("pip install -q -r requirements.txt")
del os
