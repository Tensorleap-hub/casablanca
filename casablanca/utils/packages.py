import subprocess
import sys

from casablanca.config import CONFIG


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_all_packages():
    for package_name in CONFIG['packages']:
        install(package_name)
