import os
import subprocess
import nltk
import sys

def install_requirements():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

def download_nltk():
    env_base_path = sys.prefix
    nltk_path = os.path.join(env_base_path, 'nltk_data')
    nltk.download('punkt', nltk_path)
    nltk.download('punkt_tab', nltk_path)

def setup():
    install_requirements()
    download_nltk()

if __name__ == '__main__':
    setup()