import subprocess

def install_requirements():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

def setup():
    install_requirements()

if __name__ == '__main__':
    setup()