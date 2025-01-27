# FSTT dynamics

[![version](https://img.shields.io/badge/python(64bit)_version-3.7.0-green.svg)](https://semver.org)

Would be nice if python is 64bit bebuz tensorflow will not try to kill you with errors then.
But if you like a challenge try 32bit.

This is the dynamics module code for the Formula Student project. 

## Setup
### Venv
You can setup and activate a virtual environment if wished.
##### Windows
* `>python -m venv --system-site-packages .\venv`
* `.\venv\Scripts\activate`
##### Ubuntu 
* `python3 -m venv --system-site-packages ./venv`
* `source ./venv/bin/activate`  # sh, bash, or zsh
* `../venv/bin/activate.fish`  # fish
* `source ./venv/bin/activate.csh`  # csh or tcsh

After activating the environment its a good idea to check out whats up with pip
* `pip install --upgrade pip`
* `pip list`  # show packages installed within the virtual environment

For exiting venv
* `deactivate`  # don't exit until you're done using TensorFlow
pip list  # show packages installed within the virtual environment`

### Installations
Install required python modules. If your system has an nvida graphics card - cuda is required `~ ver 10.0`
* `pip install -r requirements.txt`

If any problems
* `pip install -U pip`
* `python -m pip install --upgrade pip --user`
        
Now run the code in the specified files.

### Training in simulator (slower)
`fs_network.py`

### Training on saved data (faster)
`fs_network_train.py`

### Testing
`fs_network_test.py`