# Ubuntu Setup

## 1.0 Tensorflow GPU
This section describes the setup of Tensorflow with GPU acceleration.

### 1.1 System Configuration
With a clean installation of Ubuntu 16.04.3, perform operations listed in this section.

First, OpenSSH is installed to remote into the system.
```
$ sudo apt-get install openssh-server -y
$ sudo service ssh restart
```
Next, system dependencies to support Python and Tensorflow are installed.
```
$ sudo apt-get install build-essential python-dev
$ sudo apt-get install gfortran
$ sudo apt-get install swig
$ sudo apt-get install libatlas-dev
$ sudo apt-get install liblapack-dev

$ sudo apt-get install libfreetype6 libfreetype6-dev
$ sudo apt-get install libxft-dev
$ sudo apt-get install graphviz libgraphviz-dev
$ sudo apt-get install pandoc
$ sudo apt-get install libxml2-dev libxslt-dev zlib1g-dev
```
We will use Screen, this allows processes to continue operating when a terminal window is closed- for example Jupyter will continue to run after closing an SSH session.
```
sudo apt-get install screen
```

### 1.2 Python Configuration
Configure Python 3 (3.5 already installed)
```
$ sudo apt install python-pip
$ sudo apt-get install python3-pip
$ sudo pip3 install virtualenv
$ sudo pip3 install virtualenvwrapper
```
Then we need to configure virtualenvwrapper
```
$ mkdir DataScience
$ export WORKON_HOME=~/DataScience
```
Now add the following line to ~/.bashrc. Note that the path is dependent on your installation path for virtualenvwrapper, you can find it by typing 'which virtualenvwrapper.sh'
```
. /usr/local/bin/virtualenvwrapper.sh
```
Create a virtual environment for the project
```
$ mkvirtualenv --python=$(which python3) tensorflow-gpu
```
Then install basic packages. We won't install Tensorflow yet, that will happen after GPU configuration.
```
pip3 install pandas numpy scipy matplotlib jupyter
```
### 1.3 Launch jupyter
Next, we want to test the python installation. We will launch jupyter from a project folder where we want to work. Note the --ip flag on jupyter notebook, this is necessarry to access Jupyter remotely on the network. When logging in to Jupyter from a browser, you will be asked for a token- this token will be displayed in the SSH terminal.

First, let's make a project directory.
```
mkdir ~/Projects
cd ~/Projects
mkdir tf_wordvec
cd tf_wordvec
```

Then start a new *screen* session  called "jupytersession" and start Jupyter. Screen will make sure jupyter continues to run if we stop the SSH session.
```
screen -S jupytersession
jupyter notebook --ip='*'
```
To detach a screen session, use Ctrl-A then Ctrl-D. You can then close the SSH window and jupyter will keep running. To re-enter the session later, type 'screen -r jupytersession'.

### 1.3 GPU Configuration
