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
(tensorflow-gpu) $ pip3 install pandas numpy scipy matplotlib jupyter
```
#### 1.2.1 Launch jupyter
Next, we want to test the python installation. We will launch jupyter from a project folder where we want to work. Note the --ip flag on jupyter notebook, this is necessarry to access Jupyter remotely on the network. When logging in to Jupyter from a browser, you will be asked for a token- this token will be displayed in the SSH terminal.

First, let's make a project directory.
```
(tensorflow-gpu) $ mkdir ~/Projects
(tensorflow-gpu) $ cd ~/Projects
(tensorflow-gpu) $ mkdir tf_wordvec
(tensorflow-gpu) $ cd tf_wordvec
```

Then start a new *screen* session  called "jupytersession" and start Jupyter. Screen will make sure jupyter continues to run if we stop the SSH session.
```
$ screen -S jupytersession (this will start a new session, you need to reactivate the virutal environment)
$ workon tensorflow-gpu
(tensorflow-gpu) $ jupyter notebook --ip='*'
```
To detach a screen session, use Ctrl-A then Ctrl-D. You can then close the SSH window and jupyter will keep running. To re-enter the session later, type 'screen -r jupytersession'.

### 1.3 GPU Configuration
For Tensorflow r1.4, we need to install the following NVIDIA frameworks:
* CUDA 8.0
* cuDNN v6.0

#### 1.3.1 Drivers
First, install drivers for the GPU.
```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update (re-run if any warning/error messages)
$ sudo apt-get install nvidia-384 (press tab after - to see all versions)
```
When driver installation is complete, reboot the system:
```
$ sudo reboot
```
To verify proper installation, run the *nvidia-smi* command to inspect hardware configuration.

#### 1.3.2 CUDA
To begin, make a folder for nvidia assets and download the CUDA toolkit deb file into it:
```
$ cd /usr/local
$ mkdir nvidia
$ cd nvidia
$ sudo wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
```
Incorporate the deb file into your repos and install CUDA
```
$ sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
$ sudo apt-get update
$ sudo apt-get install cuda-8-0
```

#### 1.3.3 cuDNN
Next, cuDNN V6.0 for CUDA 8.0 is installed. Due to login requirements with the [cuDNN download](https://developer.nvidia.com/rdp/cudnn-download) page, download the file locally and scp it to the Ubuntu instance:
```
scp cudnn-8.0-linux-x64-v6.0.tgz user@192.168.1.100:/home/user/Downloads
```
Back on the Ubuntu machine, install cuDNN:
```
$ cd Downloads
$ tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
$ sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*
```
Add the following lines to ~/.bashrc to complete GPU configuration:
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-8.0
```

### 1.4 Tensorflow Installation
Finally, install tensorflow with gpu acceleration:
```
$ workon tensorflow-gpu
(tensorflow-gpu) $ pip3 install --upgrade tensorflow-gpu
```
