# CSE_842_PyTorch

## How to setup a standard Pytorch Environment?

Step 1: Install Anaconda:

```shell
# webpage: https://www.anaconda.com/
### create folder
mkdir anaconda
cd anaconda

# take linux as example:
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

sh Anaconda3-2022.05-Linux-x86_64.sh
```

Step 2: Create a conda environment
```shell
conda create -n cse842 python=3.9
conda activate cse842
```

Step 3: Install PyTorch
```shell
# webpage: https://pytorch.org/
### I prefer to use pip command to install python packages.
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

step 4: double check if successfully installed the pytorch
```
type in python in the command line
import torch
exit()
```








