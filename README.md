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
source activate cse842
```

Step 3: Install PyTorch
```shell
# webpage: https://pytorch.org/
### I prefer to use pip command to install python packages.
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchtext
pip install pytorch-crf
pip install matplotlib
```

step 4: double check if successfully installed the pytorch
```
type in python in the command line
import torch
exit()
```

## How to install Huggingface Transformer library:

```
pip install transformers
### Successfully installed huggingface-hub-0.10.0 pyyaml-6.0 regex-2022.9.13 tokenizers-0.12.1 transformers-4.22.2
```

## How to use AllenNLP?
### AllenNLP Introduction
> A natural language processing platform for building state-of-the-art models. A complete platform for solving natural language processing tasks in PyTorch.

### Install AllenNLP
```
pip install allennlp==2.1.0 allennlp-models==2.1.0
```

### AllenNLP examples

```
cd allennlp

allennlp dir:
    - named_entity_recognition
    - sementic_role_labeling
    - reading_comprehension
```








