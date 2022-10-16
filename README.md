# CSE_842_PyTorch

## Google colab links:
>- Warm UP: [link](https://colab.research.google.com/drive/1ISjZJkr-2IDzS9aPQ9RsQ0o_1kqzEFjJ?authuser=1)
>- Sentiment Analysis MLP: [link](https://colab.research.google.com/drive/1PmcbtS22Xt-PDJdc2uDgJrz7Wn6es3NL?authuser=0#scrollTo=PLn7EqNZom_Y)
>- Sentiment Analysis CNN: [link](https://colab.research.google.com/drive/1C3nFBYEOTwNbatBJj0rDUhFTMm4hOPZX?authuser=0#scrollTo=B1cQ-LbNokcN)
>- Reading Comprehension: [link](https://colab.research.google.com/drive/1bNUZdX91R0l2IpOgOX3VMCM8zy18GRl9?authuser=0#scrollTo=W8hL4ruV1o6f)
>- Bag-of-word experiment: [link](https://colab.research.google.com/drive/1imq405MXER61SnOWhL3-c_l1LfctNubL?authuser=0#scrollTo=KdAzwy8v62SP)
>- Word2Vec DIY: [link](https://colab.research.google.com/drive/1U83W7I99IokPORCcqeEAoJVHglvPC9RE?authuser=0#scrollTo=zEkrBmON7a83)

## Two Assignments (DIY your model):
>- Sentiment Analysis: [link](https://colab.research.google.com/drive/128CuZ2PPLeqkoXJyhlGeWHaVXZtQJVaT?authuser=0#scrollTo=PLn7EqNZom_Y)
>- Reading Comprehension: [link](https://colab.research.google.com/drive/1MNbOHeOroVFjoYGy_KVmM6yx_Tzq2o45?authuser=0#scrollTo=fzYf0fxzlk7A)

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








