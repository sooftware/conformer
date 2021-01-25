<p  align="center"><img src="https://user-images.githubusercontent.com/42150335/105607164-aa878e00-5de0-11eb-8474-a12dd6ac919b.png" height=100>
  
<p  align="center">PyTorch implementation of Conformer: Convolution-augmented Transformer for Speech Recognition.

***

<p  align="center"> 
     <a href="https://github.com/sooftware/jasper/blob/main/LICENSE">
          <img src="http://img.shields.io/badge/license-Apache--2.0-informational"> 
     </a>
     <a href="https://github.com/pytorch/pytorch">
          <img src="http://img.shields.io/badge/framework-PyTorch-informational"> 
     </a>
     <a href="https://www.python.org/dev/peps/pep-0008/">
          <img src="http://img.shields.io/badge/codestyle-PEP--8-informational"> 
     </a>
     <a href="https://github.com/sooftware/conformer">
          <img src="http://img.shields.io/badge/build-passing-success"> 
     </a>
     <a href="https://sooftware.github.io/conformer">
          <img src="http://img.shields.io/badge/docs-passing-success"> 
     </a>

  
Transformer models are good at capturing content-based global interactions, while CNNs exploit local features effectively. Conformer combine convolution neural networks and transformers to model both local and global dependencies of an audio sequence in a parameter-efficient way. Conformer significantly outperforms the previous Transformer and CNN based models achieving state-of-the-art accuracies. In the paper, conformer used a one-lstm Transducer decoder, currently still only implemented the conformer encoder shown in the paper.

<img src="https://user-images.githubusercontent.com/42150335/105602364-aeafad80-5dd8-11eb-8886-b75e2d9d31f4.png" height=600>
  
## Installation
This project recommends Python 3.7 or higher.
We recommend creating a new virtual environment for this project (using virtual env or conda).
  
### Prerequisites
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.  
  
### Install from source
Currently we only support installation from source code using setuptools. Checkout the source code and run the
following commands:  
  
```
pip install -e .
```

## Usage

```python
import torch
from conformer import Conformer

batch_size, sequence_length, dimension = 3, 12345, 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(batch_size, sequence_length, dimension).to(device)  
input_lengths = torch.LongTensor([12345, 12300, 12000]).to(device)

model = Conformer(num_classes=10, input_dim=dimension, encoder_dim=512, num_layers=17)
outputs = model(inputs, input_lengths)
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/conformer/issues) on github or   
contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Reference
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/pdf/2005.08100.pdf)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl)
- [espnet/espnet](https://github.com/espnet/espnet)
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
