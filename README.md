# Alignment measurement of RNN models trained on multi neurotasks

## How to install

- install poetry : curl -sSL https://install.python-poetry.org | python3 -
- run the following command to install necessary dependancies : poetry install 

## How to run

- select kernel created with poetry
- change parameters in the `config.yaml` file
- run notebook of your choice

## How to use

The code is mainly divided into 2 parts:

### The Computational Neuroscience part

Modelisation, training and evaluation of the RNN models on the neurotasks in the paper "Task representations in neural networks trained to perform many cognitive tasks" by Yang et al.
- multitask : Folder containing the code to train the models on the neurotasks
- Activity_and_Representation.ipynb : Analysis of the activity and representation of the models

### The dynamical similarity analysis part

Computation of the alignment between the models using the dynamical similarity analysis method proposed by Ostrow et al.
To show that this metric is relevant, we performe simulations of dynamical motifs:
- dsa_analysis : Folder containing the code to compute the dynamical similarity between the models
- lorenz_sanity_check.ipynb : Creation of different Lorenz Attractors and similarity analysis between them
- compositional.ipynb : Creation of compositional motifs and similarity analysis between them
- dsa_epochs.ipynb : Analysis of the evolution of the similarity between the models during training, simulated by adding noise of different strengths to the motifs


## References

- Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ila Fiete, "Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis", 	arXiv:2306.10168
- Yang, G.R., Joglekar, M.R., Song, H.F. et al. Task representations in neural networks trained to perform many cognitive tasks. Nat Neurosci 22, 297–306 (2019). https://doi.org/10.1038/s41593-018-0310-2
- Sugandha Sharma, Fernanda De La Torre, "MIT/Harvard Computational & Theoretical Neuroscience Journal Club", https://compneurojc.github.io/index.html 