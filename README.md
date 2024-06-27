# Alignment measurement of RNN models trained on multi neurotasks

## How to install

- install poetry : curl -sSL https://install.python-poetry.org | python3 -
- run the following command to install necessary dependancies : poetry install 
- create folders for models and data: mkdir models data

## How to use

The code is mainly divided into 2 parts:

### 1 : Simulations

Use 3D attractor dynamics to show if RepAL metric is relevant to capture compositional problem solving and compositional learning. 

####  Code Structure
- dsa_analysis : Folder containing the code to compute the dynamical similarity between the models
- lorenz_sanity_check.ipynb : Creation of different Lorenz Attractors and similarity analysis between them
- compositional.ipynb : Creation of compositional motifs and similarity analysis between them
- dsa_epochs.ipynb : Analysis of the evolution of the similarity between the models during training, simulated by adding noise of different strengths to the motifs

#### How to use
- select kernel created with poetry
- change parameters in the config.yaml file
- run notebook of your choice

### 2 : Application to RNN models trained on neurotasks

Use the validated metrics to apply them to a known case of compositional learning and problem solving 

#### Code Structure
- main: Folder containing the code to train the models on the neurotasks
- notebooks/analysis_rnn.ipynb : Analysis of the activity and representation of the models
- sbatch: Folder containing the scripts to run the training on the cluster
- config.yaml : Configuration file for the training
- src: Folder containing the code to train the models and analyze the results

#### How to use (in order of execution)
- generate the data: sbatch sbatch/generate_data.sh
- train the models: sbatch sbatch/train.sh
- compute dissimilarity of computational dynamics of models: sbatch sbatch/dissimilarity.sh
- compute dissimilarities of learning dynamics of models: sbatch sbatch/dissimilarity_learning.sh
- analysis of the results: notebooks/analysis_rnn.ipynb

## References

- Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ila Fiete, "Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis", 	https://arxiv.org/abs/2306.10168
- Yang, G.R., Joglekar, M.R., Song, H.F. et al. Task representations in neural networks trained to perform many cognitive tasks. Nat Neurosci 22, 297–306 (2019). https://doi.org/10.1038/s41593-018-0310-2
- Sugandha Sharma, Fernanda De La Torre, "MIT/Harvard Computational & Theoretical Neuroscience Journal Club", https://compneurojc.github.io/index.html 
- Laura Driscoll, Krischna Shenoy, David Sussillo, "Flexible multitask computation in recurrent networks utilizes shared dynamical motifs", https://www.biorxiv.org/content/10.1101/2022.08.15.503870v1
- lia Sucholutsky, Lukas Muttenthaler, Adrian Weller, Andi Peng, Andreea Bobu, Been Kim, Bradley C. Love, Erin Grant, Iris Groen, Jascha Achterberg, Joshua B. Tenenbaum, Katherine M. Collins, Katherine L. Hermann, Kerem Oktar, Klaus Greff, Martin N. Hebart, Nori Jacoby, Qiuyi Zhang, Raja Marjieh, Robert Geirhos, Sherol Chen, Simon Kornblith, Sunayana Rane, Talia Konkle, Thomas P. O'Connell, Thomas Unterthiner, Andrew K. Lampinen, Klaus-Robert Müller, Mariya Toneva, Thomas L. Griffiths, "Getting aligned on representational alignment", https://arxiv.org/abs/2310.13018
- Nathan Cloos, Markus Siegel, Scott L. Brincat, Earl K. Miller, Christopher J Cueva, "Differentiable Optimization of Similarity Scores Between Models and Brains", https://openreview.net/forum?id=C0G0mQp92K
- Stefano Sarao Mannelli, Yaraslau Ivashinka, Andrew Saxe, Luca Saglietti, "Tilting the Odds at the Lottery:
the Interplay of Overparameterisation and Curricula in Neural Networks", https://arxiv.org/abs/2406.01589
- Manuel Molano-Mazón, Joao Barbosa, Jordi Pastor-Ciurana, Marta Fradera, RU-YUAN ZHANG, Jeremy Forest, Jorge del Pozo Lerida, Li Ji-An, Christopher J Cueva, Jaime de la Rocha, Devika Narain, and Guangyu Robert Yang, "NeuroGym: An open resource for developing and sharing neuroscience tasks", https://osf.io/preprints/psyarxiv/aqc9n