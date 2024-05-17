import DSA
from dsa_analysis.tools import load_config, flatten_x
from RNN_rate_dynamics import RNNLayer
import seaborn as sns
import ipdb
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np

# load config
config = load_config('config.yaml')

simulations = np.load("data/lorenz_simulation_1000.npy")

model = [simulations[i,j,0,:,:] for i in range(3) for j in range(3)]
model_names = ['one_attractor_1', 'one_attractor_2', 'one_attractor_3','two_stable_attractors_1', 'two_stable_attractors_2', 'two_stable_attractors_3', 'two_unstable_attractors_1', 'two_unstable_attractors_2', 'two_unstable_attractors_3']

# # dsa initialization
# dsa = DSA.DSA(model,n_delays=config['dsa']['n_delays'],rank=config['dsa']['rank'],delay_interval=config['dsa']['delay_interval'],verbose=True,iters=1000,lr=1e-2)
# similarities = dsa.fit_score()
# np.save("data/similarities_5000_steps.npy",similarities)
ipdb.set_trace()


similarities = np.load("data/similarities_1000_steps.npy")
sns.heatmap(similarities)
plt.show()

df = pd.DataFrame()
df['Model Type'] = model_names
reduced = MDS(dissimilarity='precomputed').fit_transform(similarities)
df["0"] = reduced[:,0] 
df["1"] = reduced[:,1]

palette = 'plasma'
sns.scatterplot(data=df,x="0",y="1",hue="Model Type",palette=palette)
plt.xlabel(f"MDS 1")
plt.ylabel(f"MDS 2")
plt.tight_layout()
plt.show()

# # training
# hp, log, optimizer = multitask.set_hyperparameters(model_dir='debug', hp={'learning_rate': 0.001}, ruleset=config['training']['ruleset']) #, rich_output=True)
# run_model = multitask.Run_Model(hp, RNNLayer)

# # Load the state dictionary into the model
# run_model.load_state_dict(torch.load(config['analysis']['model']))

# h = multitask.all_traces(run_model, config['analysis']['rule'])
# ipdb.set_trace()
# model = [h[key] for key in h.keys() if key[1]=="stim1" ]
# model_names = [key[0] for key in h.keys() if key[1]=="stim1" ]
# ipdb.set_trace()