from os import name
import numpy as np
import neurogym as ngym
import matplotlib.pyplot as plt
from neurogym import spaces
import main

from neurogym.envs.perceptualdecisionmaking import (
    PerceptualDecisionMaking,
    PerceptualDecisionMakingDelayResponse,
)
from neurogym.envs.gonogo import GoNogo
from neurogym.envs.reachingdelayresponse import ReachingDelayResponse

if __name__ == "__main__":
    env = main.AntiPerceptualDecisionMakingDelayResponseT(
        config={"dt": 20, "mode": "test", "rng": np.random.RandomState(0)}
    )
    fig = ngym.utils.plot_env(env, num_trials=4)
    plt.show()
