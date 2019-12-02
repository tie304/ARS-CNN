import datetime
import os
import numpy as np
import gym


from modules.network import FeatureDetector
from modules.run import Run
from modules.policy import Policy
from modules.normalize import Normalizer
from parameters import Hp



def evaluate(n_steps):
    hp = Hp()
    np.random.seed(hp.seed)
    env = gym.make(hp.env_name)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    nb_inputs = env.observation_space.shape[0]
    if hp.conv_input:
        nb_inputs = 2304  # inputs after convolution TODO build function that finds output for cnn
    nb_outputs = env.action_space.shape[0]
    policy = Policy(nb_inputs, nb_outputs, hp)
    normalizer = Normalizer(nb_inputs, hp)

    normalizer.load() # load normalizer weights
    policy.load()  # load policy weights

    instance = Run(env, policy, normalizer, hp)
    instance.evaluate(n_steps)