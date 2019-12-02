
# Importing the libraries
import datetime
import os
import numpy as np


from modules.network import FeatureDetector
from environments.gym import OpenAIGym
from modules.run import Run
from modules.policy import Policy
from modules.normalize import Normalizer
from parameters import Hp


def train():
    hp = Hp()
    np.random.seed(hp.seed)
    env = OpenAIGym(hp.env_name)
    nb_inputs = env.observation_space.shape[0]
    if hp.conv_input:
        nb_inputs = 2304 # inputs after convolution TODO build function that finds output for cnn
    nb_outputs = env.action_space.shape[0]
    policy = Policy(nb_inputs, nb_outputs, hp)
    normalizer = Normalizer(nb_inputs, hp)

    if hp.train_from_previous_weights:
        policy.load()
        normalizer.load()

    instance = Run(env, policy, normalizer, hp)
    instance.train()