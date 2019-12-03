import datetime
import os
import numpy as np
import gym

from environments.gym import OpenAIGym
from modules.network import FeatureDetector
from modules.run import Run
from modules.policy import Policy
from modules.normalize import Normalizer
from parameters import Hp



def evaluate(n_steps):
    hp = Hp()
    np.random.seed(hp.seed)
    env = OpenAIGym(hp.env_name)
    #env = wrappers.Monitor(env, monitor_dir, force=True)
    nb_inputs = env.observation_space.shape[0]
    if hp.conv_input:
        # create dummy image
        test_img = np.ones(
            [env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2]]).astype(
            np.uint8)
        # create dummy Normalizer obj
        test_n = Normalizer(0, hp)
        # pass through cnn
        test_output = test_n.image_cnn(test_img)
        # get output length
        nb_inputs = len(test_output)
    nb_outputs = env.action_space.shape[0]
    policy = Policy(nb_inputs, nb_outputs, hp)
    normalizer = Normalizer(nb_inputs, hp)

    normalizer.load() # load normalizer weights
    policy.load()  # load policy weights

    instance = Run(env, policy, normalizer, hp)
    instance.evaluate(n_steps)