import numpy as np
from PIL import Image


from modules.network import FeatureDetector
import sys
# Normalizing the states


class Normalizer:

    def __init__(self, nb_inputs):
        self.cnn = FeatureDetector()
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

    def image_cnn(self, obs):
        new_obs = Image.fromarray(obs)
        new_obs = new_obs.convert('L')
        new_obs = np.array(new_obs)
        #print(new_obs.shape)
        new_obs = new_obs.reshape(1, 96, 96)
        new_obs = self.cnn(new_obs)
        #print(new_obs.shape)
        new_obs = new_obs.detach().numpy()

        return new_obs[0]


