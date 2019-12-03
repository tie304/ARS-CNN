import numpy as np
from PIL import Image


from modules.network import FeatureDetector
# Normalizing and pre-processing the states


class Normalizer:

    def __init__(self, nb_inputs, hp):
        if hp.conv_input:
            self.cnn = FeatureDetector()
        self.hp = hp
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
        new_obs = new_obs.reshape(1, obs.shape[0], obs.shape[1])
        new_obs = self.cnn(new_obs).to(self.hp.device)
        new_obs = new_obs.detach().numpy()
        return new_obs[0]

    def save(self):
        np.save(self.hp.normalizer_weights_save, [self.n, self.mean, self.mean_diff, self.var])

    def load(self):
        self.n, self.mean, self.mean_diff, self.var = np.load(self.hp.normalizer_weights_save)




