import gym
from modules.environment import Environment


class OpenAIGym(Environment):

    def __init__(self, env_name):
        super(OpenAIGym, self).__init__()
        self.env_name = env_name
        self.env = self.make()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space


    def __repr__(self):
        return f"<OpenAIGym: {self.env_name}>"

    def make(self):
        env = gym.make(self.env_name)
        return env

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return state, reward, done, _
