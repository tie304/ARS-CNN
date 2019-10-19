import os
import numpy as np
import time


class Run:

    def __init__(self, env, policy, normalizer, hp):
        self.env = env
        self.normalizer = normalizer
        self.policy = policy
        self.hp = hp
        self.evaluations = []
        self.total_steps = 0

        # Exploring the policy on one specific direction and over one episode
    def explore(self, env, normalizer, policy, direction=None, delta=None, eval=False):
        state = env.reset()
        done = False
        num_plays = 0.
        sum_rewards = 0
        while not done and num_plays < self.hp.episode_length:
            if self.hp.conv_input:
                state = normalizer.image_cnn(state)
            normalizer.observe(state)
            state = normalizer.normalize(state)
            action = policy.evaluate(state, delta, direction)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
            self.total_steps +=1
            if eval and self.hp.render_eval:
                self.env.render()
        if eval:
            return sum_rewards, num_plays
        return sum_rewards

    # Training the AI
    def train(self):
        for episode in range(self.hp.nb_episodes):

            # Initializing the perturbations deltas and the positive/negative rewards
            deltas = self.policy.sample_deltas()
            positive_rewards = [0] * self.hp.nb_directions
            negative_rewards = [0] * self.hp.nb_directions

            # Getting the positive rewards in the positive directions
            for k in range(self.hp.nb_directions):
                positive_rewards[k] = self.explore(self.env, self.normalizer, self.policy, direction="positive", delta=deltas[k])

            # Getting the negative rewards in the negative/opposite directions
            for k in range(self.hp.nb_directions):
                negative_rewards[k] = self.explore(self.env, self.normalizer, self.policy, direction="negative", delta=deltas[k])

            # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
            all_rewards = np.array(positive_rewards + negative_rewards)
            sigma_r = all_rewards.std()

            # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in
                      enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.hp.nb_best_directions]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Updating our policy
            self.policy.update(rollouts, sigma_r)

            # Printing the final reward of the policy after the update
            avg_reward_evaluation = self.explore(self.env, self.normalizer, self.policy, eval=True)
            self.evaluations.append(
                (avg_reward_evaluation, episode, self.total_steps, time.time())
            )

            if episode % self.hp.save_freq == 0:
                self.policy.save()
                self._save_results_data()


            print('Episode:', episode, 'Reward:', avg_reward_evaluation)

    def evaluate(self, n_steps):
        self.policy.load() # load policy weights
        for step in range(n_steps):
            reward_evaluation = self.explore(self.env, self.normalizer, self.policy, eval=True)
            print('Step:', step, 'Reward:', reward_evaluation)

    def _save_results_data(self):
        np.save(self.hp.results_save_dir, self.evaluations)
