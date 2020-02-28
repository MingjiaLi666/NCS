from src.policy import Policy

import numpy as np
import pickle
import gym
import os


class Visualizer(object):
    def __init__(self, game, network, train_directory):
        self.game = game
        env_name = '%sNoFrameskip-v4' % game
        env = gym.make(env_name)
        # env = gym.wrappers.Monitor(env, '/tmp/temp_%s' % game, mode='evaluation', force=True)

        vb_file = os.path.join(train_directory, "vb.npy")
        vb = np.load(vb_file)
        parameters_file = 'parameters_81'

        self.policy = Policy(env, network, "relu")

        parameters_path = os.path.join(train_directory, parameters_file)
        print('Using parameters file %s \n' % parameters_path)

        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)['parameters']

        self.policy.set_parameters(parameters)
        self.policy.set_vb(vb)

    def play_game(self):
        rews = [0]*100
        for i in range(100):
            rew,step = self.policy.rollout()
            rews[i] = rew
        print(np.mean(rews))
        print(np.max(rews))
        print(rews)




if __name__ == '__main__':
    vis = Visualizer('Venture', 'Nature', train_directory='./logs_mpi/Venture/Baseline/Nature/8/50/0.010000/1.000000/1.000000/NCSVenture2')
    vis.play_game()
