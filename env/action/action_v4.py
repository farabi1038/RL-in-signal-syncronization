import numpy as np
from gym import spaces

class action_v4:

    def __init__(self, cfg):
        self.action_dic = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

    def space(self):
        return spaces.Discrete(6)

    def process(self, split, temp, action):
        # action = np.clip(action, 0, 1)
        print(action)

        for idx in split.loc[split['signalid'] == 7216].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[0]]
            split.loc[idx, 'r1g1'] = self.action_dic[action[1]]
            split.loc[idx, 'r1g2'] = self.action_dic[action[2]]
            split.loc[idx, 'r2g1'] = self.action_dic[action[3]]
            split.loc[idx, 'r2g2'] = self.action_dic[action[4]]


        return split, temp

    def save_csv(self, path, fname, split, temp):
        split.to_csv(f"{path}/{fname}_split.csv")
        temp.to_csv(f"{path}/{fname}_temp.csv")