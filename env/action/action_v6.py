import numpy as np
from gym import spaces
import pandas as pd
import os

class action_v6:

    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dic = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
        self.spaces = [spaces.Discrete(6), spaces.Discrete(6), spaces.Discrete(6), spaces.Discrete(6), spaces.Discrete(6), \
                        spaces.Discrete(8), \
                        spaces.Discrete(6), spaces.Discrete(6)]
        [i.seed(cfg.action.seed + j) for (i,j) in zip(self.spaces,range(len(self.spaces)))]

    def space(self):
        return self.spaces

    def process(self, split, temp, action):

        for idx in split.loc[split['signalid'] == 7216].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[0]]
            split.loc[idx, 'r1g1'] = self.action_dic[action[1]]
            split.loc[idx, 'r1g2'] = self.action_dic[action[2]]
            split.loc[idx, 'r2g1'] = self.action_dic[action[3]]
            split.loc[idx, 'r2g2'] = self.action_dic[action[4]]

        temp['cycle_length'] = np.ceil(self.action_dic[action[7]]*(int(temp['maxC'][0])-int(temp['minC'][0]))/5)*5+int(temp['minC'][0])
        for idx in temp.loc[temp['signalid'] == 7216].index:
            temp.loc[idx, 'planid'] = action[5]
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[6]]*int(temp.loc[idx,'cycle_length'])))


        return split, temp

    def save_csv(self, path, fname, split, temp):
        split.to_csv(f"{path}/{fname}_split.csv")
        temp.to_csv(f"{path}/{fname}_temp.csv")