import numpy as np
from gym import spaces
import pandas as pd
import os

class action_v7:

    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dic = {0: 0.0625, 1: 0.125, 2: 0.1875, 3: 0.25, 4: 0.3125, 5: 0.37, 6: 0.4375, 7: 0.5, \
                            8: 0.5625, 9: 0.625, 10: 0.6875, 11: 0.75, 12: 0.8125, 13: 0.875, 14: 0.9375, 15: 1.0}

        self.spaces = [spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(8), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(4), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(4), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), \
                        spaces.Discrete(16), spaces.Discrete(16), \
                        spaces.Discrete(16)]
        [i.seed(cfg.action.seed + j) for (i,j) in zip(self.spaces,range(len(self.spaces)))]

    def space(self):
        return self.spaces

    def process(self, split, temp, action):

        temp['cycle_length'] = np.ceil(self.action_dic[action[0]]*(int(temp['maxC'][0])-int(temp['minC'][0]))/5)*5+int(temp['minC'][0])

        for idx in split.loc[split['signalid'] == 7216].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[1]]
            split.loc[idx, 'r1g1'] = self.action_dic[action[2]]
            split.loc[idx, 'r1g2'] = self.action_dic[action[3]]
            split.loc[idx, 'r2g1'] = self.action_dic[action[4]]
            split.loc[idx, 'r2g2'] = self.action_dic[action[5]]

        for idx in temp.loc[temp['signalid'] == 7216].index:
            temp.loc[idx, 'planid'] = action[6]
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[7]]*int(temp.loc[idx,'cycle_length'])))
        
        for idx in split.loc[split['signalid'] == 7217].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[8]]
            split.loc[idx, 'r1g1'] = self.action_dic[action[9]]

        for idx in temp.loc[temp['signalid'] == 7217].index:
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[10]]*int(temp.loc[idx,'cycle_length'])))
        
        for idx in split.loc[split['signalid'] == 7218].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[11]]
            split.loc[idx, 'r1g1'] = self.action_dic[action[12]]
            split.loc[idx, 'r2g1'] = self.action_dic[action[13]]

        for idx in temp.loc[temp['signalid'] == 7218].index:
            temp.loc[idx, 'planid'] = action[14]
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[15]]*int(temp.loc[idx,'cycle_length'])))
        
        for idx in split.loc[split['signalid'] == 7219].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[16]]
            split.loc[idx, 'r1g1'] = self.action_dic[action[17]]
            split.loc[idx, 'r1g2'] = self.action_dic[action[18]]
            split.loc[idx, 'r2g1'] = self.action_dic[action[19]]
            split.loc[idx, 'r2g2'] = self.action_dic[action[20]]

        for idx in temp.loc[temp['signalid'] == 7219].index:
            temp.loc[idx, 'planid'] = action[21]
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[22]]*int(temp.loc[idx,'cycle_length'])))
        
        for idx in split.loc[split['signalid'] == 7503].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[23]]

        for idx in temp.loc[temp['signalid'] == 7503].index:
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[24]]*int(temp.loc[idx,'cycle_length'])))
        
        for idx in split.loc[split['signalid'] == 7220].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[25]]
            split.loc[idx, 'r1g1'] = self.action_dic[action[26]]
            split.loc[idx, 'r2g1'] = self.action_dic[action[27]]

        for idx in temp.loc[temp['signalid'] == 7220].index:
            temp.loc[idx, 'planid'] = action[28]
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[29]]*int(temp.loc[idx,'cycle_length'])))

        for idx in split.loc[split['signalid'] == 7221].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[30]]
            split.loc[idx, 'r1g2'] = self.action_dic[action[31]]
            split.loc[idx, 'r2g2'] = self.action_dic[action[31]]

        for idx in temp.loc[temp['signalid'] == 7221].index:
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[32]]*int(temp.loc[idx,'cycle_length'])))
        
        for idx in split.loc[split['signalid'] == 7222].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[33]]

        for idx in temp.loc[temp['signalid'] == 7222].index:
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[34]]*int(temp.loc[idx,'cycle_length'])))
        
        for idx in split.loc[split['signalid'] == 7223].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[35]]

        for idx in temp.loc[temp['signalid'] == 7223].index:
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[36]]*int(temp.loc[idx,'cycle_length'])))
        
        for idx in split.loc[split['signalid'] == 7371].index:
            split.loc[idx, 'main_ratio'] = self.action_dic[action[37]]

        for idx in temp.loc[temp['signalid'] == 7371].index:
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[38]]*int(temp.loc[idx,'cycle_length'])))

        for idx in temp.loc[temp['signalid'] == 7274].index:
            temp.loc[idx, 'offset'] = int(round(self.action_dic[action[39]]*int(temp.loc[idx,'cycle_length'])))

        return split, temp

    def save_csv(self, path, fname, split, temp, index):
        split.to_csv(f"{path}/{fname}_split_{index}.csv")
        temp.to_csv(f"{path}/{fname}_temp_{index}.csv")

