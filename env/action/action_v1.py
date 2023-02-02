import numpy as np
from gym import spaces

class action_v1:

    def __init__(self, cfg):
        self.dummy = 0

    def space(self):
        return spaces.Box(np.full(5,0).astype(np.float32), np.full(5,1).astype(np.float32))

    def process(self, split, temp, action):
        action = np.clip(action, 0, 1)

        for idx in split.loc[split['signalid'] == 7216].index:
            split.loc[idx, 'main_ratio'] = action[0]
            split.loc[idx, 'r1g1'] = action[1]
            split.loc[idx, 'r1g2'] = action[2]
            split.loc[idx, 'r2g1'] = action[3]
            split.loc[idx, 'r2g2'] = action[4]

        # for idx in split.loc[split['signalid'] == 7217].index:
        #     split.loc[idx, 'main_ratio'] = action[5]
        #     split.loc[idx, 'r1g1'] = action[6]

        # for idx in split.loc[split['signalid'] == 7218].index:
        #     split.loc[idx, 'main_ratio'] = action[7]
        #     split.loc[idx, 'r1g1'] = action[8]
        #     split.loc[idx, 'r2g1'] = action[9]

        # for idx in split.loc[split['signalid'] == 7219].index:
        #     split.loc[idx, 'main_ratio'] = action[10]
        #     split.loc[idx, 'r1g1'] = action[11]
        #     split.loc[idx, 'r1g2'] = action[12]
        #     split.loc[idx, 'r2g1'] = action[13]
        #     split.loc[idx, 'r2g2'] = action[14]

        # for idx in split.loc[split['signalid'] == 7503].index:
        #     split.loc[idx, 'main_ratio'] = action[15]

        # for idx in split.loc[split['signalid'] == 7220].index:
        #     split.loc[idx, 'main_ratio'] = action[16]
        #     split.loc[idx, 'r1g1'] = action[17]
        #     split.loc[idx, 'r2g1'] = action[18]

        # for idx in split.loc[split['signalid'] == 7221].index:
        #     split.loc[idx, 'main_ratio'] = action[19]
        #     split.loc[idx, 'r1g2'] = action[20]
        #     split.loc[idx, 'r2g2'] = action[20]

        # for idx in split.loc[split['signalid'] == 7222].index:
        #     split.loc[idx, 'main_ratio'] = action[21]

        # for idx in split.loc[split['signalid'] == 7223].index:
        #     split.loc[idx, 'main_ratio'] = action[22]

        # for idx in split.loc[split['signalid'] == 7371].index:
        #     split.loc[idx, 'main_ratio'] = action[23]

        return split, temp

    def save_csv(self, path, fname, split, temp):
        split.to_csv(f"{path}/{fname}_split.csv")
        temp.to_csv(f"{path}/{fname}_temp.csv")