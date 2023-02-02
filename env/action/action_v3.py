import numpy as np
from gym import spaces

class action_v3:

    def __init__(self, cfg):
        self.dummy = 0

    def space(self):
        return spaces.Box(np.full(36,0).astype(np.float32), np.full(36,1).astype(np.float32))

    def process(self, split, temp, action):
        action = np.clip(action, 0, 1)

        for idx in split.loc[split['signalid'] == 7216].index:
            split.loc[idx, 'main_ratio'] = action[0]
            split.loc[idx, 'r1g1'] = action[1]
            split.loc[idx, 'r1g2'] = action[2]
            split.loc[idx, 'r2g1'] = action[3]
            split.loc[idx, 'r2g2'] = action[4]

        for idx in split.loc[split['signalid'] == 7217].index:
            split.loc[idx, 'main_ratio'] = action[5]
            split.loc[idx, 'r1g1'] = action[6]

        for idx in split.loc[split['signalid'] == 7218].index:
            split.loc[idx, 'main_ratio'] = action[7]
            split.loc[idx, 'r1g1'] = action[8]
            split.loc[idx, 'r2g1'] = action[9]

        for idx in split.loc[split['signalid'] == 7219].index:
            split.loc[idx, 'main_ratio'] = action[10]
            split.loc[idx, 'r1g1'] = action[11]
            split.loc[idx, 'r1g2'] = action[12]
            split.loc[idx, 'r2g1'] = action[13]
            split.loc[idx, 'r2g2'] = action[14]

        for idx in split.loc[split['signalid'] == 7503].index:
            split.loc[idx, 'main_ratio'] = action[15]

        for idx in split.loc[split['signalid'] == 7220].index:
            split.loc[idx, 'main_ratio'] = action[16]
            split.loc[idx, 'r1g1'] = action[17]
            split.loc[idx, 'r2g1'] = action[18]

        for idx in split.loc[split['signalid'] == 7221].index:
            split.loc[idx, 'main_ratio'] = action[19]
            split.loc[idx, 'r1g2'] = action[20]
            split.loc[idx, 'r2g2'] = action[20]

        for idx in split.loc[split['signalid'] == 7222].index:
            split.loc[idx, 'main_ratio'] = action[21]

        for idx in split.loc[split['signalid'] == 7223].index:
            split.loc[idx, 'main_ratio'] = action[22]

        for idx in split.loc[split['signalid'] == 7371].index:
            split.loc[idx, 'main_ratio'] = action[23]

        minc = int(temp['minC'][0])
        maxc = int(temp['maxC'][0])

        temp['cycle_length'] = np.ceil(action[24]*(maxc-minc)/5)*5+minc

        temp['offset'][0] = int(round(action[25]*temp['cycle_length'][0]))
        temp['offset'][1] = int(round(action[26]*temp['cycle_length'][0]))
        temp['offset'][2] = int(round(action[27]*temp['cycle_length'][0]))
        temp['offset'][3] = int(round(action[28]*temp['cycle_length'][0]))
        temp['offset'][4] = int(round(action[29]*temp['cycle_length'][0]))
        temp['offset'][5] = int(round(action[30]*temp['cycle_length'][0]))
        temp['offset'][6] = int(round(action[31]*temp['cycle_length'][0]))
        temp['offset'][7] = int(round(action[32]*temp['cycle_length'][0]))
        temp['offset'][8] = int(round(action[33]*temp['cycle_length'][0]))
        temp['offset'][9] = int(round(action[34]*temp['cycle_length'][0]))
        temp['offset'][10] = int(round(action[35]*temp['cycle_length'][0]))

        return split, temp

    def save_csv(self, path, fname, split, temp):
        split.to_csv(f"{path}/{fname}_split.csv")
        temp.to_csv(f"{path}/{fname}_temp.csv")