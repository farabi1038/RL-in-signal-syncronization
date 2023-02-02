import numpy as np

class state_v1:
    '''
    Signal 7216 lanearea detectors queue sensor
    Each direction is summed up to one value, hence size is 4
    '''
    def __init__(self):
        self.size = 4
        self.temp = []
        self.lane_dic = {"NS":0, "WE": 1, "SN": 2, "EW": 3}
        self.edge_reduced = False
        self.edge_full = False
        self.edge_full_side = False
        self.induction = False
        self.lanearea = True
        self.multi = False

    def __len__(self):
        return self.size

    def set(self, value):
        return np.full(self.size, value).astype(np.float32)

    def reset(self):
        self.temp = []

    def update(self, loop, edge, lanearea, multi, env):
        self.local_temp = np.zeros(self.size)
        for item in lanearea:
            lvl_1 = item[0:4]
            lvl_2 = item[5:7]
            self.local_temp[self.lane_dic[lvl_2]] += list(lanearea[item].values())[0]

        self.temp.append(self.local_temp)

    def retrieve(self):
        return np.sum(np.array(self.temp), axis=0).astype(np.float32)
