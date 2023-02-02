import numpy as np

class state_v2:
	'''
	Signal 7216 lanearea detectors queue sensor
	Each direction have 2 values, through and left turn, hence size is 8
	'''
	def __init__(self):
		self.size = 8
		self.temp = []
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
		self.local_temp[0] += list(lanearea["7216_WE_4"].values())[0]
		self.local_temp[0] += list(lanearea["7216_WE_3"].values())[0]
		self.local_temp[1] += list(lanearea["7216_WE_2"].values())[0]
		self.local_temp[1] += list(lanearea["7216_WE_1"].values())[0]
		self.local_temp[1] += list(lanearea["7216_WE_0"].values())[0]
		self.local_temp[2] += list(lanearea["7216_SN_2"].values())[0]
		self.local_temp[3] += list(lanearea["7216_SN_1"].values())[0]
		self.local_temp[3] += list(lanearea["7216_SN_0"].values())[0]
		self.local_temp[4] += list(lanearea["7216_EW_3"].values())[0]
		self.local_temp[5] += list(lanearea["7216_EW_2"].values())[0]
		self.local_temp[5] += list(lanearea["7216_EW_1"].values())[0]
		self.local_temp[5] += list(lanearea["7216_EW_0"].values())[0]
		self.local_temp[6] += list(lanearea["7216_NS_3"].values())[0]
		self.local_temp[6] += list(lanearea["7216_NS_2"].values())[0]
		self.local_temp[7] += list(lanearea["7216_NS_1"].values())[0]
		self.local_temp[7] += list(lanearea["7216_NS_0"].values())[0]

		self.temp.append(self.local_temp)

	def retrieve(self):
		return np.sum(np.array(self.temp), axis=0).astype(np.float32)

