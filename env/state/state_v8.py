import numpy as np

class state_v8:
	'''
	Signal 7216 lanearea detectors queue sensor
	Each direction have 2 values, through and left turn, hence size is 8
	'''
	def __init__(self):
		self.size = 30
		self.reset_size = 22
		self.temp = []
		self.edge_reduced = True
		self.edge_full = False
		self.edge_full_side = False
		self.induction = True
		self.lanearea = False
		self.multi = False

	def __len__(self):
		return self.size

	def set(self, value):
		return np.full(self.size, value).astype(np.float32)

	def reset(self):
	    self.temp = np.zeros(self.reset_size)

	def update(self, loop, edge, lanearea, multi, env):
		for idx, i in enumerate(env.edge_dic["EB_edgeIDs_reduced"]):
			if isinstance(i, tuple):
				temp = 0
				for inner in i:
					temp += list(edge[inner].values())[0]
				self.temp[idx] += temp
			else:
				self.temp[idx] += list(edge[i].values())[0]

		for idx, i in enumerate(env.edge_dic["WB_edgeIDs_reduced"]):
			if isinstance(i, tuple):
				temp = 0
				for inner in i:
					temp += list(edge[inner].values())[0]
				self.temp[idx + 11] += temp
			else:
				self.temp[idx + 11] += list(edge[i].values())[0]

	def retrieve(self):
		return np.array(self.temp).astype(np.float32)

