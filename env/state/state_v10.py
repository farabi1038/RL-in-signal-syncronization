import numpy as np

class state_v10:
	'''
	Signal 7216 lanearea detectors queue sensor
	Each direction have 2 values, through and left turn, hence size is 8
	'''
	def __init__(self):
		self.size = 63
		self.reset_size = 22
		self.temp = []
		self.edge_reduced = False
		self.edge_full = False
		self.edge_full_side = False
		self.induction = True
		self.lanearea = False
		self.multi = True

	def __len__(self):
		return self.size

	def set(self, value):
		return np.full(self.size, value).astype(np.float32)

	def reset(self):
	    self.temp = np.zeros(self.reset_size)

	def update(self, loop, edge, lanearea, multi, env):
		for i in multi:
			env.multi_dic[i].compute(multi[i][18])
			self.temp[0] = env.multi_dic[i].volume
		

	def retrieve(self):
		return np.array(self.temp).astype(np.float32)
