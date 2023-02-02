import numpy as np

class state_v11:
	'''
	Signal 7216 lanearea detectors queue sensor
	Each direction have 2 values, through and left turn, hence size is 8
	'''
	def __init__(self):
		self.size = 152
		self.reset_size = 111
		self.temp = []
		self.edge_reduced = False
		self.edge_full = False
		self.edge_full_side = True
		self.induction = True
		self.lanearea = False
		self.multi = False

	def __len__(self):
		return self.size

	def set(self, value):
		return np.full(self.size, value).astype(np.float32)

	def reset(self):
	    self.temp = np.zeros(self.reset_size)
	    self.main = 0
	    self.side = 0

	def update(self, loop, edge, lanearea, multi, env):
		for i in env.EB_edgeIDs:
			if i in edge.keys():
				self.main += list(edge[i].values())[0]

		for i in env.WB_edgeIDs:
			if i in edge.keys():
				self.main += list(edge[i].values())[0]

		for i in env.NS_edgeIDs:
			if i in edge.keys():
				self.side += list(edge[i].values())[0]

		for i in env.SB_edgeIDs:
			if i in edge.keys():
				self.side += list(edge[i].values())[0]

		for idx, i in enumerate(edge.keys()):
			self.temp[idx] += list(edge[i].values())[0]

	def retrieve(self):
		return np.array(self.temp).astype(np.float32)

