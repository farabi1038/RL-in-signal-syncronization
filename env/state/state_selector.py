from .state_v1 import state_v1
from .state_v2 import state_v2
from .state_v3 import state_v3
from .state_v4 import state_v4
from .state_v5 import state_v5
from .state_v6 import state_v6
from .state_v7 import state_v7
from .state_v8 import state_v8
from .state_v9 import state_v9
from .state_v10 import state_v10
from .state_v11 import state_v11

def state_selector(cfg):
	if cfg.state.state.name == "v1":
		return state_v1()
	elif cfg.state.state.name == "v2":
		return state_v2()
	elif cfg.state.state.name == "v3":
		return state_v3()
	elif cfg.state.state.name == "v4":
		return state_v4()
	elif cfg.state.state.name == "v5":
		return state_v5()
	elif cfg.state.state.name == "v6":
		return state_v6()
	elif cfg.state.state.name == "v7":
		return state_v7()
	elif cfg.state.state.name == "v8":
		return state_v8()
	elif cfg.state.state.name == "v9":
		return state_v9()
	elif cfg.state.state.name == "v10":
		return state_v10()
	elif cfg.state.state.name == "v11":
		return state_v11()