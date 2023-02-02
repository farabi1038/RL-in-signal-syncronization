from .action_v1 import action_v1
from .action_v2 import action_v2
from .action_v3 import action_v3
from .action_v4 import action_v4
from .action_v5 import action_v5
from .action_v6 import action_v6
from .action_v7 import action_v7

def action_selector(cfg):
	if cfg.action.name == "v1":
		return action_v1(cfg)
	elif cfg.action.name == "v2":
		return action_v2(cfg)
	elif cfg.action.name == "v3":
		return action_v3(cfg)
	elif cfg.action.name == "v4":
		return action_v4(cfg)
	elif cfg.action.name == "v5":
		return action_v5(cfg)
	elif cfg.action.name == "v6":
		return action_v6(cfg)
	elif cfg.action.name == "v7":
		return action_v7(cfg)