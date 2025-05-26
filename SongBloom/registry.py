from typing import Any, Optional
import os, sys
from omegaconf import MISSING, OmegaConf,DictConfig

sys.path.insert(0, os.path.dirname(__file__))



def load_config() -> DictConfig:
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: OmegaConf.load(x))
    cmd_cfg = OmegaConf.from_cli()
    
    cfg_file_path = cmd_cfg.get("cfg_file", None) 
    file_cfg = OmegaConf.load(open(cfg_file_path, 'r')) if cfg_file_path is not None \
                else OmegaConf.create()
    
    cfgs = OmegaConf.merge(file_cfg, cmd_cfg)
    # OmegaConf.resolve(cfgs)

    return cfgs
