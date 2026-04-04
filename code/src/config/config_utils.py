from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf

class DictToClass:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
            
def load_config(config_path: str, env: Optional[str] = None, to_class: bool = True):
    config_path = Path(config_path)
    if not env:
        config = OmegaConf.load(config_path / 'default.yaml')
    else:
        config = OmegaConf.load(config_path / f'{env}.yaml')
    if not to_class:
        return config
    return DictToClass(config)

def combine_configs(*configs, names=["default", "head", "heart", "wrist"]):
    """Combine multiple configs into a single config."""
    combined = {}
    for name, config in zip(names, configs):
        combined[name] = config

if __name__ == '__main__':
    config = load_config('default.yaml')
    print(config)