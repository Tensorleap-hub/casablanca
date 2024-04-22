import os
from typing import Dict, Any
import yaml


def load_od_config() -> Dict[str, Any]:
    # Load the existing YAML config
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['local_storage_path'] = os.path.join(os.getenv("HOME"), "tensorleap", "data", config['BUCKET_NAME'])

    return config


CONFIG = load_od_config()
