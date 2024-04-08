import os
from typing import Dict, Any
import yaml


def load_config() -> Dict[str, Any]:
    # Load the existing YAML config
    config_file_name = 'config.yml'
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, config_file_name)
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    def secs2snapshots(seconds: int) -> int:
        sec2snapshot = 1 / config['time_resolution']
        return int(seconds * sec2snapshot)

    # update resolution dependant properties
    config['sec2seconds'] = 1 / config['time_resolution']
    config['past_window_rows'] = secs2snapshots(config['past_window_seconds'])
    config['future_window_rows'] = secs2snapshots(config['future_window_seconds'])
    config['future_window_width_rows'] = secs2snapshots(config['future_window_width_seconds'])
    stride_sec = secs2snapshots(config['time_window_stride_seconds'])
    if stride_sec == 'same':
        config['time_window_stride_rows'] = 1
    else:
        config['time_window_stride_rows'] = secs2snapshots(config['time_window_stride_seconds'])

    if config['time_resolution'] == 1.0:
        config['start_date'] = '2019-01-15 14:45:01.00'
    elif config['time_resolution'] == 0.2:
        config['start_date'] = '2019-01-15 14:45:00.20'
    else:
        raise ValueError(f" time resolution should be in [1.0, 0.2], got {config['time_resolution']}")
    # Write the updated config back to the YAML file
    # with open(config_file_name, 'w') as file:
    #     yaml.dump(config, file)

    return config


config = load_config()
