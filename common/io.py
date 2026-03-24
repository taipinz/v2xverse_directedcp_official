from typing import Dict
import os
import logging
from pathlib import Path

import yaml


def _resolve_external_path_value(value, repo_root: Path):
    if isinstance(value, str) and value.startswith('external_paths/'):
        return str((repo_root / value).resolve(strict=False))
    if isinstance(value, list):
        return [_resolve_external_path_value(item, repo_root) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_external_path_value(val, repo_root) for key, val in value.items()}
    return value


def load_config_from_yaml(cfg_file : str, verbose : bool=True) -> Dict:
    """Load YAML config"""
    if not os.path.exists(cfg_file):
        if verbose:
            logging.error(f'{cfg_file} does not exist.')
        return None
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _resolve_external_path_value(cfg, repo_root)
    if verbose:
        logging.info(f'config at {cfg_file} loaded')

    return cfg
