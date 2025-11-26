import yaml
import json


def load_config():
    with open("../config/config.yaml") as f:
        return yaml.safe_load(f)


def get_config_hash(config):
    import hashlib
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]