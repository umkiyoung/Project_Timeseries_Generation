import yaml
import importlib

def cycle(dl):
    while True:
        for data in dl:
            yield data

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    
    return config

def instantiate_from_config(config):
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    
    return cls(**config.get("params", dict()))

def exists(x):
    return x is not None

def default(val, d):
    # used
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    # used
    return t

def extract(a, t, x_shape):
    # used
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
