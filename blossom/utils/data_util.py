# -*- coding: utf-8 -*-

import os

def str2bool(value):
    return str(value).lower() in ('yes', 'true', 't', '1')

def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()

    if key in registry:
        return registry[key]
    else:
        raise ValueError(f"Key `{key}` not supported, available options: {registry.keys()}")
