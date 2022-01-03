def add_dict(c, d):
    # Register new key-value pairs
    for k, v in d.items():
        c.__dict__[k] = v
    return c