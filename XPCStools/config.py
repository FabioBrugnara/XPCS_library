import sys
from types import ModuleType


class ConfigDict(dict):

    def __getitem__(self, key):
        if key == "Npx":
            nx, ny = self.get("Nx"), self.get("Ny")
            if nx is None or ny is None:
                raise ValueError("Both 'Nx' and 'Ny' must be set before accessing 'Npx'.")
            return nx * ny
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key == "Npx":
            raise ValueError("'Npx' is a read-only computed property from 'Nx' and 'Ny'.")
        if key == "movement_axis" and value is not None:
            if str(value).upper() not in ("X", "Y", "-X", "-Y"):
                raise ValueError(f"Invalid movement_axis '{value}'. Must be 'X' or 'Y'.")
            value = str(value).upper()  # Ensures uppercase 'X' or 'Y'
        super().__setitem__(key, value)

    def __contains__(self, key):
        if key == "Npx":
            return True
        return super().__contains__(key)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.keys()) | {"Npx"})

    def update(self, *args, **kwargs):
        other = dict(*args, **kwargs)
        for k, v in other.items():
            self[k] = v


# Instantiate as the central config
config = ConfigDict({

    "Ei": None,  # Beam energy in keV

    "Nx": None,
    "Ny": None,
    "lxp": None,
    "lyp": None,

    "X0": None,
    "Y0": None,
    "L": None,
    "movement_axis": None,
    
    "itime": None,

    "ofvalue": 2**32 - 1,
})

def set_config(key, value=None):
    if isinstance(key, dict):
        config.update(key)
    else:
        config[key] = value


class _ConfigModule(ModuleType):
    """Module wrapper allowing direct access/assignment to config keys on the config module."""

    def __getattr__(self, name: str):
        if name in config or name == "Npx":
            return config[name]
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name in config or name == "Npx":
            config[name] = value
        else:
            super().__setattr__(name, value)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(config.keys()) | {"Npx"})


def __getattr__(name: str):
    if name in config or name == "Npx":
        return config[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals().keys()) | set(config.keys()) | {"Npx"})


sys.modules[__name__].__class__ = _ConfigModule
