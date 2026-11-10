from collections import UserDict

class XPCStoolsConfig(UserDict):
    def __init__(self, initialdata=None, **kwargs):
        super().__init__(initialdata, **kwargs)
        # Valori di default
        self.data.update({
            "Ei": None,
            "Nx": None,
            "Ny": None,
            "lxp": None,
            "lyp": None,
            "X0": None,
            "Y0": None,
            "L": None,
            "movement_axis": None,
            "itime": None,
            "ofvalue": 2**32,
        })

    def __setitem__(self, key, value):
        if key not in self.data:
            raise AttributeError(f"Configuration has no parameter '{key}'")

        if key == "movement_axis" and value is not None:
            val_str = str(value).upper()
            if val_str not in ("X", "Y", "-X", "-Y"):
                raise ValueError(f"Invalid movement_axis '{value}'. Must be 'X', 'Y', '-X', or '-Y'.")
            value = val_str

        super().__setitem__(key, value)

    # Permette di accedere anche tramite attributi (es. config.Nx)
    def __getattr__(self, key):
        if key == "data":
            return super().__getattribute__(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'XPCStoolsConfig' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "data":
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __repr__(self):
        return f"XPCStoolsConfig({self.data})"

# Istanza globale pulita
config = XPCStoolsConfig()

def set_config(key, value=None):
    if isinstance(key, dict):
        config.update(key)
    else:
        setattr(config, key, value)