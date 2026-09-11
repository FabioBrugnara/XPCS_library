class ConfigDict(dict):

    def __getitem__(self, key):
        if key == "Npx":
            nx, ny = self.get("Nx"), self.get("Ny")
            if nx is None or ny is None:
                raise ValueError("Both 'Nx' and 'Ny' must be set before accessing 'Npx'.")
            return nx * ny
        return super().__getitem__(key)

    def __setitem__(self, key, value):
            if key == "movement_axis" and value is not None:
                if str(value).upper() not in ("X", "Y"):
                    raise ValueError(f"Invalid movement_axis '{value}'. Must be 'X' or 'Y'.")
                value = str(value).upper()  # Ensures uppercase 'X' or 'Y'
            super().__setitem__(key, value)

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

    "of_value4plot": 2**32 - 1,
})

def set_config(key, value=None):
    if isinstance(key, dict):
        config.update(key)
    else:
        config[key] = value