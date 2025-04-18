import os
import pickle
from contextlib import contextmanager


class _SingletonDict(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def __getitem__(cls, key):
        return cls._instances[cls][key]

    def delitem(cls, key):
        del cls._instances[cls][key]


class ConfigRegistry(dict, metaclass=_SingletonDict):
    """
    Singleton registry of all Configurations
    """

    recent_config_name = None

    @staticmethod
    def loadpy(filename, **kwargs):
        with open(os.path.expandvars(filename)) as fh:
            return ConfigRegistry.loadpys(fh.read(), **kwargs)

    @staticmethod
    def loadpys(config_string, **kwargs):
        string_unixlf = config_string.replace("\r", "")
        exec(string_unixlf, kwargs)
        return ConfigRegistry.get_latest_config()

    @staticmethod
    def get_latest_config():
        return ConfigRegistry[ConfigRegistry.recent_config_name]

    def __init__(self):
        self.__dict__ = self

    @staticmethod
    @contextmanager
    def register_config(name=None, base=None):
        registry = ConfigRegistry()
        if base is not None:
            assert base in registry, (
                "no base configuration (%s) found in the registry" % base
            )
            config = registry[base].clone()
        else:
            config = Config()
        yield config
        if name is not None:
            registry[name] = config
            ConfigRegistry.recent_config_name = name

    @staticmethod
    def keys():
        registry = ConfigRegistry()
        return [k for k, v in registry.items()]

    @staticmethod
    def get(name):
        return ConfigRegistry[name]

    @staticmethod
    def clean():
        for k in ConfigRegistry.keys():
            ConfigRegistry.delitem(k)


class AttrDict(dict):
    """
    dict class that can address its keys as fields, e.g.
    d['key'] = 1
    assert d.key == 1
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def clone(self):
        result = AttrDict()
        for k, v in self.items():
            if isinstance(v, AttrDict):
                result[k] = v.clone()
            else:
                result[k] = v
        return result


class Config(AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loads(self, buff):
        rv = pickle.loads(buff)
        self.clear()
        self.update(rv)
        return self

    def clone(self):
        result = Config()
        for k, v in self.items():
            if isinstance(v, AttrDict):
                result[k] = v.clone()
            else:
                result[k] = v
        return result

    def dumps(self):
        return pickle.dumps(self)

    def load(self, filename):
        with open(os.path.expandvars(filename)) as fh:
            self.loads(fh.read())
        return self

    def dump(self, filename):
        with open(os.path.expandvars(filename), "w") as fh:
            return fh.write(self.dumps())

    def __str__(self):
        return "ShipGeoConfig:\n  " + "\n  ".join(
            [
                f"{k}: {self[k].__str__()}"
                for k in sorted(self.keys())
                if not k.startswith("_")
            ]
        )
