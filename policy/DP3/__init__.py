"""Load legacy deployment only when its API is requested."""
from importlib import import_module

__all__ = ["get_model", "encode_obs", "eval", "reset_model"]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module(".deploy_policy", __name__), name)
    raise AttributeError(name)
