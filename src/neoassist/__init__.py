from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neoassist")
except PackageNotFoundError:
    __version__ = "0+local"