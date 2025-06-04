__version__ = "0.1.3"
__author__ = "Sviatoslav"


_globals = globals()

if _globals.get('__name__') and _globals['__name__'] != '<string>':
    from .multivex import (
        scan,
    )

    __all__ = [
        "scan",
        "__version__",
        "__author__",
    ]


del _globals
