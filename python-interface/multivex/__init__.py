__version__ = "0.1.0"
__author__ = "Sviatoslav"


_globals = globals()

if _globals.get('__name__') and _globals['__name__'] != '<string>':
    from .multivex import (
        scan,
        reduce,
        stream_compaction,
        sort1
    )

    __all__ = [
        "scan",
        "reduce",
        "stream_compaction",
        "sort1",
        "__version__",
        "__author__",
    ]


del _globals
