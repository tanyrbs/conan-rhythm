from . import hifigan

_HIFIGAN_NSF_IMPORT_ERROR = None

try:
    from . import hifigan_nsf
except Exception as exc:
    # Training / structural debugging can proceed without NSF vocoder extras,
    # even when the optional import fails due to ABI / binary loader issues.
    hifigan_nsf = None
    _HIFIGAN_NSF_IMPORT_ERROR = exc
