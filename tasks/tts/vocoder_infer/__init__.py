from . import hifigan

try:
    from . import hifigan_nsf
except ModuleNotFoundError:
    # Training / structural debugging can proceed without NSF vocoder extras.
    hifigan_nsf = None
