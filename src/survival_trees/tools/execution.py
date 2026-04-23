import datetime
import os
import sys
import time
import warnings
from contextlib import contextmanager
from functools import wraps


class ColorsOut:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    Red = '\033[91m'
    Green = '\033[92m'
    Blue = '\033[94m'
    Cyan = '\033[96m'
    White = '\033[97m'
    Yellow = '\033[93m'
    Magenta = '\033[95m'
    Grey = '\033[90m'
    Black = '\033[90m'
    Default = '\033[99m'
    DEFAULT = Blue
    TIME = Grey


def print_(output, color="DEFAULT"):
    print(
        f"""{ColorsOut.__getattribute__(ColorsOut, color)}{output}{ColorsOut.ENDC}""")


def execution_time(method):
    @wraps(method)
    def timed(*args, **kw):
        from memory_profiler import memory_usage
        ts = time.time()
        mem, result = memory_usage((method, args, kw), retval=True, timeout=200,
                                   interval=1e-7)
        te = time.time()

        msg = "[" + method.__name__ + "] execution time :"
        msg += "-" * (40 - len(msg)) + "  "
        msg += str(datetime.timedelta(milliseconds=(te - ts) * 1000))
        msg += "  " + f'Memory {int(max(mem) - min(mem))}' + " MiB"
        if hasattr(args[0], "active_execution_time"):
            if not args[0].active_execution_time:
                return result
            else:
                print_(msg, color="TIME")
        else:
            print_(msg, color="TIME")
        return result

    return timed


@contextmanager
def silence_stdout():
    stdout_stream = open(os.devnull, "w")
    stderr_stream = open(os.devnull, "w")

    old_target_std_out = sys.stdout
    old_target_std_err = sys.stderr

    sys.stdout = stdout_stream
    sys.stderr = stderr_stream
    try:
        yield stdout_stream, stderr_stream
    finally:
        sys.stdout = old_target_std_out
        sys.stderr = old_target_std_err


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
