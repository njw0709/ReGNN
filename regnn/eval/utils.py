import stata_setup

# Track Stata initialization status
_stata_initialized = False


def init_stata():
    global _stata_initialized
    if not _stata_initialized:
        stata_setup.config("/usr/local/stata17", "mp")
        _stata_initialized = True
    from pystata import stata

    return stata
