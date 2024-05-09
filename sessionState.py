# sessionstate.py

class _SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class SessionState:
    instance = None

    def __init__(self, **kwargs):
        if not SessionState.instance:
            SessionState.instance = _SessionState(**kwargs)
        else:
            for key, value in kwargs.items():
                setattr(SessionState.instance, key, value)

def get(**kwargs):
    return SessionState(**kwargs)
