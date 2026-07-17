"""The one expected-failure exception, caught at the CLI edge."""


class FatalError(Exception):
    """An expected failure with a user-facing message: printed as one line by main(),
    never a traceback. Tracebacks are for bugs."""
