class FunctionCache(dict):
    """A simple dict-subclass for caching the return values of a function."""

    def __init__(self, func):
        self.func = func

    def __missing__(self, key):
        value = self[key] = self.func(key)
        return value


class cached_property:
    """A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.

    Source:
    https://github.com/pydanny/cached-property
    https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
