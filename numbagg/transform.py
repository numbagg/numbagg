import inspect
import re


def _apply_source_transform(func, transform_source):
    """A horrible hack to make the syntax for writing aggregators more
    Pythonic.

    This should go away once numba is more fully featured.
    """
    orig_source = inspect.getsource(func)
    source = transform_source(orig_source)
    scope = {}
    exec(source, func.func_globals, scope)
    try:
        return scope['__transformed_func']
    except KeyError:
        raise TypeError('failed to rewrite function definition:\n%s'
                        % orig_source)


def _transform_agg_source(func):
    """Transforms aggregation functions into something numba can handle.

    To be more precise, it converts functions with source that looks like

        @ndreduce
        def my_func(x)
            ...
            return foo

    into

        def __sub__gufunc(x, __out):
            ...
            __out[0] = foo

    which is the form numba needs for writing a gufunc that returns a scalar
    value.
    """
    def transform_source(source):
        # nb. the right way to do this would be use Python's ast module instead
        # of regular expressions.
        source = re.sub(
            r'^@ndreduce[^\n]*\ndef\s+[a-zA-Z_][a-zA-Z_0-9]*\((.*?)\)\:',
            r'def __transformed_func(\1, __out):', source, flags=re.DOTALL)
        source = re.sub(r'return\s+(.*)', r'__out[0] = \1', source)
        return source
    return _apply_source_transform(func, transform_source)


def _transform_moving_source(func):
    """Transforms moving aggregation functions into something numba can handle.
    """
    def transform_source(source):
        source = re.sub(
            r'^@ndmoving[^\n]*\ndef\s+[a-zA-Z_][a-zA-Z_0-9]*\((.*?)\)\:',
            r'def __transformed_func(\1):', source, flags=re.DOTALL)
        source = re.sub(r'^(\s+.*)(window)', r'\1window[0]', source,
                        flags=re.MULTILINE)
        return source
    return _apply_source_transform(func, transform_source)
