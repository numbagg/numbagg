import ast
import inspect


def rewrite_ndreduce(func):
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
    return _apply_ast_rewrite(func, _NDReduceTransformer())


_OUT_NAME = "__numbagg_out"
_TRANFORMED_FUNC_NAME = "__numbagg_transformed_func"

import astor


def _apply_ast_rewrite(func, node_transformer):
    """A hack to make the syntax for writing aggregators more Pythonic.

    This should go away once numba is more fully featured.
    """
    orig_source = inspect.getsource(func)

    tree = ast.parse(orig_source)
    tree = node_transformer.visit(tree)
    ast.fix_missing_locations(tree)
    # print(ast.dump(tree, indent=4))
    # print(astor.to_source(tree))
    source = compile(tree, filename="<ast>", mode="exec")

    scope: dict = {}
    exec(source, func.__globals__, scope)
    try:
        return scope[_TRANFORMED_FUNC_NAME]
    except KeyError:
        raise TypeError("failed to rewrite function definition:\n%s" % orig_source)


class _NDReduceTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        args = node.args.args + [ast.arg(arg=_OUT_NAME, annotation=None)]
        arguments = ast.arguments(
            args=args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
            posonlyargs=[],
        )
        function_def = ast.FunctionDef(
            name=_TRANFORMED_FUNC_NAME,
            args=arguments,
            body=node.body,
            decorator_list=[],
        )
        return self.generic_visit(function_def)

    def visit_Return(self, node):
        subscript = ast.Subscript(
            value=ast.Name(id=_OUT_NAME, ctx=ast.Load()),
            slice=ast.Index(value=ast.Num(n=0)),
            ctx=ast.Store(),
        )
        assign = ast.Assign(targets=[subscript], value=node.value)
        return assign
