import ast
import inspect
import sys
import types
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")


def rewrite_ndreduce(func: Callable[P, R]) -> Callable[P, R]:
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
_TRANSFORMED_FUNC_NAME = "__numbagg_transformed_func"


def _apply_ast_rewrite(
    func: Callable[P, R], node_transformer: "_NDReduceTransformer"
) -> Callable[P, R]:
    """A hack to make the syntax for writing aggregators more Pythonic.

    This should go away once numba is more fully featured.
    """
    orig_source: str = inspect.getsource(func)

    tree: ast.Module = ast.parse(orig_source)
    tree = node_transformer.visit(tree)
    ast.fix_missing_locations(tree)
    source = compile(tree, filename="<ast>", mode="exec")

    scope: dict[str, Callable[P, R]] = {}
    # Cast to FunctionType to access __globals__ attribute
    func_obj = cast(types.FunctionType, func)
    exec(source, func_obj.__globals__, scope)
    try:
        return scope[_TRANSFORMED_FUNC_NAME]
    except KeyError:
        raise TypeError(f"failed to rewrite function definition:\n{orig_source}")


class _NDReduceTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        args: list[ast.arg] = node.args.args + [ast.arg(arg=_OUT_NAME, annotation=None)]
        arguments = ast.arguments(
            args=args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
            posonlyargs=[],
        )
        # Python 3.12+ requires type_params argument
        if sys.version_info >= (3, 12):
            function_def = ast.FunctionDef(
                name=_TRANSFORMED_FUNC_NAME,
                args=arguments,
                body=node.body,
                decorator_list=[],
                returns=None,
                type_comment=None,
                type_params=[],
            )
        else:
            function_def = ast.FunctionDef(
                name=_TRANSFORMED_FUNC_NAME,
                args=arguments,
                body=node.body,
                decorator_list=[],
                returns=None,
                type_comment=None,
            )
        return self.generic_visit(function_def)

    def visit_Return(self, node):
        subscript = ast.Subscript(
            value=ast.Name(id=_OUT_NAME, ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Store(),
        )
        assign = ast.Assign(targets=[subscript], value=node.value)
        return assign
