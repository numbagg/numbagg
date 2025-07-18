import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from numbagg import GROUPED_FUNCS
from numbagg.grouped import (
    group_nanall,
    group_nanany,
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmax,
    group_nanmean,
    group_nanmin,
    group_nanprod,
    group_nanstd,
    group_nansum,
    group_nansum_of_squares,
    group_nanvar,
)
from numbagg.test.conftest import COMPARISONS

FUNCTIONS = [
    (group_nanall, lambda x: x.all(), None),
    (group_nanany, lambda x: x.any(), None),
    (group_nanargmax, lambda x: x.idxmax(), np.nanargmax),
    (group_nanargmin, lambda x: x.idxmin(), np.nanargmin),
    (group_nancount, lambda x: x.count(), None),
    (group_nanfirst, lambda x: x.first(), None),
    (group_nanlast, lambda x: x.last(), None),
    (group_nanmax, lambda x: x.max(), np.nanmax),
    (group_nanmean, lambda x: x.mean(), np.nanmean),
    (group_nanmin, lambda x: x.min(), np.nanmin),
    (group_nanprod, lambda x: x.prod(), np.nanprod),
    (group_nanstd, lambda x: x.std(), np.nanstd),
    (group_nansum, lambda x: x.sum(), np.nansum),
    (group_nanvar, lambda x: x.var(), np.nanvar),
    (
        group_nansum_of_squares,
        lambda x: x.agg(lambda y: (y**2).sum()),
        lambda x: np.nansum(x**2),
    ),
]

# Functions which return the same scalar if one is passed
FUNCTIONS_CONSTANT = [
    fs
    for fs in FUNCTIONS
    if fs[0]
    in {
        group_nanmean,
        group_nansum,
        group_nanfirst,
        group_nanlast,
        group_nanprod,
        group_nanmin,
        group_nanmax,
    }
]


@pytest.fixture(autouse=True)
def silence_pandas_idx_warnings():
    # Not sure whether we adopt this behavior, but no need to litter with
    # warnings in the meantime...
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of Series.* with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError",
        )
        yield


@pytest.fixture(params=[np.float64, np.int32, np.bool_], scope="module")
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def labels(rs):
    # `-1` is a special value for missing labels
    labels = rs.choice([-1, 0, 1, 2, 3, 4], size=200)
    # The tests are dependent on this being dense
    assert len(np.unique(labels)) == 6
    return labels


@pytest.fixture(scope="module")
def values(rs, labels, dtype):
    if dtype == np.int32:
        return rs.randint(-100, 100, size=200)
    elif dtype == np.float64:
        vals = rs.rand(200)
        vals = np.where(vals > 0.1, vals, np.nan)
        # Have one group all missing
        return np.where(labels != 2, vals, np.nan)
    elif dtype == np.bool_:
        return rs.choice([True, False], size=200)
    else:
        raise ValueError(f"dtype {dtype} not supported")


@pytest.mark.parametrize("numbagg_func, pandas_func, _", FUNCTIONS)
def test_group_pandas_comparison(values, labels, numbagg_func, pandas_func, _, dtype):
    """
    Old test approach, directly parametrizing over functions. Can be removed once we're
    confident the new approach covers everything.
    """
    # Pandas uses `NaN` rather than `-1` for missing labels
    pandas_labels = np.where(labels >= 0, labels, np.nan)
    expected = pandas_func(pd.Series(values).groupby(pandas_labels))
    if dtype in [np.int32, np.int64]:
        if not numbagg_func.supports_ints:
            pytest.skip(f"{numbagg_func} doesn't support ints")
        if numbagg_func == group_nanprod:
            pytest.skip("group_nanprod result too large")
        result = numbagg_func(values, labels)
        assert_almost_equal(result, expected.values.astype(np.int32))
    elif dtype == np.bool_:
        if not numbagg_func.supports_bool:
            pytest.skip(f"{numbagg_func} doesn't support bools")
    else:
        result = numbagg_func(values, labels)
        assert_almost_equal(result, expected.values)


@pytest.mark.parametrize(
    "func",
    GROUPED_FUNCS,
)
@pytest.mark.parametrize("shape", [(1, 500)], indirect=True)
def test_group_pandas_comp(array, func):
    """
    New test approach, using `COMPARISONS`
    """
    c = COMPARISONS[func]

    result = c["numbagg"](array)()
    expected_pandas = c["pandas"](array)().values

    assert_allclose(result, expected_pandas)


@pytest.mark.parametrize(
    "func",
    [f for f in GROUPED_FUNCS if f.supports_ddof],
)
@pytest.mark.parametrize("shape", [(1, 500)], indirect=True)
@pytest.mark.parametrize("ddof", [0, 1, 5])
def test_group_pandas_comp_ddof(array, func, ddof):
    c = COMPARISONS[func]

    result = c["numbagg"](array, ddof=ddof)()
    expected_pandas = c["pandas"](array, ddof=ddof)().values

    assert_allclose(result, expected_pandas)


@pytest.mark.parametrize("numbagg_func, pandas_func, _", FUNCTIONS)
def test_all_nan_for_label(numbagg_func, pandas_func, _):
    values = np.array([1.0, 2.0, np.nan, np.nan])
    group = np.array([0, 0, 1, 1])
    expected = pandas_func(pd.Series(values).groupby(group))
    result = numbagg_func(values, group)
    assert_almost_equal(result, expected.values)


@pytest.mark.parametrize("numbagg_func, pandas_func, _", FUNCTIONS)
def test_no_valid_labels(numbagg_func, pandas_func, _):
    values = np.array([1.0, 2.0, 3.0, 4.0])
    group = np.array([-1, -1, -1, -1])

    # Replace -1 with nan for pandas group
    pandas_group = group.copy().astype(float)
    pandas_group[pandas_group == -1] = np.nan

    expected = pandas_func(pd.Series(values).groupby(pandas_group))
    result = numbagg_func(values, group)
    assert_almost_equal(result, expected.values)


@pytest.mark.parametrize("numbagg_func, pandas_func, _", FUNCTIONS)
def test_single_nan_for_label(numbagg_func, pandas_func, _):
    values = np.array([1.0, 2.0, np.nan])
    group = np.array([0, 0, 1])
    expected = pandas_func(pd.Series(values).groupby(group))
    result = numbagg_func(values, group)
    assert_almost_equal(result, expected.values)


@pytest.mark.parametrize("numbagg_func, pandas_func, _", FUNCTIONS)
def test_all_values_are_nan(numbagg_func, pandas_func, _):
    values = np.array([np.nan, np.nan, np.nan])
    group = np.array([0, 1, 2])
    expected = pandas_func(pd.Series(values).groupby(group))
    result = numbagg_func(values, group)
    assert_almost_equal(result, expected.values)


def groupby_mean_pandas(values, group):
    # like pd.Series(values).groupby(group).mean()
    labels, uniques = pd.factorize(group, sort=True)
    result = group_nanmean(values, labels, num_labels=len(uniques))
    return pd.Series(result, index=uniques)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.bool_, np.int32, np.int64], indirect=True
)
def test_groupby_mean_types(dtype):
    rs = np.random.RandomState(0)
    values = rs.rand(2000).astype(dtype)
    group = rs.choice([np.nan, 1, 2, 3, 4, 5], size=values.shape)
    expected = pd.Series(values).groupby(group).mean()
    result = groupby_mean_pandas(values, group)
    assert_almost_equal(result, expected.values)  # type: ignore[arg-type,unused-ignore]


@pytest.mark.parametrize(
    "numbagg_func, pandas_func, exp",
    [
        (group_nansum, lambda x: x.sum(), 0),
        (group_nanprod, lambda x: x.prod(), 1),
        (group_nanargmin, lambda x: x.idxmin(), np.nan),
    ],
)
def test_groupby_empty_numeric_operations(numbagg_func, pandas_func, exp):
    values = np.array([np.nan, np.nan, np.nan])
    group = np.array([0, 1, 2])
    expected_array = np.array([exp] * len(values))
    expected = pandas_func(pd.Series(values).groupby(group))

    result = numbagg_func(values, group)

    np.testing.assert_equal(result, expected_array)
    assert_almost_equal(result, expected.values)


@pytest.mark.parametrize("func", [f[0] for f in FUNCTIONS])
def test_additional_dim_equivalence(func, values, labels, dtype):
    if dtype == np.bool_ and not func.supports_bool:
        pytest.skip(f"{func} doesn't support bools")
    if dtype in [np.int32, np.int64]:
        if not func.supports_ints:
            pytest.skip(f"{func} doesn't support ints")
    values = values[:10]
    labels = labels[:10]
    expected = func(values, labels)
    values_2d = np.tile(values[:10], (2, 1))
    result = func(values_2d, labels, axis=1)[0]
    assert_almost_equal(result, expected)


@pytest.mark.parametrize("labels_type", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("func, _, npfunc", [f for f in FUNCTIONS_CONSTANT])
def test_group_func_axis_1d_labels(func, _, npfunc, labels_type):
    if npfunc is None:
        pytest.skip("No numpy equivalent")

    values = np.arange(5.0)
    labels = np.arange(5, dtype=labels_type)
    result = func(values, labels)
    assert_almost_equal(result, values)

    values = np.arange(25.0).reshape(5, 5)
    labels = np.arange(5)

    with pytest.raises(ValueError) as excinfo:
        result = func(values, labels)
    assert "axis required" in str(excinfo.value)

    result = func(values, labels, axis=1)
    assert_almost_equal(result, values)

    result = func(values, labels, axis=(1,))
    assert_almost_equal(result, values)

    result = func(values, labels, axis=0)
    assert_almost_equal(result, values.T)

    with pytest.raises(ValueError) as excinfo:
        func(values, labels[:4], axis=0)
    assert "must have same shape" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        func(values, labels[:4], axis=(0,))
    assert "must have same shape" in str(excinfo.value)

    result = func(values, [0, 0, -1, 1, 1], axis=1)
    expected = np.stack(
        [npfunc(values[:, :2], axis=1), npfunc(values[:, 3:], axis=1)], axis=-1
    )
    assert_almost_equal(result, expected)


@pytest.mark.parametrize("func", [f[0] for f in FUNCTIONS_CONSTANT])
def test_group_axis_2d_labels(func):
    values = np.arange(25.0).reshape(5, 5)
    labels = np.arange(25).reshape(5, 5)
    result = func(values, labels)
    assert_almost_equal(result, values.ravel())

    values = np.arange(125.0).reshape(5, 5, 5)
    result = func(values, labels, axis=(1, 2))
    assert_almost_equal(result, values.reshape(5, -1))


def test_numeric_int_nancount():
    values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)

    result = group_nancount(values, labels)
    assert_almost_equal(result, np.array([3, 2]))


def test_numeric_int_nanmean():
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)

    result = group_nanmean(values, labels)
    assert_almost_equal(result, np.array([2, 4, 6]))


def test_numeric_int_nanmin():
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)

    result = group_nanmin(values, labels)
    assert_almost_equal(result, np.array([1, 4, 6]))


def test_numeric_int_nanprod():
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)

    result = group_nanprod(values, labels)
    assert_almost_equal(result, np.array([6, 20, 6]))

    values = np.array([1, 2, -3, 4, 5, 6], dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)

    result = group_nanprod(values, labels)
    assert_almost_equal(result, np.array([-6, 20, 6]))


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("func", [group_nanmean, group_nansum])
@pytest.mark.parametrize("n", [127, 128, 255, 256, 1000])
def test_int8(n, func, dtype):
    data = np.random.randn(n)
    assert_almost_equal(
        getattr(np, func.__name__.removeprefix("group_nan"))(data),
        func(data, np.zeros((n,), dtype=dtype))[0],
    )


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("func", [group_nanmean, group_nansum])
def test_int8_again(dtype, func):
    array = np.array(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 9, 9, 9, 9, 10],
        ],
    )
    by = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=dtype)

    expected = getattr(
        pd.DataFrame(array.T).groupby(by), func.__name__.removeprefix("group_nan")
    )().T.astype(dtype)

    # https://github.com/numbagg/numbagg/issues/213
    assert_almost_equal(func(array, by, axis=-1), expected)
    # Amazingly it can also be more incorrect with another run!
    assert_almost_equal(func(array, by, axis=-1), expected)


def test_dimensionality():
    func = group_nansum

    values = np.arange(6)
    labels = np.arange(6)

    result = func(values, labels)
    assert result.shape == (6,)

    result = func(values, labels, axis=-1)
    assert result.shape == (6,)

    result = func(values.reshape(1, 6), labels, axis=-1)
    assert result.shape == (1, 6)

    with pytest.raises(ValueError):
        func(values.reshape(1, 6), labels)

    values = np.arange(24).reshape(6, 4)
    labels = np.arange(4)

    result = func(values, labels, axis=-1)
    assert result.shape == (6, 4)

    result = func(values.reshape(1, 6, 4), labels, axis=-1)
    assert result.shape == (1, 6, 4)

    result = func(values.reshape(1, 6, 2, 2), labels.reshape(2, 2), axis=(-1, -2))
    assert result.shape == (1, 6, 4)

    with pytest.raises(ValueError):
        func(values, labels)
