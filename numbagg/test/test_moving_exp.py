import numpy as np
import pytest
from numpy.testing import assert_allclose

from numbagg import (
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
)

from .conftest import COMPARISONS


@pytest.mark.parametrize(
    "func",
    [
        move_exp_nancount,
        move_exp_nanmean,
        move_exp_nanstd,
        move_exp_nansum,
        move_exp_nanvar,
        move_exp_nancov,
        move_exp_nancorr,
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 0.1])
@pytest.mark.parametrize("shape", [(3, 500)], indirect=True)
def test_move_exp_comp(array, alpha, func):
    c = COMPARISONS[func]

    result = c["numbagg"](array, alpha=alpha)()
    expected = c["pandas"](array, alpha=alpha)()

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "func",
    [
        move_exp_nancount,
        move_exp_nanmean,
        move_exp_nanstd,
        move_exp_nansum,
        move_exp_nanvar,
    ],
)
def test_move_exp_min_weight(func):
    # Make an array of 25 values, with the first 5 being NaN, and then look at the final
    # 19. We can't look at the whole series, because `nanvar` will always return NaN for
    # the first value.
    array = np.ones(25)
    array[:5] = np.nan

    # min_weight of 0 should produce values everywhere
    result = np.sum(~np.isnan(func(array, min_weight=0.0, alpha=0.2))[6:])
    expected = 19
    assert result == expected
    result = np.sum(~np.isnan(func(array, min_weight=0.0, alpha=0.8))[6:])
    expected = 19
    assert result == expected

    result = np.sum(~np.isnan(func(array, min_weight=0.5, alpha=0.2))[6:])
    expected = 17
    assert result == expected
    result = np.sum(~np.isnan(func(array, min_weight=0.5, alpha=0.8))[6:])
    expected = 19
    assert result == expected

    result = np.sum(~np.isnan(func(array, min_weight=0.9, alpha=0.2))[6:])
    expected = 10
    assert result == expected
    result = np.sum(~np.isnan(func(array, min_weight=0.9, alpha=0.8))[6:])
    expected = 19
    assert result == expected

    # min_weight of 1 should never produce values
    result = np.sum(~np.isnan(func(array, min_weight=1.0, alpha=0.2))[6:])
    expected = 0
    assert result == expected
    result = np.sum(~np.isnan(func(array, min_weight=1.0, alpha=0.8))[6:])
    expected = 0
    assert result == expected


@pytest.mark.parametrize("shape", [(10,), (200,)], indirect=True)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("test_nans", [True, False])
def test_move_exp_min_weight_numerical(alpha, array, test_nans):
    if not test_nans:
        array = np.nan_to_num(array)
    # High alphas mean fast decays, mean initial weights are higher
    initial_weight = alpha
    weights = (
        np.array([(1 - alpha) ** (i - 1) for i in range(array.size, 0, -1)])
        * initial_weight
    )
    assert_allclose(weights[-1], initial_weight)
    # Fill weights with NaNs where array has them
    weights = np.where(np.isnan(array), np.nan, weights)

    # This is the weight of the final value
    weight = np.nansum(weights)

    # Run with min_weight slightly above the final value required, assert it doesn't let
    # it through
    result = move_exp_nanmean(array, alpha=alpha, min_weight=weight + 0.01)
    assert np.isnan(result[-1])

    # And with min_weight slightly below
    result = move_exp_nanmean(array, alpha=alpha, min_weight=weight - 0.01)
    assert not np.isnan(result[-1])


def test_move_exp_nancount_numeric():
    array = np.array([1, 0, np.nan, np.nan, 1, 0])

    result = move_exp_nancount(array, alpha=0.5)
    expected = np.array([1.0, 1.5, 0.75, 0.375, 1.1875, 1.59375])
    assert_allclose(result, expected)

    result = move_exp_nancount(array, alpha=0.25)
    expected = np.array([1.0, 1.75, 1.3125, 0.984375, 1.7382812, 2.3037109])
    assert_allclose(result, expected)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_move_exp_nancount_nansum(alpha):
    # An array with NaNs & 1s should be the same for count & sum
    array = np.array([1, 1, np.nan, np.nan, 1, 1])

    result = move_exp_nancount(array, alpha=alpha)
    expected = move_exp_nansum(array, alpha=alpha)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "func_n",
    [
        # Functions that are variance-like
        (move_exp_nanvar, 1),
        (move_exp_nanstd, 1),
        (move_exp_nancov, 2),
        (move_exp_nancorr, 2),
    ],
)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_move_exp_nans_var(func_n, alpha):
    # This test is related to producing values with `move_exp_nancorr` on arrays with
    # `[0.9, np.nan]`, when we shouldn't. We could probably replace some of it with
    # property testing.

    func, n = func_n

    array = np.array([np.nan, np.nan])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True])
    assert_allclose(result, expected)

    array = np.array([5.0, np.nan])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True])
    assert_allclose(result, expected)

    array = np.array([1.0, np.nan, 2.0])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True, False])
    assert_allclose(result, expected)

    array = np.array([1.0, np.nan, 1.0])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    if func != move_exp_nancorr:
        expected = np.array([True, True, False])
    if func == move_exp_nancorr:
        # Correlation of values that are all the same is undefined
        expected = np.array([True, True, True])
    assert_allclose(result, expected)

    array = np.array([1.0, np.nan])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True])
    assert_allclose(result, expected)

    array = np.array([0.1, np.nan])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True])
    assert_allclose(result, expected)

    array = np.array([0.75, np.nan])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True])
    assert_allclose(result, expected)

    array = np.array([0.5, np.nan])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True])
    assert_allclose(result, expected)

    array = np.array([0.9, np.nan])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True])
    assert_allclose(result, expected)

    # # NEXT: why does this return a different result to the one above??
    array = np.array([0.95, np.nan, 1.0])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True, False])
    assert_allclose(result, expected)

    array = np.array([0.59288027, np.nan, 0.4758262, 0.70877039])
    result = np.isnan(func(*[array] * n, alpha=alpha))
    expected = np.array([True, True, False, False])
    assert_allclose(result, expected)


def test_move_exp_nanmean_numeric():
    array = np.array([10, 0, np.nan, 10])

    result = move_exp_nanmean(array, alpha=0.5)
    expected = np.array([10.0, 3.3333333, 3.3333333, 8.1818182])
    assert_allclose(result, expected)

    result = move_exp_nanmean(array, alpha=0.25)
    expected = np.array([10.0, 4.2857143, 4.2857143, 7.1653543])
    assert_allclose(result, expected)


def test_move_exp_nansum_numeric():
    array = np.array([10, 0, np.nan, 10])

    result = move_exp_nansum(array, alpha=0.5)
    expected = np.array([10.0, 5.0, 2.5, 11.25])
    assert_allclose(result, expected)

    result = move_exp_nansum(array, alpha=0.25)
    expected = np.array([10.0, 7.5, 5.625, 14.21875])
    assert_allclose(result, expected)


def test_move_exp_nancorr_numeric():
    array1 = np.array([10, 0, 5, 10])
    array2 = np.array([10, 0, 10, 5])

    result = move_exp_nancorr(array1, array2, alpha=0.5)
    expected = np.array([np.nan, 1.0, 0.8485281, 0.2274294])
    assert_allclose(result, expected)

    result = move_exp_nancorr(array1, array2, alpha=0.25)
    expected = np.array([np.nan, 1.0, 0.85, 0.4789468])
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "func",
    [
        move_exp_nancount,
        move_exp_nanmean,
        move_exp_nanstd,
        move_exp_nansum,
        move_exp_nanvar,
        move_exp_nancov,
        move_exp_nancorr,
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 0.1])
@pytest.mark.parametrize("shape", [(3, 500)], indirect=True)
def test_move_exp_alphas(array, alpha, func):
    c = COMPARISONS[func]

    # Supply alphas as a 1D array
    alphas = np.full(array.shape[-1], alpha)

    result = c["numbagg"](array, alpha=alphas)()
    expected = c["numbagg"](array, alpha=alpha)()
    assert_allclose(result, expected)

    result = c["numbagg"](array.T, alpha=alphas)(axis=0).T
    assert_allclose(result, expected)

    result = c["numbagg"](array.T, alpha=alpha)(axis=0).T
    assert_allclose(result, expected)
