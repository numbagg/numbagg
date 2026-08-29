from typing import TypeVar

import numpy as np
from numba import float32, float64, int64, njit

from .decorators import _ENABLE_CACHE, ndmove
from .utils import FloatArray

T = TypeVar("T", bound=FloatArray)


FLOAT32_EPSILON = np.finfo(np.float32).eps
FLOAT64_EPSILON = np.finfo(np.float64).eps
FLOAT32_SQRT_EPSILON = np.sqrt(FLOAT32_EPSILON)
FLOAT64_SQRT_EPSILON = np.sqrt(FLOAT64_EPSILON)
# If one magnitude is eps**(-1/4) larger than every other magnitude, its squared
# contribution is eps**(-1/2) larger. Adding and later removing it can therefore
# discard more than sqrt(eps) of the remaining second-moment state.
FLOAT32_EPSILON_QUARTER = np.sqrt(FLOAT32_SQRT_EPSILON)
FLOAT64_EPSILON_QUARTER = np.sqrt(FLOAT64_SQRT_EPSILON)

# A trailing cluster needs four observations (three adjacent differences) before
# it is treated as a new low-spread regime. One close pair is common in ordinary
# data and is not enough evidence to abandon the raw path. This is an event
# filter, not a coefficient in the roundoff bound.
MIN_CONTRACTION_EDGES = 3


@njit(cache=_ENABLE_CACHE, inline="never")
def _move_variance_stable_suffix(a, window, min_count, out, start, take_sqrt):
    """Overwrite ``out[start:]`` using causally shifted float64 moments.

    Keeping this recovery loop out of line is a measured performance choice: even
    a never-taken copy of its state machine inside the raw loop materially slows
    ordinary inputs. Difficult inputs pay for this second pass instead.
    """
    total = 0.0
    total_sq = 0.0
    origin = 0.0
    origin_index = -1
    count = 0
    updates = 0

    for i in range(start, len(a)):
        dominant_exited = False
        origin_expired = False
        if i > start:
            if i >= window:
                old = np.float64(a[i - window])
                if not np.isnan(old):
                    centered_old = old - origin
                    product_old = centered_old * centered_old
                    remaining_sq = total_sq - product_old
                    # If one contribution is more than 1 / sqrt(eps) times the
                    # rest, adding and removing it can discard more than
                    # sqrt(eps) of the smaller state. Recompute once it leaves.
                    dominant_exited = count >= 2 and abs(
                        product_old
                    ) * FLOAT64_SQRT_EPSILON > abs(remaining_sq)
                    total -= centered_old
                    total_sq = remaining_sq
                    count -= 1
                    updates += 1
                    origin_expired = origin_index == i - window

            value = np.float64(a[i])
            if not np.isnan(value):
                if count == 0:
                    origin = value
                    origin_index = i
                    total = 0.0
                    total_sq = 0.0
                    updates = 0
                centered = value - origin
                product = centered * centered
                total += centered
                total_sq += product
                count += 1
                updates += 1

        needs_rescan = i == start or origin_expired or dominant_exited
        if count >= min_count:
            correction = total * total / count
            numerator = total_sq - correction
            roundoff_fraction = updates * FLOAT64_EPSILON
            condition_scale = abs(total_sq) + abs(correction)
            needs_rescan = needs_rescan or (
                not np.isfinite(numerator)
                or numerator < 0.0
                or roundoff_fraction * condition_scale > abs(numerator)
            )

        if needs_rescan:
            window_start = max(0, i - window + 1)
            # Use the newest valid value: unlike the oldest value, it cannot
            # expire on the next update and leave a stale origin behind.
            for j in range(i, window_start - 1, -1):
                value = np.float64(a[j])
                if not np.isnan(value):
                    origin = value
                    origin_index = j
                    break
            total = 0.0
            total_sq = 0.0
            count = 0
            updates = 0
            for j in range(window_start, i + 1):
                value = np.float64(a[j])
                if not np.isnan(value):
                    centered = value - origin
                    product = centered * centered
                    total += centered
                    total_sq += product
                    count += 1
                    updates += 1

            if count >= min_count:
                correction = total * total / count
                numerator = total_sq - correction

        if count >= min_count:
            variance = max(numerator / (count - 1), 0.0)
            out[i] = np.sqrt(variance) if take_sqrt else variance
        else:
            out[i] = np.nan


@njit(cache=_ENABLE_CACHE, inline="always")
def _move_variance(a, window, min_count, out, take_sqrt):
    """Raw rolling variance core shared by ``move_var`` and ``move_std``."""
    total = 0.0
    total_sq = 0.0
    count = 0
    trigger_index = -1
    input_epsilon = FLOAT32_EPSILON if a.itemsize == 4 else FLOAT64_EPSILON
    sqrt_input_epsilon = (
        FLOAT32_SQRT_EPSILON if a.itemsize == 4 else FLOAT64_SQRT_EPSILON
    )
    min_count = max(min_count, 2)

    for i in range(len(a)):
        dominant_exited = False
        if i >= window:
            old = a[i - window]
            if not np.isnan(old):
                product_old = old * old
                remaining_sq = total_sq - product_old
                # See the stable suffix for the derivation of sqrt(eps). Squared
                # terms are non-negative, so the remaining square sum is a
                # non-cancelling scale for this comparison.
                dominant_exited = count >= 2 and abs(
                    np.float64(product_old)
                ) * sqrt_input_epsilon > abs(np.float64(remaining_sq))
                total -= old
                total_sq = remaining_sq
                count -= 1

        value = a[i]
        if not np.isnan(value):
            total += value
            total_sq += value * value
            count += 1

        # Recover even if this window is below min_count and emits NaN; otherwise
        # the one-time destructive removal would be forgotten.
        if dominant_exited:
            trigger_index = i
            break

        if count >= min_count:
            correction = total * total / count
            numerator = total_sq - correction
            # The input epsilon covers product rounding in its dtype; the update
            # count covers the float64 accumulators' addition/removal error.
            # Unlike a fixed coefficient, the bound grows with the history that
            # produced this state.
            roundoff_fraction = input_epsilon + (
                max(i + 1, 2 * (i + 1) - window) * FLOAT64_EPSILON
            )
            condition_scale = abs(total_sq) + abs(correction)
            if (
                (a.itemsize == 4 and count == 2)
                or not np.isfinite(numerator)
                or numerator < 0.0
                or roundoff_fraction * condition_scale > abs(numerator)
            ):
                trigger_index = i
                break

            variance = max(numerator / (count - 1), 0.0)
            out[i] = np.sqrt(variance) if take_sqrt else variance
        else:
            out[i] = np.nan

    if trigger_index >= 0:
        _move_variance_stable_suffix(
            a, window, min_count, out, trigger_index, take_sqrt
        )


@njit(cache=_ENABLE_CACHE, inline="never")
def _inspect_pairwise_window(a, b, end, window):
    """Return non-cancelling scale information for one causal window."""
    start = max(0, end - window + 1)
    minimum_a = np.inf
    maximum_a = -np.inf
    minimum_b = np.inf
    maximum_b = -np.inf
    largest_a = 0.0
    runner_up_a = 0.0
    largest_index_a = -1
    largest_b = 0.0
    runner_up_b = 0.0
    largest_index_b = -1
    total_sq_a = 0.0
    total_sq_b = 0.0
    for j in range(start, end + 1):
        value_a = np.float64(a[j])
        value_b = np.float64(b[j])
        if not (np.isnan(value_a) or np.isnan(value_b)):
            minimum_a = min(minimum_a, value_a)
            maximum_a = max(maximum_a, value_a)
            minimum_b = min(minimum_b, value_b)
            maximum_b = max(maximum_b, value_b)
            magnitude_a = abs(value_a)
            magnitude_b = abs(value_b)
            if magnitude_a > largest_a:
                runner_up_a = largest_a
                largest_a = magnitude_a
                largest_index_a = j
            elif magnitude_a > runner_up_a:
                runner_up_a = magnitude_a
            if magnitude_b > largest_b:
                runner_up_b = largest_b
                largest_b = magnitude_b
                largest_index_b = j
            elif magnitude_b > runner_up_b:
                runner_up_b = magnitude_b
            # Float32 products are promoted on the raw path and do not need the
            # startup cancellation bound. Numba removes this branch per signature.
            if a.itemsize == 8:
                total_sq_a += value_a * value_a
                total_sq_b += value_b * value_b
    return (
        minimum_a,
        maximum_a,
        minimum_b,
        maximum_b,
        largest_a,
        runner_up_a,
        largest_index_a,
        largest_b,
        runner_up_b,
        largest_index_b,
        total_sq_a,
        total_sq_b,
    )


@njit(cache=_ENABLE_CACHE, inline="never")
def _trailing_pair_boundaries(a, b, end, window, threshold_a, threshold_b):
    """Find an older regime followed by a sustained low-spread trailing cluster."""
    start = max(0, end - window + 1)
    previous_a = 0.0
    previous_b = 0.0
    have_previous = False
    active_a = True
    active_b = True
    small_edges_a = 0
    small_edges_b = 0
    boundary_a = -1
    boundary_b = -1
    for j in range(end, start - 1, -1):
        value_a = np.float64(a[j])
        value_b = np.float64(b[j])
        if not (np.isnan(value_a) or np.isnan(value_b)):
            if not have_previous:
                previous_a = value_a
                previous_b = value_b
                have_previous = True
            else:
                if active_a:
                    if abs(previous_a - value_a) <= threshold_a:
                        small_edges_a += 1
                    else:
                        if small_edges_a >= MIN_CONTRACTION_EDGES:
                            boundary_a = j
                        active_a = False
                if active_b:
                    if abs(previous_b - value_b) <= threshold_b:
                        small_edges_b += 1
                    else:
                        if small_edges_b >= MIN_CONTRACTION_EDGES:
                            boundary_b = j
                        active_b = False
                previous_a = value_a
                previous_b = value_b
                if not (active_a or active_b):
                    break
    return boundary_a, boundary_b


@njit(cache=_ENABLE_CACHE, inline="never")
def _move_covariance_stable_suffix(a, b, window, min_count, out, start, correlation):
    """Overwrite a suspicious covariance/correlation suffix with shifted moments."""
    total_a = 0.0
    total_b = 0.0
    total_ab = 0.0
    total_abs_ab = 0.0
    total_sq_a = 0.0
    total_sq_b = 0.0
    origin_a = 0.0
    origin_b = 0.0
    origin_index = -1
    count = 0
    updates = 0

    for i in range(start, len(a)):
        dominant_exited = False
        origin_expired = False
        if i > start:
            if i >= window:
                old_a = np.float64(a[i - window])
                old_b = np.float64(b[i - window])
                if not (np.isnan(old_a) or np.isnan(old_b)):
                    centered_old_a = old_a - origin_a
                    centered_old_b = old_b - origin_b
                    product_old = centered_old_a * centered_old_b
                    total_a -= centered_old_a
                    total_b -= centered_old_b
                    total_ab -= product_old
                    if correlation:
                        product_sq_a = centered_old_a * centered_old_a
                        product_sq_b = centered_old_b * centered_old_b
                        remaining_sq_a = total_sq_a - product_sq_a
                        remaining_sq_b = total_sq_b - product_sq_b
                        dominant_exited = dominant_exited or (
                            count >= 2
                            and (
                                abs(product_sq_a) * FLOAT64_SQRT_EPSILON
                                > abs(remaining_sq_a)
                                or abs(product_sq_b) * FLOAT64_SQRT_EPSILON
                                > abs(remaining_sq_b)
                            )
                        )
                        total_sq_a = remaining_sq_a
                        total_sq_b = remaining_sq_b
                    else:
                        remaining_abs_ab = total_abs_ab - abs(product_old)
                        dominant_exited = count >= 2 and abs(
                            product_old
                        ) * FLOAT64_SQRT_EPSILON > abs(remaining_abs_ab)
                        total_abs_ab = remaining_abs_ab
                    count -= 1
                    updates += 1
                    origin_expired = origin_index == i - window

            value_a = np.float64(a[i])
            value_b = np.float64(b[i])
            if not (np.isnan(value_a) or np.isnan(value_b)):
                if count == 0:
                    origin_a = value_a
                    origin_b = value_b
                    origin_index = i
                    total_a = 0.0
                    total_b = 0.0
                    total_ab = 0.0
                    total_abs_ab = 0.0
                    total_sq_a = 0.0
                    total_sq_b = 0.0
                    updates = 0
                centered_a = value_a - origin_a
                centered_b = value_b - origin_b
                product = centered_a * centered_b
                total_a += centered_a
                total_b += centered_b
                total_ab += product
                if correlation:
                    total_sq_a += centered_a * centered_a
                    total_sq_b += centered_b * centered_b
                else:
                    total_abs_ab += abs(product)
                count += 1
                updates += 1

        needs_rescan = i == start or origin_expired or dominant_exited
        numerator_a = 0.0
        numerator_b = 0.0
        if count >= min_count:
            correction_ab = total_a * total_b / count
            numerator_ab = total_ab - correction_ab
            roundoff_fraction = updates * FLOAT64_EPSILON
            needs_rescan = needs_rescan or not np.isfinite(numerator_ab)
            if correlation:
                correction_a = total_a * total_a / count
                correction_b = total_b * total_b / count
                numerator_a = total_sq_a - correction_a
                numerator_b = total_sq_b - correction_b
                covariance_scale = np.sqrt(
                    max(numerator_a, 0.0) * max(numerator_b, 0.0)
                )
                needs_rescan = needs_rescan or (
                    not np.isfinite(numerator_a)
                    or not np.isfinite(numerator_b)
                    or numerator_a < 0.0
                    or numerator_b < 0.0
                    or roundoff_fraction * (abs(total_sq_a) + abs(correction_a))
                    > abs(numerator_a)
                    or roundoff_fraction * (abs(total_sq_b) + abs(correction_b))
                    > abs(numerator_b)
                    or roundoff_fraction * (abs(total_ab) + abs(correction_ab))
                    > covariance_scale
                )
            else:
                needs_rescan = needs_rescan or (
                    roundoff_fraction * (abs(total_ab) + abs(correction_ab))
                    > abs(numerator_ab)
                )

        if needs_rescan:
            window_start = max(0, i - window + 1)
            for j in range(i, window_start - 1, -1):
                value_a = np.float64(a[j])
                value_b = np.float64(b[j])
                if not (np.isnan(value_a) or np.isnan(value_b)):
                    origin_a = value_a
                    origin_b = value_b
                    origin_index = j
                    break
            total_a = 0.0
            total_b = 0.0
            total_ab = 0.0
            total_abs_ab = 0.0
            total_sq_a = 0.0
            total_sq_b = 0.0
            count = 0
            updates = 0
            for j in range(window_start, i + 1):
                value_a = np.float64(a[j])
                value_b = np.float64(b[j])
                if not (np.isnan(value_a) or np.isnan(value_b)):
                    centered_a = value_a - origin_a
                    centered_b = value_b - origin_b
                    product = centered_a * centered_b
                    total_a += centered_a
                    total_b += centered_b
                    total_ab += product
                    if correlation:
                        total_sq_a += centered_a * centered_a
                        total_sq_b += centered_b * centered_b
                    else:
                        total_abs_ab += abs(product)
                    count += 1
                    updates += 1

            if count >= min_count:
                correction_ab = total_a * total_b / count
                numerator_ab = total_ab - correction_ab
                if correlation:
                    numerator_a = total_sq_a - total_a * total_a / count
                    numerator_b = total_sq_b - total_b * total_b / count

        if count >= min_count:
            if correlation:
                denominator_sq = max(numerator_a, 0.0) * max(numerator_b, 0.0)
                if denominator_sq > 0.0:
                    value = numerator_ab / np.sqrt(denominator_sq)
                    out[i] = min(max(value, -1.0), 1.0)
                else:
                    out[i] = np.nan
            else:
                out[i] = numerator_ab / (count - 1)
        else:
            out[i] = np.nan


@njit(cache=_ENABLE_CACHE, inline="always")
def _move_covariance_float32(a, b, window, min_count, out, correlation):
    """Promoted raw pairwise moments with event-local recovery checks."""
    total_a = 0.0
    total_b = 0.0
    total_ab = 0.0
    total_sq_a = 0.0
    total_sq_b = 0.0
    count = 0
    inspected = False
    trigger_index = -1
    danger_index_a = -1
    danger_index_b = -1
    lower_guard_a = 0.0
    upper_guard_a = 0.0
    lower_guard_b = 0.0
    upper_guard_b = 0.0
    min_count = max(min_count, 1 if correlation else 2)

    for i in range(len(a)):
        if i >= window:
            old_a = np.float64(a[i - window])
            old_b = np.float64(b[i - window])
            if not (np.isnan(old_a) or np.isnan(old_b)):
                total_a -= old_a
                total_b -= old_b
                total_ab -= old_a * old_b
                if correlation:
                    total_sq_a -= old_a * old_a
                    total_sq_b -= old_b * old_b
                count -= 1
            if danger_index_a == i - window or danger_index_b == i - window:
                trigger_index = i
                break

        value_a = np.float64(a[i])
        value_b = np.float64(b[i])
        if not (np.isnan(value_a) or np.isnan(value_b)):
            if inspected and (
                value_a < lower_guard_a
                or value_a > upper_guard_a
                or value_b < lower_guard_b
                or value_b > upper_guard_b
            ):
                trigger_index = i
                break
            total_a += value_a
            total_b += value_b
            total_ab += value_a * value_b
            if correlation:
                total_sq_a += value_a * value_a
                total_sq_b += value_b * value_b
            count += 1

        if count >= min_count:
            numerator_ab = total_ab - total_a * total_b / count
            numerator_a = 0.0
            numerator_b = 0.0
            if correlation:
                numerator_a = total_sq_a - total_a * total_a / count
                numerator_b = total_sq_b - total_b * total_b / count
            if not inspected:
                (
                    minimum_a,
                    maximum_a,
                    minimum_b,
                    maximum_b,
                    largest_a,
                    runner_up_a,
                    largest_index_a,
                    largest_b,
                    runner_up_b,
                    largest_index_b,
                    _,
                    _,
                ) = _inspect_pairwise_window(a, b, i, window)
                range_a = maximum_a - minimum_a
                range_b = maximum_b - minimum_b
                if largest_a * FLOAT32_EPSILON_QUARTER > runner_up_a:
                    danger_index_a = largest_index_a
                if largest_b * FLOAT32_EPSILON_QUARTER > runner_up_b:
                    danger_index_b = largest_index_b
                # sqrt(window * eps) is the accumulated resolution of adjacent
                # differences across this window, scaled by its observed range.
                tail_a, tail_b = _trailing_pair_boundaries(
                    a,
                    b,
                    i,
                    window,
                    range_a * np.sqrt(window * FLOAT32_EPSILON),
                    range_b * np.sqrt(window * FLOAT32_EPSILON),
                )
                if tail_a >= 0 and (danger_index_a < 0 or tail_a < danger_index_a):
                    danger_index_a = tail_a
                if tail_b >= 0 and (danger_index_b < 0 or tail_b < danger_index_b):
                    danger_index_b = tail_b
                # A value outside range / eps**(1/4) can dominate the certified
                # scale enough that its later removal loses sqrt(eps) of state.
                growth_a = range_a / FLOAT32_EPSILON_QUARTER
                growth_b = range_b / FLOAT32_EPSILON_QUARTER
                lower_guard_a = minimum_a - growth_a
                upper_guard_a = maximum_a + growth_a
                lower_guard_b = minimum_b - growth_b
                upper_guard_b = maximum_b + growth_b
                inspected = True
            if not np.isfinite(numerator_ab) or (
                correlation and (numerator_a < 0.0 or numerator_b < 0.0)
            ):
                trigger_index = i
                break

            if correlation:
                denominator_sq = numerator_a * numerator_b
                if denominator_sq > 0.0:
                    value = numerator_ab / np.sqrt(denominator_sq)
                    out[i] = min(max(value, -1.0), 1.0)
                else:
                    out[i] = np.nan
            else:
                out[i] = numerator_ab / (count - 1)
        else:
            out[i] = np.nan

    if trigger_index >= 0:
        _move_covariance_stable_suffix(
            a, b, window, min_count, out, trigger_index, correlation
        )


@njit(cache=_ENABLE_CACHE, inline="always")
def _move_covariance_float64(a, b, window, min_count, out, correlation):
    """Native raw pairwise moments with a causal startup scale certificate."""
    total_a = 0.0
    total_b = 0.0
    total_ab = 0.0
    total_sq_a = 0.0
    total_sq_b = 0.0
    count = 0
    inspected = False
    trigger_index = -1
    danger_index_a = -1
    danger_index_b = -1
    lower_guard_a = 0.0
    upper_guard_a = 0.0
    lower_guard_b = 0.0
    upper_guard_b = 0.0
    contraction_threshold_a = 0.0
    contraction_threshold_b = 0.0
    contraction_count_a = 0
    contraction_count_b = 0
    previous_a = 0.0
    previous_b = 0.0
    min_count = max(min_count, 1 if correlation else 2)

    for i in range(len(a)):
        if i >= window:
            old_a = a[i - window]
            old_b = b[i - window]
            if not (np.isnan(old_a) or np.isnan(old_b)):
                total_a -= old_a
                total_b -= old_b
                total_ab -= old_a * old_b
                if correlation:
                    total_sq_a -= old_a * old_a
                    total_sq_b -= old_b * old_b
                count -= 1
            if danger_index_a == i - window or danger_index_b == i - window:
                trigger_index = i
                break

        value_a = a[i]
        value_b = b[i]
        if not (np.isnan(value_a) or np.isnan(value_b)):
            if inspected:
                if (
                    value_a < lower_guard_a
                    or value_a > upper_guard_a
                    or value_b < lower_guard_b
                    or value_b > upper_guard_b
                ):
                    trigger_index = i
                    break
                contraction_count_a = (
                    contraction_count_a + 1
                    if abs(value_a - previous_a) <= contraction_threshold_a
                    else 0
                )
                contraction_count_b = (
                    contraction_count_b + 1
                    if abs(value_b - previous_b) <= contraction_threshold_b
                    else 0
                )
                if (
                    contraction_count_a >= min_count - 1
                    or contraction_count_b >= min_count - 1
                ):
                    trigger_index = i
                    break
                previous_a = value_a
                previous_b = value_b
            total_a += value_a
            total_b += value_b
            total_ab += value_a * value_b
            if correlation:
                total_sq_a += value_a * value_a
                total_sq_b += value_b * value_b
            count += 1

        if count >= min_count:
            correction_ab = total_a * total_b / count
            numerator_ab = total_ab - correction_ab
            numerator_a = 0.0
            numerator_b = 0.0
            if correlation:
                numerator_a = total_sq_a - total_a * total_a / count
                numerator_b = total_sq_b - total_b * total_b / count
            if not inspected:
                (
                    minimum_a,
                    maximum_a,
                    minimum_b,
                    maximum_b,
                    largest_a,
                    runner_up_a,
                    largest_index_a,
                    largest_b,
                    runner_up_b,
                    largest_index_b,
                    inspected_sq_a,
                    inspected_sq_b,
                ) = _inspect_pairwise_window(a, b, i, window)
                range_a = maximum_a - minimum_a
                range_b = maximum_b - minimum_b
                correction_a = total_a * total_a / count
                correction_b = total_b * total_b / count
                inspected_a = inspected_sq_a - correction_a
                inspected_b = inspected_sq_b - correction_b
                # For a sample in [min, max], the unnormalized second moment is
                # between range**2 / 2 and count * range**2 / 4 (Popoviciu).
                lower_a = 0.5 * range_a * range_a
                lower_b = 0.5 * range_b * range_b
                upper_a = count * range_a * range_a / 4.0
                upper_b = count * range_b * range_b / 4.0
                upper_ab = count * range_a * range_b / 4.0
                # One epsilon covers each product and count epsilons cover the
                # first accumulation. These are startup bounds, not fitted knobs.
                roundoff_a = (
                    (count + 1)
                    * FLOAT64_EPSILON
                    * (abs(inspected_sq_a) + abs(correction_a))
                )
                roundoff_b = (
                    (count + 1)
                    * FLOAT64_EPSILON
                    * (abs(inspected_sq_b) + abs(correction_b))
                )
                roundoff_ab = (
                    (count + 1) * FLOAT64_EPSILON * (abs(total_ab) + abs(correction_ab))
                )
                if (
                    not np.isfinite(numerator_ab)
                    or inspected_a < 0.0
                    or inspected_b < 0.0
                    or roundoff_a > upper_a
                    or roundoff_b > upper_b
                    or roundoff_ab > upper_ab
                    or inspected_a < lower_a - roundoff_a
                    or inspected_b < lower_b - roundoff_b
                    or inspected_a > upper_a + roundoff_a
                    or inspected_b > upper_b + roundoff_b
                    or abs(numerator_ab) > upper_ab + roundoff_ab
                ):
                    trigger_index = i
                    break
                if largest_a * FLOAT64_EPSILON_QUARTER > runner_up_a:
                    danger_index_a = largest_index_a
                if largest_b * FLOAT64_EPSILON_QUARTER > runner_up_b:
                    danger_index_b = largest_index_b
                tail_a, tail_b = _trailing_pair_boundaries(
                    a,
                    b,
                    i,
                    window,
                    range_a * np.sqrt(window * FLOAT64_EPSILON),
                    range_b * np.sqrt(window * FLOAT64_EPSILON),
                )
                if tail_a >= 0 and (danger_index_a < 0 or tail_a < danger_index_a):
                    danger_index_a = tail_a
                if tail_b >= 0 and (danger_index_b < 0 or tail_b < danger_index_b):
                    danger_index_b = tail_b
                growth_a = range_a / FLOAT64_EPSILON_QUARTER
                growth_b = range_b / FLOAT64_EPSILON_QUARTER
                lower_guard_a = minimum_a - growth_a
                upper_guard_a = maximum_a + growth_a
                lower_guard_b = minimum_b - growth_b
                upper_guard_b = maximum_b + growth_b
                contraction_threshold_a = range_a * FLOAT64_SQRT_EPSILON
                contraction_threshold_b = range_b * FLOAT64_SQRT_EPSILON
                previous_a = value_a
                previous_b = value_b
                inspected = True
            if not np.isfinite(numerator_ab) or (
                correlation and (numerator_a < 0.0 or numerator_b < 0.0)
            ):
                trigger_index = i
                break

            if correlation:
                denominator_sq = numerator_a * numerator_b
                if denominator_sq > 0.0:
                    value = numerator_ab / np.sqrt(denominator_sq)
                    out[i] = min(max(value, -1.0), 1.0)
                else:
                    out[i] = np.nan
            else:
                out[i] = numerator_ab / (count - 1)
        else:
            out[i] = np.nan

    if trigger_index >= 0:
        _move_covariance_stable_suffix(
            a, b, window, min_count, out, trigger_index, correlation
        )


@ndmove.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_mean(a: T, window: int, min_count: int, out: T) -> None:
    asum = 0.0
    count = 0
    min_count = max(min_count, 1)

    # We previously had an initial loop which filled NaNs before `min_count`, but it
    # didn't have a discernible effect on performance.

    for i in range(window):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = asum / count if count >= min_count else np.nan

    for i in range(window, len(a)):
        ai = a[i]
        aold = a[i - window]

        ai_valid: bool = not np.isnan(ai)
        aold_valid: bool = not np.isnan(aold)

        # We previously had a single operation where both variables are valid, but it
        # caused some numerical instability for float32 values. For example the
        # `test_numerical_issues_float32_move_mean_1` test fails. While it had a 10%
        # performance impact relative to the previous if / elif, the current mode with
        # just two `if` branches is about 10% faster than the previous mode; maybe it
        # can execute both branches in parallel?

        # if ai_valid and aold_valid:
        #     asum += ai - aold
        # elif ...

        if aold_valid:
            asum -= aold
            count -= 1
        if ai_valid:
            asum += ai
            count += 1

        out[i] = asum / count if count >= min_count else np.nan


@ndmove.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_sum(a: T, window: int, min_count: int, out: T) -> None:
    asum = 0.0
    count = 0

    # We don't generally split these up into two loops, but in `move_sum` & `move_mean`,
    # they're sufficiently different that it's worthwhile.

    for i in range(window):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = asum if count >= min_count else np.nan

    for i in range(window, len(a)):
        ai = a[i]
        aold = a[i - window]

        ai_valid: bool = not np.isnan(ai)
        aold_valid: bool = not np.isnan(aold)

        # Similar to the comment in `move_mean`, we previously had a single operation if
        # both were valid. That causes numerical instability for float32 values with a
        # window of 1.
        #
        # But possibly — particularly with a sum — the old and new values are likely to
        # be closer to each other than to the accumulator, so the numerical instability
        # is worse with this approach. When testing — for example with
        # `test_numerical_issues_float32_move_sum_100`, both approaches seem to fail
        # when increasing the multiplier at approximately the same rate.

        if ai_valid:
            asum += ai
            count += 1
        if aold_valid:
            asum -= aold
            count -= 1

        out[i] = asum if count >= min_count else np.nan


# TODO: pandas doesn't use a `min_count`, which maybe makes sense, but also makes it inconsistent?
# @ndmove.wrap(
#     [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
# )
# def move_count(a, window, min_count, out):

#     count = 0

#     for i in range(window):
#         if not np.isnan(a[i]):
#             count += 1
#         out[i] = count if count >= min_count else np.nan

#     for i in range(window, len(a)):
#         if not np.isnan(a[i]):
#             count += 1
#         if not np.isnan(a[i - window]):
#             count -= 1
#         out[i] = count if count >= min_count else np.nan


@ndmove.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_std(a: T, window: int, min_count: int, out: T) -> None:
    # The shared core is force-inlined, so this wrapper adds no runtime call while
    # keeping standard deviation and variance on exactly the same state machine.
    _move_variance(a, window, min_count, out, True)


@ndmove.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_var(a: T, window: int, min_count: int, out: T) -> None:
    _move_variance(a, window, min_count, out, False)


@ndmove.wrap(
    [
        (float32[:], float32[:], int64, int64, float32[:]),
        (float64[:], float64[:], int64, int64, float64[:]),
    ]
)
def move_cov(a: T, b: T, window: int, min_count: int, out: T) -> None:
    if a.itemsize == 4:
        _move_covariance_float32(a, b, window, min_count, out, False)
    else:
        _move_covariance_float64(a, b, window, min_count, out, False)


@ndmove.wrap(
    [
        (float32[:], float32[:], int64, int64, float32[:]),
        (float64[:], float64[:], int64, int64, float64[:]),
    ]
)
def move_corr(a: T, b: T, window: int, min_count: int, out: T) -> None:
    if a.itemsize == 4:
        _move_covariance_float32(a, b, window, min_count, out, True)
    else:
        _move_covariance_float64(a, b, window, min_count, out, True)


# Re-export matrix functions for backward compatibility
from .moving_matrix import move_corrmatrix, move_covmatrix

__all__ = [
    "move_mean",
    "move_sum",
    "move_std",
    "move_var",
    "move_cov",
    "move_corr",
    "move_corrmatrix",
    "move_covmatrix",
]
