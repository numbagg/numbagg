import numpy as np

DTYPES = [np.float32, np.float64]


def arrays(func_name, dtypes=DTYPES):
    return array_iter(array_generator, func_name, dtypes)


def array_iter(arrays_func, *args):
    for a in arrays_func(*args):
        if a.ndim < 2:
            yield a
        #  this is good for an extra check but in everyday development it
        #  is a pain because it doubles the unit test run time
        #  elif a.ndim == 3:
        #      for axes in permutations(range(a.ndim)):
        #          yield np.transpose(a, axes)
        else:
            yield a
            yield a.T


def array_generator(func_name, dtypes):
    "Iterator that yields arrays to use for unit testing."

    # define nan and inf
    if func_name in ("partition", "argpartition"):
        nan = 0
    else:
        nan = np.nan
    if func_name in ("move_sum", "move_mean", "move_std", "move_var"):
        # these functions can't handle inf
        inf = 8
    else:
        inf = np.inf

    # nan and inf
    yield np.array([inf, nan])
    yield np.array([inf, -inf])
    yield np.array([nan, 2, 3])
    yield np.array([-inf, 2, 3])
    if func_name != "nanargmin":
        yield np.array([nan, inf])

    # byte swapped
    yield np.array([1, 2, 3], dtype=">f4")
    yield np.array([1, 2, 3], dtype="<f4")

    # make sure slow is callable
    yield np.array([1, 2, 3], dtype=np.float16)

    # regression tests
    yield np.array([1, 2, 3]) + 1e9  # check that move_std is robust
    yield np.array([0, 0, 0])  # nanargmax/nanargmin
    yield np.array([1, nan, nan, 2])  # nanmedian
    yield np.array([2 ** 31], dtype=np.int64)  # overflows on windows
    yield np.array([[1.0, 2], [3, 4]])[..., np.newaxis]  # issue #183

    # ties
    yield np.array([0, 0, 0])
    yield np.array([0, 0, 0], dtype=np.float64)
    yield np.array([1, 1, 1], dtype=np.float64)

    # 0d input
    if not func_name.startswith("move"):
        yield np.array(-9)
        yield np.array(0)
        yield np.array(9)
        yield np.array(-9.0)
        yield np.array(0.0)
        yield np.array(9.0)
        yield np.array(-inf)
        yield np.array(inf)
        yield np.array(nan)

    # automate a bunch of arrays to test
    ss = {}
    ss[0] = {"size": 0, "shapes": [(0,), (0, 0), (2, 0), (2, 0, 1)]}
    ss[1] = {"size": 8, "shapes": [(8,)]}
    ss[2] = {"size": 12, "shapes": [(2, 6), (3, 4)]}
    ss[3] = {"size": 16, "shapes": [(2, 2, 4)]}
    ss[4] = {"size": 24, "shapes": [(1, 2, 3, 4)]}
    for seed in (1, 2):
        rs = np.random.RandomState(seed)
        for ndim in ss:
            size = ss[ndim]["size"]
            shapes = ss[ndim]["shapes"]
            for dtype in dtypes:
                a = np.arange(size, dtype=dtype)
                if issubclass(a.dtype.type, np.inexact):
                    if func_name not in ("nanargmin", "nanargmax"):
                        # numpy can't handle eg np.nanargmin([np.nan, np.inf])
                        idx = rs.rand(*a.shape) < 0.2
                        a[idx] = inf
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = nan
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] *= -1
                rs.shuffle(a)
                for shape in shapes:
                    yield a.reshape(shape)

    # non-contiguous arrays
    yield np.array([[1, 2], [3, 4]])[:, [1]]  # gh 161
    for dtype in dtypes:
        # 1d
        a = np.arange(12).astype(dtype)
        for start in range(3):
            for step in range(1, 3):
                yield a[start::step]  # don't use astype here; copy created
    for dtype in dtypes:
        # 2d
        a = np.arange(12).reshape(4, 3).astype(dtype)
        yield a[::2]
        yield a[:, ::2]
        yield a[::2][:, ::2]
    for dtype in dtypes:
        # 3d
        a = np.arange(24).reshape(2, 3, 4).astype(dtype)
        for start in range(2):
            for step in range(1, 2):
                yield a[start::step]
                yield a[:, start::step]
                yield a[:, :, start::step]
                yield a[start::step][::2]
                yield a[start::step][::2][:, ::2]


def array_order(a):
    f = a.flags
    string = []
    if f.c_contiguous:
        string.append("C")
    if f.f_contiguous:
        string.append("F")
    if len(string) == 0:
        string.append("N")
    return ",".join(string)
