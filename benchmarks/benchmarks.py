import numpy as np
import pandas as pd

from numbagg import move_exp_nanmean
from numbagg.moving import move_exp_nansum, move_exp_nanvar, move_mean


class Suite:
    params = [
        [
            (move_exp_nanmean, lambda x: x.mean()),
            (move_exp_nansum, lambda x: x.sum()),
            (move_exp_nanvar, lambda x: x.var()),
        ],
        [1_000, 100_000, 10_000_000],
    ]
    param_names = ["func", "n"]
    # While we're still in development mode, make these fast at the cost of small sample
    # size.
    repeat = 1
    rounds = 1
    number = 1

    def setup(self, func, n):
        array = np.random.RandomState(0).rand(3, n)
        self.array = np.where(array > 0.1, array, np.nan)
        self.df_ewm = pd.DataFrame(self.array.T).ewm(alpha=0.5)
        # One run for JIT (asv states that it does this before runs, but this still
        # seems to make a difference)
        func[0](self.array, 0.5)
        func[1](self.df_ewm)

    def time_numbagg(self, func, n):
        func[0](self.array, 0.5)

    def time_numbagg_min_weight(self, func, n):
        func[0](self.array, 0.5, min_weight=0.2)

    def time_pandas(self, func, n):
        func[1](self.df_ewm)


class Moving:
    params = [
        [
            (move_mean, lambda x: x.mean()),
        ],
        [1_000, 100_000, 10_000_000],
    ]
    param_names = ["func", "n"]
    repeat = 1
    rounds = 1
    number = 1

    def setup(self, func, n):
        array = np.random.RandomState(0).rand(3, n)
        self.array = np.where(array > 0.1, array, np.nan)
        self.df_rolling = pd.DataFrame(self.array.T).rolling(window=20)
        # One run for JIT (asv states that it does this before runs, but this still
        # seems to make a difference)
        func[0](self.array, 20)
        func[1](self.df_rolling)

    def time_numbagg(self, func, n):
        func[0](self.array, 20)

    def time_pandas(self, func, n):
        func[1](self.df_rolling)
