# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
import pandas as pd

from numbagg import move_exp_nanmean
from numbagg.moving import move_exp_nansum, move_exp_nanvar


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

    def setup(self, func, n):
        array = np.random.RandomState(0).rand(20, n)
        self.array = np.where(array > 0.1, array, np.nan)
        # self.array = np.random.rand(20, n)
        self.df_ewm = pd.DataFrame(self.array.T).ewm(alpha=0.5)
        # One run for JIT
        func[0](self.array, 0.5)
        func[1](self.df_ewm)

    def time_numbagg(self, func, n):
        func[0](self.array, 0.5)

    def time_pandas(self, func, n):
        func[1](self.df_ewm)
