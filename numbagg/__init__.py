from .funcs import (allnan, anynan, count, nanargmax, nanargmin, nanmax,
                    nanmean, nanstd, nanvar, nanmin, nansum)
from .moving import move_nanmean
from .decorators import ndreduce, ndmoving

dtypes = ['float32', 'float64']
