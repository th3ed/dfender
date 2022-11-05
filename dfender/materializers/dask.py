from __future__ import annotations
from typing import Callable, Any
from functools import partial
import pandas as pd
import dask.dataframe as dd
from .core import materializer, MaterializerInterface

@materializer.register('dask')
class DaskMaterializer(MaterializerInterface):
    @property
    def series(self) -> type[dd.Series]:
        return dd.Series

    @property
    def df(self) -> type[dd.DataFrame]:
        return dd.DataFrame

    def map(self,
    func: Callable) -> Callable:
        def wrapper(self, df: Any[dd.Series, dd.DataFrame], *args, **kwargs):
            func_ = partial(func, self)
            return df.map_partitions(func_, *args, **kwargs)
        return wrapper
