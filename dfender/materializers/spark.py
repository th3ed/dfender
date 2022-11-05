from __future__ import annotations
from typing import Callable, Any
from functools import partial
import pandas as pd
import pyspark.sql.dataframe as ps
from .core import materializer, MaterializerInterface

@materializer.register('spark')
class SparkMaterializer(MaterializerInterface):
    @property
    def series(self) -> type[ps.DataFrame]:
        return ps.DataFrame

    @property
    def df(self) -> type[ps.DataFrame]:
        return ps.DataFrame

    def map(self,
    func: Callable) -> Callable:
        def wrapper(self, df: ps.DataFrame, *args, **kwargs):
            func_ = partial(func, self, *args, **kwargs)
            return df.mapInPandas(func_, )
        return wrapper

    def _pandas_meta(self, df: ps.DataFrame) -> pd.DataFrame:
        '''
        Converts a pysprk df view to an empty pandas dataframe
        to be used for inferring the schema of outputs
        '''
        dtypes = df.pandas_api().dtypes.to_dict()
        return pd.DataFrame({col : pd.Series(dtype=dtype) for col, dtype in dtypes.items()})