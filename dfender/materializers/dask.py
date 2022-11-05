from __future__ import annotations
from typing import Callable, Any, Union
from functools import partial
import pandas as pd
import dask.dataframe as dd
from interface_meta import override
from .core import materializer, MaterializerInterface

@materializer.register('dask')
class DaskMaterializer(MaterializerInterface):
    @property
    def series(self) -> type[dd.Series]:
        return dd.Series

    @property
    def df(self) -> type[dd.DataFrame]:
        return dd.DataFrame

    def map_partitions(
        self,
        df: Union[dd.Series, dd.DataFrame],
        meta: Union[pd.Series, pd.DataFrame],
        func: Callable
        ) -> Union[dd.Series, dd.DataFrame]:
        return df.map_partitions(func, meta=meta)

    @override
    def pandas_meta(self, df: Union[dd.Series, dd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        return df._meta

    @override
    def native_meta(self, df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        return df.head(0)
