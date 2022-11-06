from __future__ import annotations
from typing import Callable, Union
from functools import partial
import pandas as pd
import pyspark.sql.dataframe as ps
from pyspark.sql.types import DataType
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, struct
from interface_meta import override
from .core import MaterializerInterface
import dask.dataframe as dd

class SparkSQLMaterializer(MaterializerInterface):
    @property
    def series(self) -> type[ps.DataFrame]:
        return ps.DataFrame

    @property
    def df(self) -> type[ps.DataFrame]:
        return ps.DataFrame

    def map_partitions(
        self,
        df: ps.DataFrame,
        meta: str,
        func: Callable
        ) -> ps.DataFrame:
        # Convert to a pandas udf
        func_ = pandas_udf(meta)(func)

        return func_(struct(*df.columns))

    @override
    def pandas_meta(self, df: ps.DataFrame) -> pd.DataFrame:
        dtypes = df.pandas_api().dtypes.to_dict()
        
        # Dask dataframes generate a non-empty metadata df we can use here
        tmp = pd.DataFrame({col : pd.Series(dtype=dtype) for col, dtype in dtypes.items()})
        return dd.from_pandas(tmp, npartitions=1)._meta_nonempty

    @override
    def native_meta(self, df: Union[pd.Series, pd.DataFrame]) -> DataType:
        # Parse out the spark udf signature from the derived datatypes
        spark = SparkSession.builder.getOrCreate()
        df_spark = spark.createDataFrame(df)

        if isinstance(df, pd.Series):
            return df_spark.schema[0].dataType.simpleString()
        elif isinstance(df, pd.DataFrame):
            dtypes = {
                element.name : element.dataType.simpleString() for element in df_spark.schema
            }
            return ', '.join([f"{col} {dtype}" for col, dtype in dtypes.items()])
        raise ValueError(f"df is neither a Series nor DataFrame, got type {type(df)}")