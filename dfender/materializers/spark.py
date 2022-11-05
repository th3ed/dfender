from __future__ import annotations
from typing import Callable, Union
from functools import partial
import pandas as pd
import pyspark.sql.dataframe as ps
from pyspark.sql.types import DataType
from pyspark.sql import SparkSession
from interface_meta import override
from .core import materializer, MaterializerInterface

@materializer.register('spark-sql')
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
        print(meta)
        return df.mapInPandas(func, meta)

    @override
    def pandas_meta(self, df: ps.DataFrame) -> pd.DataFrame:
        dtypes = df.pandas_api().dtypes.to_dict()
        return pd.DataFrame({col : pd.Series(dtype=dtype) for col, dtype in dtypes.items()})

    @override
    def native_meta(self, df: pd.DataFrame) -> DataType:
        # Parse out the spark udf signature from the derived datatypes
        spark = SparkSession.builder.getOrCreate()
        df_spark = spark.createDataFrame(df)
        dtypes = {
            element.name : element.dataType.simpleString() for element in df_spark.schema
        }
        return ', '.join([f"{col} {dtype}" for col, dtype in dtypes.items()])
