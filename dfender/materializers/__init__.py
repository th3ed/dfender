from .core import materializer

# Lazy-load of materializers based on installed packages
@materializer.register('dask')
def _register_dask():
    from .dask import DaskMaterializer
    return DaskMaterializer()

@materializer.register('pyspark.sql')
def _register_pyspark_sql():
    from .pyspark_sql import SparkSQLMaterializer
    return SparkSQLMaterializer()
