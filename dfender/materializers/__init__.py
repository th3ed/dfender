from .core import MaterializerInterface, materializer
from .dask import DaskMaterializer
from .spark import SparkSQLMaterializer

# Lazy-load of materializers based on installed packages
