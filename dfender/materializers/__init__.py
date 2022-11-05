from .core import MaterializerInterface, materializer
from .dask import DaskMaterializer
from .spark import SparkMaterializer

# Lazy-load of materializers based on installed packages
