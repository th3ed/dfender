from __future__ import annotations
from functools import partial
from typing import Callable, get_type_hints, Union, Any
from abc import abstractproperty, abstractmethod
from interface_meta import InterfaceMeta
import pandas as pd
from multipledispatch.dispatcher import MethodDispatcher

class MaterializerInterface(metaclass=InterfaceMeta):
    '''Base Interface for Materializers'''
    INTERFACE_EXPLICIT_OVERRIDES = False
    INTERFACE_RAISE_ON_VIOLATION = True
    INTERFACE_SKIPPED_NAMES = {}

    @abstractproperty
    def series(self):
        '''The series class of the implemented library'''
        raise NotImplementedError

    @abstractproperty
    def df(self):
        '''The dataframe class of the implemented library'''
        raise NotImplementedError

    @abstractmethod
    def map_partitions(self, df: Any, meta: Any, func: Callable) -> Any:
        pass

    def to_type(self, cls: type[Union[pd.Series, pd.DataFrame]]):
        if cls == pd.Series:
            return self.series
        elif cls == pd.DataFrame:
            return self.df
        raise ValueError(f"cls must be a pandas series or dataframe class, got {cls}")

    def pandas_meta(self, df: Any) -> Union[pd.Series, pd.DataFrame]:
        return df

    def native_meta(self, df: pd.DataFrame) -> Any:
        return df

    def map(self, func: Callable) -> Callable:
        def wrapper(_, df: Any, *args, **kwargs):
            # Generate a new partial function which only needs the input df
            func_ = partial(func, self, *args, **kwargs)

            # Get the pandas equivalent metadata to pass to the function
            meta_in = self.pandas_meta(df)

            # Pass the meta through the function and then generate
            # the appropriate native metadata
            meta_out = self.native_meta(func_(meta_in))

            # Map the function using the 
            return self.map_partitions(df, meta_out, func_)
        return wrapper

class Materializer():
    def __init__(self):
        self._registry = {}

    def __repr__(self):
        out = "Dfender Materializer Registry\n\nImplementations: \n"
        out += "\n".join([f"{k}: [{v.series}, {v.df}]" for k, v in self.registry.items()])
        return out

    def dispatcher(self, func: Callable) -> type[MethodDispatcher]:
        dispatcher = MethodDispatcher(func.__name__)

        # Get the function type hints to determine how we are going to dispatch
        if len(func.__annotations__) < 1:
            # TODO: Add a better check for type-hinted inputs
            raise ValueError(f"Function \"{func.__name__}\" does not have required type hints")
        sig = next(iter(get_type_hints(func).values()))

        # Base implementation
        dispatcher.register(sig)(func)

        # Distributed implementations
        for r in self.registry.values():
            dispatcher.register(r.to_type(sig))(r.map(func))

        return dispatcher

    def register(self, name: str) -> Callable:
        def _register(mat: type[MaterializerInterface]):
            self.registry[name] = mat()
            return mat
        return _register
    
    @property
    def registry(self):
        return self._registry

materializer = Materializer()