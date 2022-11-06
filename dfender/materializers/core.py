from __future__ import annotations
from functools import partial
from typing import Callable, Union, Any, Tuple, List
from abc import abstractproperty, abstractmethod
from interface_meta import InterfaceMeta
import pandas as pd
from multipledispatch import dispatch
from multipledispatch.dispatcher import Dispatcher, MethodDispatcher
import importlib

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

    def template(self, dispatcher: Dispatcher, sig: Tuple[type], func: Callable):
        # Ensure signature is a tuple, then map to the new types
        sig = (sig,) if not isinstance(sig, tuple) else sig
        sig = tuple(self.to_type(s) for s in sig)
        dispatcher.add(sig, self.map(func))

    def map(self, func: Callable) -> Callable:
        def wrapper(_, df: Any, *args, **kwargs):
            # Generate a new partial function which only needs the input df
            def func_(df):
                return func(self, df, *args, **kwargs)
            # func_ = partial(func, self, *args, **kwargs)

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
        self.implementations = {}

    def __repr__(self):
        out = "Materializer Registry\n\nImplementations: \n"
        out += "\n".join([f"{k}: [{v.series}, {v.df}]" for k, v in self.implementations.items()])
        return out

    def register(self, module):
        def decorator(func: Callable):
            # Check if the module is importable, and if so register it
            found = importlib.util.find_spec(module) is not None
            if found:
                self.implementations[module] = func()
            return func
        return decorator

    def template(self, dispatcher: Dispatcher, sig: Union[type, Tuple[type], List[type]], func: Callable):
        for imp in self.implementations.values():
            imp.template(dispatcher, sig, func)

materializer = Materializer()