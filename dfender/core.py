from typing import Callable, Union, Tuple, List
from multipledispatch import dispatch
from .materializers import materializer

def distributable(sig: Union[type, List, Tuple]):
    def decorator(func: Callable):
        dispatcher = dispatch(sig)(func)

        # Register available distributed implementations
        materializer.template(dispatcher, sig, func)

        # Return the base implementation as well
        return dispatcher

    return decorator
