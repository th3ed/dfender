from typing import Callable
from .materializers import materializer

def distribute(func: Callable):
    return materializer.dispatcher(func)
