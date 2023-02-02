
import typing as t
from os import PathLike
import numpy as np

if t.TYPE_CHECKING:
    import numpy.typing as npt

T = t.TypeVar("T")
FuncType = t.Callable[..., t.Any]
F = t.TypeVar("F", bound=FuncType)
