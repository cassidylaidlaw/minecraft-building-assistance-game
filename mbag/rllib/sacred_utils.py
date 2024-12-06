import copy
from collections.abc import Iterable, Mapping
from typing import Any

from sacred.config.custom_containers import DogmaticDict, DogmaticList
from sacred.observers import FileStorageObserver

# Fix for sacred issue that DogmaticDicts and DogmaticLists don't deepcopy correctly.
# This is a hacky fix, but it works for now.


def _dogmatic_dict_copy(self):
    return DogmaticDict(dict(self))


def _dogmatic_dict_deepcopy(self, memo):
    return DogmaticDict(copy.deepcopy(dict(self), memo=memo))


def _dogmatic_list_copy(self):
    return DogmaticList(list(self))


def _dogmatic_list_deepcopy(self, memo):
    return DogmaticList(copy.deepcopy(list(self), memo=memo))


DogmaticDict.__copy__ = _dogmatic_dict_copy  # type: ignore
DogmaticDict.__deepcopy__ = _dogmatic_dict_deepcopy  # type: ignore
DogmaticList.__copy__ = _dogmatic_list_copy  # type: ignore
DogmaticList.__deepcopy__ = _dogmatic_list_deepcopy  # type: ignore


def convert_dogmatics_to_standard(obj: Any) -> Any:
    """Recursively converts an object with Sacred Dogmatics to a standard Python object."""
    if isinstance(obj, DogmaticDict):
        return {k: convert_dogmatics_to_standard(v) for k, v in obj.items()}
    elif isinstance(obj, DogmaticList):
        return [convert_dogmatics_to_standard(elem) for elem in obj]
    elif isinstance(obj, Mapping):
        return {k: convert_dogmatics_to_standard(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        # Exclude strings as they are also iterable but should not be treated as
        # a list of characters here.
        return [convert_dogmatics_to_standard(elem) for elem in obj]
    else:
        return obj


class NoTypeAnnotationsFileStorageObserver(FileStorageObserver):
    def save_json(self, obj, filename):
        if isinstance(obj, dict) and "__annotations__" in obj:
            del obj["__annotations__"]
        return super().save_json(obj, filename)
