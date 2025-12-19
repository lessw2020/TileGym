"""Shared CUDA Tile constant type aliases.

Many cuTile kernels repeat these aliases:

- ConstInt = ct.Constant[int]
- ConstBool = ct.Constant[bool]
- ConstFloat = ct.Constant[float]

Import them from here to keep kernels consistent and reduce boilerplate.
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]

__all__ = ["ConstInt", "ConstBool", "ConstFloat"]
