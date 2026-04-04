"""Lightweight numpy-compatible subset for offline testing."""
from __future__ import annotations
import math

uint8 = int
float32 = float

class ndarray:
    def __init__(self, data):
        if isinstance(data, ndarray):
            self.data = list(data.data)
        else:
            self.data = list(data)
    @property
    def shape(self):
        return (len(self.data),)
    def astype(self, _dtype):
        return ndarray([float(x) for x in self.data])
    def flatten(self):
        flat=[]
        for x in self.data:
            if isinstance(x,list):
                flat.extend(x)
            else:
                flat.append(x)
        return ndarray(flat)
    def tolist(self):
        return list(self.data)
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data)
    def __getitem__(self, item):
        if isinstance(item, slice):
            return ndarray(self.data[item])
        return self.data[item]
    def __mul__(self, other):
        return ndarray([x*other for x in self.data])
    def __truediv__(self, other):
        return ndarray([x/other for x in self.data])


def array(values, dtype=None):
    if isinstance(values, ndarray):
        return ndarray(values)
    if values and isinstance(values[0], list):
        return ndarray([float(v) for row in values for v in row])
    return ndarray([float(v) for v in values])


def asarray(values, dtype=None):
    return array(values, dtype=dtype)


def dot(v1, v2):
    a = asarray(v1).data
    b = asarray(v2).data
    return sum(x*y for x, y in zip(a, b))


def frombuffer(buf, dtype=None):
    return ndarray(list(buf))


class _Linalg:
    @staticmethod
    def norm(vec):
        data = asarray(vec).data
        return math.sqrt(sum(x*x for x in data))


linalg = _Linalg()
