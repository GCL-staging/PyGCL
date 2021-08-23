from __future__ import annotations


class Func1:
    def __init__(self, func):
        self.func = func
        pass

    def __call__(self, x):
        return self.func(x)

    def compose(self, g: Func1):
        return Func1(lambda x: self.func(g(x)))

    def __matmul__(self, g: Func1):
        return self.compose(g)
