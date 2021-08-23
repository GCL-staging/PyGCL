import copy
from ..lens import Lens


def _idx(idx):
    def run_lens(lift):
        # lift: a -> f b
        # func: (x_1, x_2, ..., a, ..., x_n) -> f (x_1, x_2, ..., b, ..., x_n)
        def func(tup):
            a = tup[idx]

            def f(b):
                return tup[:idx] + (b,) + tup[idx+1:]

            return lift(a).map(f)

        return func
    return Lens(run_lens)


_1 = _idx(0)
_2 = _idx(1)
