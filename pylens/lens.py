from .functors import Const, Identity
from .functions import Func1


class Lens:
    def __init__(self, run_lens, **sub_lenses):
        # run_lens: forall F. Functor F => (a -> F b) -> (s -> F t)
        self.run_lens = run_lens
        self.extra = dict()
        self.__dict__.update({k: self >> v for k, v in sub_lenses.items()})

    def view(self):
        def f(x):
            def make_const(x):
                return Const(x)
            return self.run_lens(make_const)(x).value

        return Func1(f)

    def update(self, func):
        def f(x):
            def make_identity(x):
                return Identity(x).map(func)
            return self.run_lens(make_identity)(x).value

        return Func1(f)

    def set(self, v):
        return self.update(lambda _: v)

    def compose(self, other):
        return other >> self

    def __rshift__(self, other):
        run_lens1 = lambda lift: self.run_lens(other.run_lens(lift))
        return Lens(run_lens1)
