class Functor:
    def map(self, func):
        raise NotImplementedError(f'{self.__class__.__name__} is not implemented.')


class Identity(Functor):
    def __init__(self, value):
        super(Identity, self).__init__()
        self.value = value

    def map(self, func):
        return Identity(func(self.value))


class Const(Functor):
    def __init__(self, value):
        super(Const, self).__init__()
        self.value = value

    def map(self, _):
        return self
