from typing import *
import copy
from dataclasses import is_dataclass, fields
from ..lens import Lens


def field_lens(field_name, **sub_lenses):
    def run(lift):
        def func(obj):
            a = getattr(obj, field_name)

            def f(b):
                new_obj = copy.copy(obj)
                setattr(new_obj, field_name, b)
                return new_obj

            return lift(a).map(f)
        return func
    return Lens(run, **sub_lenses)


def derive_lenses(cls):
    assert is_dataclass(cls)

    def get_lenses(cls) -> Dict[str, Lens]:
        assert is_dataclass(cls)

        def build_lens(field) -> Lens:
            tp = field.type
            name = field.name
            if is_dataclass(tp):
                sub_lenses = get_lenses(tp)
                return field_lens(name, **sub_lenses)
            else:
                return field_lens(name)

        return {f'_{f.name}': build_lens(f) for f in fields(cls)}

    lenses = get_lenses(cls)

    for k, v in lenses.items():
        setattr(cls, k, v)

    return cls
