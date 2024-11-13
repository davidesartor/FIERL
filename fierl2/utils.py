from typing import Self
from jaxtyping import Float, Array, Key
import equinox as eqx


State = Float[Array, "x"]
Control = Float[Array, "u"]
Output = Float[Array, "y"]
Fault = Float[Array, "z"]


class RESET(eqx.Module):
    pass


class Module(eqx.Module):
    def replace(self, **kwargs) -> Self:
        where = lambda s: tuple(getattr(s, n) for n in kwargs.keys())
        return eqx.tree_at(where, self, tuple(kwargs.values()))

    def __post_init__(self):
        fields = self.__dataclass_fields__.values()
        need_reset = [field.name for field in fields if field.default_factory == RESET]
        if need_reset:
            other = self.reset(rng=None)
            for field in need_reset:
                setattr(self, field, getattr(other, field))
                if isinstance(getattr(self, field), RESET):
                    raise ValueError(f"reset failed for field '{field}'")

    def reset(self, *, rng: Key | None) -> Self:
        raise NotImplementedError
