from typing import Tuple, List, Optional, Callable, Union
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx

Array = jax.Array
Key = jr.PRNGKey


class Linear(eqx.Module):
    weight: Array
    bias: Array

    def __init__(self, in_size, out_size, *, key):
        key_w, _ = jr.split(key)
        lim = jnp.sqrt(1. / (in_size + 1.))
        self.weight = jr.normal(
            key_w, (out_size, in_size)
        ) * lim
        # self.weight = jr.truncated_normal(
        #     key_w, shape=(out_size, in_size), lower=-2., upper=2.
        # ) * lim
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: Array) -> Array:
        return self.weight @ x + self.bias


class ArcSinhScaling(eqx.Module):
    a: Array
    b: Array
    c: Array
    d: Array

    def __init__(self, shape: Tuple[int] = (1,)):
        """ Should this be data-dimensional? """
        self.a = jnp.ones(shape) 
        self.b = jnp.ones(shape) 
        self.c = jnp.zeros(shape) 
        self.d = jnp.zeros(shape)
    
    def __call__(self, x: Array) -> Array:
        return self.a * jnp.arcsinh(x * self.b + self.c) + self.d



class IMNNMLP(eqx.Module):
    layers: List
    scale_fn: Optional[Callable]

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        width_size: int, 
        depth: int, 
        activation: Callable, 
        scale_fn: Union[Callable, eqx.Module] = None, 
        layernorm: Optional[bool] = False,
        *, 
        key: Key
    ):
        layers = []
        dimensions = [in_size] + [width_size] * depth + [out_size]
        for _in, _out in zip(dimensions[:-1], dimensions[1:]):
            key, _key = jr.split(key)
            if layernorm:
                layers.append(eqx.nn.LayerNorm((_in,)))
            # layers.append(eqx.nn.Linear(_in, _out, key=_key))
            layers.append(Linear(_in, _out, key=_key))
            # layers.append(eqx.nn.LayerNorm((_out,)))
            layers.append(activation)
        self.layers = tuple(layers[:-1])
        self.scale_fn = scale_fn

    def __call__(self, x: Array) -> Array:
        if self.scale_fn is not None:
            x = self.scale_fn(x)
        for l in self.layers:
            x = l(x)
        return x


class IMNNCNN(eqx.Module):
    layers: List[eqx.Module]
    out: eqx.nn.Linear

    def __init__(
        self, 
        data_dim,
        out_size,
        width_size, 
        kernel_size=5,
        padding=2,
        depth=3, 
        activation=jax.nn.tanh,
        *, 
        key
    ):
        """
            Convolutional network in 1D
            - padding and kernel size must be: (1, 3), (2, 5), ...
            - in_channels fixed to 1 for PDFs
        """
        dimensions = [1] + [width_size] * depth 
        layers = []
        for _in, _out in zip(dimensions[:-1], dimensions[1:]):
            key, _key = jr.split(key)
            layers.append(
                eqx.nn.Conv1d(
                    _in, 
                    _out, 
                    kernel_size=kernel_size, 
                    padding=(padding,), 
                    stride=1, 
                    key=_key
                )
            )
            layers.append(
                # eqx.nn.MaxPool1d(kernel_size=1, stride=2))
                eqx.nn.AvgPool1d(kernel_size=1, stride=2)
            )
            layers.append(activation)
        self.layers = tuple(layers)
        # Flatten before this summarisation
        self.out = eqx.nn.Linear(
            width_size * (1 + int(data_dim / (2 ** (depth)))),
            out_size, 
            key=key
        )

    def __call__(self, x):
        x = x[jnp.newaxis, :]
        for l in self.layers:
            x = l(x)
        return self.out(x.flatten())