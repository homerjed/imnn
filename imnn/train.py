from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
from jaxtyping import Key, Array, PyTree
import optax

from ._imnn import get_F

OptState = PyTree
GradientTransformation = optax.GradientTransformation


def regularise_C_f(C_f, C_f_inv):
    """
        The Frobenius Norm of a matrix is defined as the square 
        root of the sum of the squares of the elements of the matrix. 
        ||A||_F = tr(sqrt(AA^T))
    """
    I = jnp.eye(C_f.shape[-1])
    # return jnp.linalg.norm(C_f - I) + jnp.linalg.norm(C_f_inv - I)
    # a, b = jnp.linalg.slogdet(C_f)
    # return a * b
    return 0.5 * jnp.add(
        jnp.linalg.norm(jnp.subtract(C_f, I), ord="fro"),
        jnp.linalg.norm(jnp.subtract(C_f_inv, I), ord="fro")
    )


def logdet(F):
    # Numpy docs => det = logdet[0] * exp(logdet[1])
    logdet = jnp.linalg.slogdet(F)
    return logdet[0] * logdet[1]
     

def get_alpha(f, eps):
    # eps = how close to 1 the det. of C_f, C_f_inv should be
    # f = scalar for covariance regularisation strength
    return -jnp.log(eps * (f - 1.) + eps ** 2. / (1. + eps)) / eps


def get_r(covariance_reg, f=10., eps=0.1):
    alpha = get_alpha(f, eps)
    return f * covariance_reg / (covariance_reg + jnp.exp(-alpha * covariance_reg))


# def get_r(covariance_reg, eps=0.1, f=10.):
#     return f * covariance_reg / (covariance_reg + jnp.exp(-eps * covariance_reg))


@eqx.filter_jit
def loss_fn(
    net: eqx.Module, 
    d0: Array, 
    fiducials_and_derivatives: Tuple[Array, Array], 
    f: float = 10., 
    eps: float = 0.1,
) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
    """
        Given network params and physics parameters, simulate
        a dataset {d}, compress to {x} and calculate the Fisher
        and covariance matrices, the sum of determinants of which 
        are the loss function.
    """
    # Survey scale Finv later
    F, (_, _, C_f, C_f_inv) = get_F(
        d0, net, fiducials_and_derivatives
    )

    # Maximise Fisher information, summaries covariance -> identity
    L_F = logdet(F) 
    L_C = regularise_C_f(C_f, C_f_inv)
    
    # Secondary loss term to scale 'x', Fisher info alone being invariant to linear rescaling of x
    L = -L_F + get_r(L_C, f=f, eps=eps) * L_C 

    # Return det(C) for plotting here?
    return L, (L_F, L_C, C_f, C_f_inv)


@eqx.filter_jit
def update(
    net: eqx.Module, 
    grads: eqx.Module, 
    opt_state: OptState, 
    optimizer: GradientTransformation
) -> Tuple[eqx.Module, OptState]:
    updates, opt_state = optimizer.update(grads, opt_state, net) 
    return eqx.apply_updates(net, updates), opt_state 