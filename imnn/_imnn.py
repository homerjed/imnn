from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array


def filter_value_and_jacfwd(f: eqx.Module, x: Array) -> Tuple[Array, Array]:
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, J = eqx.filter_vmap(
        partial(eqx.filter_jvp, f, (x,)), out_axes=(None, 1)
    )((basis,))
    return y, J 


def get_f_d_alpha(
    net: eqx.Module, 
    d_0: Array, 
    d_0_derivative: Array
) -> Tuple[Array, Array]:

    x, J_x_d = filter_value_and_jacfwd(net, d_0)

    # Chain rule d(summaries)/d(alpha) = d(summaries)/d(data) * d(data)/d(alpha)
    # (n_summaries, n_data) * (n_parameters, n_data) -> (n_summaries, n_parameters)
    mu_f_alpha = jnp.einsum("ij, kj -> ik", J_x_d, d_0_derivative) # Eq. 4.3

    return x, mu_f_alpha # = mu_f(x),alpha


def get_summaries_covariance(x: Array) -> Tuple[Array, Array]:
    n_data, data_dim = x.shape
    C_f = jnp.cov(x, rowvar=False)
    if x.shape[-1] == 1:
        C_f = jnp.array([[C_f]])
    C_f_inv = jnp.linalg.inv(C_f)
    H = (n_data - 2. - data_dim) / (n_data - 1.) # Hartlap factor
    return C_f, H * C_f_inv


@eqx.filter_jit
def get_F(
    fiducials: Array, 
    net: eqx.Module, 
    fiducials_and_derivatives: Tuple[Array, Array],
) -> Tuple[Array, Tuple[Array, Array, Array, Array]]:

    # Get the derivatives of the fiducial summaries w.r.t. summaries
    _get_mu_f_alpha = eqx.filter_vmap(partial(get_f_d_alpha, net))

    d_0, dd_0 = fiducials_and_derivatives
    x0, mu_f_alpha = _get_mu_f_alpha(d_0, dd_0) # same x, just less of them?
    mu_f_alpha_mean = mu_f_alpha.mean(axis=0)

    x = jax.vmap(net)(fiducials) # x0 are first 500 fiducial summaries

    # Stack all the summaries we have to calculate F (this function NEVER calculates via latins)
    C_f, C_f_inv = get_summaries_covariance(jnp.concatenate([x, x0]))

    # (n_summaries, n_parameters) * (n_summaries, n_summaries) * (n_summaries, n_parameters) -> (n_parameters, n_parameters)
    F = jnp.einsum(
        "pq, pr, rs -> qs", 
        mu_f_alpha_mean, 
        C_f_inv, # Already Hartlap'd
        mu_f_alpha_mean
    )
    return F, (x, mu_f_alpha_mean, C_f, C_f_inv)


@eqx.filter_jit
def get_x(
    alpha: Array, 
    fiducials: Array,
    d: Array, 
    net: eqx.Module,
    fiducials_and_derivatives: Tuple[Array, Array]
) -> Tuple[Array, Array]:

    F, (x0, mu_f_alpha_mean, C_f, C_f_inv) = get_F(
        fiducials, net, fiducials_and_derivatives
    )
    mu = x0.mean(axis=0)
    Finv = jnp.linalg.inv(F) # survey scaling

    # Summaries of data (not necessarily fiducials) with validation network  
    x_d = jax.vmap(net)(d)

    # Derive MLE of arbitrary d by score compression of Gaussian IMNN likelihood
    mle = lambda x: jnp.einsum(
        "pq, rq, rs, s -> p", 
        Finv,
        mu_f_alpha_mean, 
        C_f_inv, 
        x - mu
    )
    s_x_d = alpha + jax.vmap(mle)(x_d)

    # qMLEs and summaries
    return s_x_d, x_d # score compressed summaries of IMNN Gaussian likelihood and IMNN outputs