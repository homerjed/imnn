import os
from typing import Tuple, List, Optional, Callable, Union
import json
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import optax
import numpy as np 

Array = jnp.ndarray
Key = jr.PRNGKey
Net = eqx.Module
OptState = optax.OptState
GradientTransformation = optax.GradientTransformation


def trunc_init(weight: Array, key: Key) -> Array:
    _, in_, *_ = weight.shape
    stddev = jnp.sqrt(1. / in_)
    return stddev * jr.truncated_normal(
        key, lower=-2., upper=2., shape=weight.shape
    )


def init_weights(model: Net, init_fn: Callable, key: Key) -> Net:
    is_linear = lambda x: isinstance(x, (eqx.nn.Conv1d, eqx.nn.Linear))
    get_weights = lambda m: [
        x.weight 
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jr.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def filter_value_and_jacfwd(f: Net, x: Array) -> Tuple[Array, Array]:
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, J = eqx.filter_vmap(
        partial(eqx.filter_jvp, f, (x,)), out_axes=(None, 1)
    )((basis,))
    return y, J 


def get_f_d_alpha(
    net: Net, 
    d_0: Array, 
    d_0_derivative: Array
) -> Tuple[Array, Array]:
    """ 
        For one of the 500 fiducials `d` with derivatives,
        calculate the summary and derivative `d(x)/d(d)`
        then multiply it with the model derivative `d(d)/d(pi)`.
        - this will be vmapped over the matched fiducial and its derivative pairs
    """
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
    net: Net, 
    fiducials_and_derivatives: Tuple[Array, Array],
    F_planck: Optional[Array] = None
) -> Tuple[Array, Tuple[Array, Array, Array, Array]]:
    """ 
        - take fiducials and derivatives pairs, calculate mu(x),pi
        - take a set of data generated at pi_0, summarise it. 
        - only fiducial data realisations are ever put into this
        - don't need to get x by vmapping just do it on case by case when getting summaries
    """
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
    if F_planck is not None:
        F = F + F_planck 
    return F, (x, mu_f_alpha_mean, C_f, C_f_inv)


@eqx.filter_jit
def get_x(
    alpha: Array, 
    fiducials: Array,
    d: Array, 
    net: Net,
    fiducials_and_derivatives: Tuple[Array, Array],
    F_planck: Optional[Array] = None
) -> Tuple[Array, Array]:
    """
        Get summaries of data 'd' using Fisher matrix of fiducial summaries.
        - alpha^ = alpha0 + Finv_ab * d(mu_i)/d(alpha_b) * Cinv_ij(x - mu)_j

        USE VALIDATION NETWORK!!!
    """
    F, (x0, mu_f_alpha_mean, C_f, C_f_inv) = get_F(
        fiducials, net, fiducials_and_derivatives, F_planck=F_planck
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


def get_final_products(
    key: Key,
    fiducials: Array, # fiducials (valid)
    d: Array, # latins
    net: Net,
    fiducials_and_derivatives: Tuple[Array, Array],
    F_planck: Optional[Array] = None,
    results_dir: Optional[str] = None
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:

    # Fisher matrix, fiducial summaries, their derivatives and covariances
    F, (x0, mu_f_alpha_mean, C_f, C_f_inv) = get_F(
        fiducials, net, fiducials_and_derivatives, F_planck
    )
    Finv = jnp.linalg.inv(F) 

    # Latin summaries and fiducial summaries
    x_d = jax.vmap(net)(d)
    # All fiducial summaries
    x0 = jnp.concatenate([jax.vmap(net)(fiducials), x0])
    # Observed data summary
    noise = jr.multivariate_normal(
        key, 
        mean=jnp.zeros(fiducials.shape[-1]), 
        cov=jnp.cov(fiducials, rowvar=False)
    )
    d_ = fiducials.mean(axis=0) + noise
    xhat = net(d_) # x(d^)

    returns = (
        xhat, x0, x_d, mu_f_alpha_mean, C_f_inv, F, Finv
    )

    if results_dir is not None:
        names = (
            "data_x", "fiducial_x", "latin_x", "derivatives_x", "Cinv_x", "F", "Finv"
        )
        for item, name in zip(returns, names):
            jnp.save(os.path.join(results_dir, "finals/", name), item)
            print(f"Saved item {name} {item.shape}")
    return returns


'''
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
    net: Net, 
    d0: Array, 
    fiducials_and_derivatives: Tuple[Array, Array], 
    F_planck: Optional[Array] = None,
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
        d0, net, fiducials_and_derivatives, F_planck
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
    net: Net, 
    grads: Net, 
    opt_state: OptState, 
    optimizer: GradientTransformation
) -> Tuple[Net, OptState]:
    updates, opt_state = optimizer.update(grads, opt_state, net) 
    return eqx.apply_updates(net, updates), opt_state 
'''


# def save_model(filename, hyperparams, model):
#     """
#         Make attribute of NDEs/IMNNs
#         that returns all attributes as a dict...
#     """
#     with open(filename, "wb") as f:
#         hyperparam_str = json.dumps(hyperparams)
#         f.write((hyperparam_str + "\n").encode())
#         eqx.tree_serialise_leaves(f, model)


# def load_model(model, filename):
#     with open(filename, "rb") as f:
#         hyperparameters = json.loads(f.readline().decode())
#         model = model(**hyperparameters)
#         return eqx.tree_deserialise_leaves(f, model)


# def make_dirs(imnn_dir, results_dir, cora=None, linear=None, moments=None):
#     dirs = [
#         "figs/",
#         "figs/F/",
#         "figs/fiducials/", 
#         "figs/fiducials/targets/", 
#         "figs/fiducials/x/", 
#         "figs/latins/", 
#         "figs/latins/x/", 
#         "figs/latins/targets/", 
#         "params/",
#         "outputs/",
#         "outputs/fisher/",
#         "outputs/latins/",
#         "outputs/datas/",
#         "outputs/finals/"
#     ]

#     if not os.path.exists(imnn_dir):
#         os.mkdir(imnn_dir)

#     if cora is not None and linear is not None:
#         results_dir = os.path.join(
#             imnn_dir, 
#             "bulk/" if cora else "bulk_tails/",
#             "linear/" if linear else "",
#             results_dir
#         )
#     # Either "moments/" or "cumulants/"
#     if moments:
#         results_dir = os.path.join(
#             imnn_dir, moments, "linear/" if linear else "", results_dir
#         )
#     else:
#         results_dir = os.path.join(imnn_dir, results_dir)

#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir, exist_ok=True)

#     for dir_ in dirs:
#         if not os.path.exists(os.path.join(results_dir, dir_)):
#             os.mkdir(os.path.join(results_dir, dir_))

#     figs_dir = os.path.join(results_dir, "figs/")
#     output_dir = os.path.join(results_dir, "outputs/")

#     return results_dir, figs_dir, output_dir

'''
def make_dirs(imnn_dir, results_dir, cora=False, linear=False):
    dirs = [
        "figs/",
        "figs/F/",
        "figs/fiducials/", 
        "figs/fiducials/targets/", 
        "figs/fiducials/x/", 
        "figs/latins/", 
        "figs/latins/x/", 
        "figs/latins/targets/", 
        "params/",
        "outputs/",
        "outputs/fisher/",
        "outputs/latins/",
        "outputs/datas/",
        "outputs/finals/"
    ]
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if cora:
        results_dir = os.path.join(results_dir, "bulk/")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
    if linear:
        results_dir = os.path.join(results_dir, "linear/")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

    for dir_ in dirs:
        if not os.path.exists(os.path.join(results_dir, dir_)):
            os.mkdir(os.path.join(os.getcwd(), results_dir, dir_))

    figs_dir = os.path.join(results_dir, "figs/")
    output_dir = os.path.join(results_dir, "outputs/")

    return figs_dir, output_dir
'''

def gaussian_approx(x_d):
    lower = jnp.array([0.10, 0.03, 0.50, 0.80, 0.60])
    upper = jnp.array([0.50, 0.07, 0.90, 1.20, 1.00])
    gridsize = 100
    n_targets = x_d.shape[0]
    Finv = jnp.linalg.inv(F)
    Finv = Finv.reshape(1, *Finv.shape)
    ranges = jnp.stack([lower, upper])
    marginals = []
    for row in range(n_parameters):
        marginals.append([])
        for column in range(n_parameters):
            if column == row:
                marginals[row].append(
                    jax.vmap(
                        lambda mean, _invF: jax.scipy.stats.norm.pdf(
                            ranges[column],
                            mean,
                            np.sqrt(_invF)))(
                                x_d[:, column],
                                Finv[:, column, column]))
            elif column < row:
                X, Y = np.meshgrid(ranges[row], ranges[column])
                unravelled = np.vstack([X.ravel(), Y.ravel()]).T
                marginals[row].append(
                    jax.vmap(
                        lambda mean, _Finv: jax.scipy.stats.multivariate_normal.pdf(
                            unravelled, mean, _Finv
                        ).reshape(((gridsize[column], gridsize[row]))))(
                        x_d[:, [row, column]],
                        Finv[:,
                            [row, row, column, column],
                            [row, column, row, column]].reshape((n_targets, 2, 2))))
    return marginals

# @eqx.filter_jit
# def get_x_hat(
#     key: Key,
#     alpha: Array,
#     fiducials: Array,
#     net: Net,
#     fiducials_and_derivatives: Tuple[Array, Array],
#     data_covariance: Array,
#     F_planck: Optional[Array] = None
# ) -> Array:
#     F, (x0, mu_f_alpha_mean, C_f, C_f_inv) = get_F(
#         fiducials, net, fiducials_and_derivatives, F_planck
#     )
#     mu = x0.mean(axis=0)
#     Finv = jnp.linalg.inv(F) # Planck added if required

#     fiducials_mean = fiducials.mean(axis=0)
#     noise = jr.multivariate_normal(
#         key, jnp.zeros_like(fiducials_mean), data_covariance
#     )
#     obs = fiducials_mean + noise
#     x_d = jax.vmap(net)(obs)

#     mle = lambda x: jnp.einsum(
#         "pq, rq, rs, s -> p", 
#         Finv,
#         mu_f_alpha_mean, 
#         C_f_inv, 
#         x - mu
#     )    
#     s_x_d = alpha + jax.vmap(mle)(x_d)

#     # qMLEs and summaries
#     return s_x_d, x_d # score compressed summaries of IMNN Gaussian likelihood and IMNN outputs

