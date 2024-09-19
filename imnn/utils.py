import os, json
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import equinox as eqx
import numpy as np 


def get_sharding():
    n_devices = len(jax.local_devices())
    print(f"Running on {n_devices} devices: \n\t{jax.local_devices()}")

    use_sharding = n_devices > 1
    # Sharding mesh: speed and allow training on high resolution?
    if use_sharding:
        # Split array evenly across data dimensions, this reshapes automatically
        mesh = Mesh(jax.devices(), ('x',))
        sharding = NamedSharding(mesh, P('x'))
        print(f"Sharding:\n {sharding}")
    else:
        sharding = None


def count_params(net):
    return sum(
        x.size for x in jax.tree_util.tree_leaves(net)
        if eqx.is_array(x)
    )


def scale_fn(x): 
    x = 1. / x 
    y = jnp.log(x + jnp.sqrt(jnp.square(x) + 1.))
    return y


def log_scale_pdfs(scale_fn, fiducials, latin_pdfs, derivatives):
    """ Log-scale data, returning adjusted derivatives. """
    print("nans?", ~jnp.all(jnp.isfinite(fiducials)), ~jnp.all(jnp.isfinite(latin_pdfs)))
    
    # Derivatives of log(d_fiducial) w.r.t. d_fiducial:
    # > d(log(pdf))/d(alpha) = d(log(pdf))/d(pdf) * d(pdf)/d(alpha)
    # Assuming the derivatives supplied by Quijote are run at the first 500 fiducials  
    dfdd = jax.vmap(jax.jacfwd(scale_fn))(fiducials[:len(derivatives)])

    print("log d's / d's", dfdd.shape, derivatives.shape)

    adjusted_derivatives = jnp.einsum("nij, nkj -> nki", dfdd, derivatives)

    print("derivatives", derivatives.shape)
    return scale_fn(fiducials), scale_fn(latin_pdfs), adjusted_derivatives


import abc
import jax.random as jr
from jaxtyping import Key, Array

class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, data, targets, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class _InMemoryDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        simulations: Array, 
        parameters: Array = None, 
        *, 
        key: Key
    ): 
        self.simulations = simulations 
        self.parameters = parameters 
        self.key = key

    @property 
    def n_batches(self, batch_size):
        return max(int(self.simulations.shape[0] / batch_size), 1)

    def loop(self, batch_size: int):
        # Loop through dataset, batching, while organising data for NPE or NLE
        dataset_size = self.simulations.shape[0]
        one_batch = batch_size >= dataset_size
        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            # Yield whole dataset if batch size is larger than dataset size
            if one_batch:
                out = self.simulations
                if self.parameters is not None:
                    out = (out, self.parameters)
                yield out
            else:
                key, subkey = jr.split(key)
                perm = jr.permutation(subkey, indices)
                start = 0
                end = batch_size
                while end < dataset_size:
                    batch_perm = perm[start:end]
                    out = self.simulations[batch_perm]
                    if self.parameters is not None:
                        out = (out, self.parameters[batch_perm])
                    yield out
                    start = end
                    end = start + batch_size


def get_dataloaders(
    key, data_tuple_train, data_tuple_valid
):
    # Xt, Yt = data_tuple_train
    # Xv, Yv = data_tuple_valid
    # return (
    #     _InMemoryDataLoader(Xt, Yt, key=key), 
    #     _InMemoryDataLoader(Xv, Yv, key=key)
    # )
    Xt = data_tuple_train
    Xv = data_tuple_valid
    return (
        _InMemoryDataLoader(Xt, key=key), 
        _InMemoryDataLoader(Xv, key=key)
    )


def get_fiducials_derivatives_dataloaders(
    key, fids_and_dd_train, fids_and_dd_valid
):
    ft, ddt = fids_and_dd_train 
    fv, ddv = fids_and_dd_valid 
    return (
        _InMemoryDataLoader(ft, ddt, key=key), 
        _InMemoryDataLoader(fv, ddv, key=key)
    )


# def get_dataloaders(
#     key,
#     data_tuple_train,
#     data_tuple_valid, 
#     batch_size,
#     valid_batch_size=None,
#     repeat=False):
#     n_train = data_tuple_train.shape[0] 

#     train_dataset = tf.data.Dataset.from_tensor_slices(data_tuple_train)
#     valid_dataset = tf.data.Dataset.from_tensor_slices(data_tuple_valid)

#     seed = int(key.sum())

#     if repeat:
#         train_dataset = (
#             train_dataset
#             .repeat()
#             .shuffle(n_train, seed=seed)
#             .batch(batch_size, drop_remainder=True)
#         )
#         valid_dataset = valid_dataset.repeat()
#     else:
#         train_dataset = (
#             train_dataset
#             .shuffle(n_train, seed=seed)
#             .batch(batch_size, drop_remainder=True)
#         )

#     valid_dataset = valid_dataset.batch(
#         valid_batch_size if valid_batch_size is not None else batch_size
#     ) 

#     return (
#         train_dataset.as_numpy_iterator(), 
#         valid_dataset.as_numpy_iterator()
#     )


# def get_fiducials_derivatives_dataloader(
#     rng, 
#     fids_and_dd_train, 
#     fids_and_dd_valid, 
#     batch_size, 
#     repeat=False
# ):
#     """ Get a dataloader that zips the fiducials and their derivatives together. """
#     n_train, n_valid = len(fids_and_dd_train[0]), len(fids_and_dd_valid[0])

#     train_dataset = tf.data.Dataset.from_tensor_slices(fids_and_dd_train)
#     valid_dataset = tf.data.Dataset.from_tensor_slices(fids_and_dd_valid)

#     seed = int(rng.sum())
#     train_dataset = train_dataset.shuffle(n_train, seed=seed)

#     if repeat:
#         train_dataset = train_dataset.repeat()
#         valid_dataset = valid_dataset.repeat()

#     # Batch, attending to length of derivatives being greater than number of derivatives 
#     train_dataset = train_dataset.batch(
#         n_train if batch_size > n_train else batch_size, drop_remainder=True
#     )
#     valid_dataset = valid_dataset.batch(
#         n_valid if batch_size > n_valid else batch_size, drop_remainder=True
#     )
#     return (
#         train_dataset.as_numpy_iterator(), 
#         valid_dataset.as_numpy_iterator()
#     )


def split_fiducials_and_derivatives(fiducials, derivatives, split):
    """
        Returns matched fids/derivatives in first two tuples
        and train/valid fiducial pdfs.
    """
    n_pdfs, *_ = fiducials.shape
    n_derivatives, *_ = derivatives.shape

    # Need dataloader to return (d, d(data)/dalpha, alpha)
    # though during training alpha is always alpha^0
    n_train = int(split * n_pdfs) - n_derivatives # first 500 pdfs removed 
    n_valid = n_pdfs - n_train

    # Split derivatives (same as in dataloaders)
    # and PDFs (0 -> 500), where first 500 given to derivatives dataloaders
    # Dataloaders of fiducial pdfs for epoch, don't use first 500 pdfs belonging to derivatives
    dd_train, dd_valid = jnp.split(
        derivatives, [int(split * n_derivatives)]
    )
    fiducials_train, fiducials_valid = jnp.split(
        fiducials[:n_derivatives], [int(split * n_derivatives)]
    )
    _d0_train, _d0_valid = jnp.split(
        fiducials[n_derivatives:], [split * (n_pdfs - n_derivatives)]
    )
    return (
        (dd_train, dd_valid),
        (fiducials_train, fiducials_valid),
        (_d0_train, _d0_valid),
    )


def get_F_and_logdetF(Finv):
    F = jnp.linalg.inv(Finv)
    return F, np.prod(jnp.linalg.slogdet(F))


def get_metric_arrays(_d0_train, _d0_valid, batch_size):
    """
        Returns containers for metrics in each epoch
    """
    L_steps_train = np.zeros((int(len(_d0_train) / batch_size) + 1, 2))
    C_steps_train = np.zeros((int(len(_d0_train) / batch_size) + 1, 2))
    L_steps_valid = np.zeros((int(len(_d0_valid) / batch_size) + 1, 2))
    C_steps_valid = np.zeros((int(len(_d0_valid) / batch_size) + 1, 2))
    return (
        L_steps_train, L_steps_valid,
        C_steps_train, C_steps_valid,
    )

def save_model(filename, hyperparams, model):
    """
        Make attribute of NDEs/IMNNs
        that returns all attributes as a dict...
    """
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(model, filename):
    with open(filename, "rb") as f:
        hyperparameters = json.loads(f.readline().decode())
        model = model(**hyperparameters)
        return eqx.tree_deserialise_leaves(f, model)