import jax
import jax.numpy as jnp


def gram_matrix(f):
    f = f.reshape(f.shape[0], -1)
    gram_matrix = f @ f.transpose()

    gram_matrix = gram_matrix / f.shape[-1]
    return gram_matrix


def create_appearance_loss(features, exemplars):
    features_exemplars = jax.vmap(features)(exemplars)
    gmatrices_exemplars = jax.tree_map(jax.vmap(gram_matrix), features_exemplars)
    gmatrics_avg = jax.tree_map(lambda x: jnp.mean(x, axis=0), gmatrices_exemplars)
    mse = lambda x, y: jnp.mean((x - y) ** 2)

    def gram_loss(samples):
        # key is never used here but declared for API compatibility
        features_samples = jax.vmap(features)(samples)
        gmatrices_samples = jax.tree_map(jax.vmap(gram_matrix), features_samples)

        loss = sum(jax.tree_map(mse, gmatrics_avg, gmatrices_samples))
        return loss

    return gram_loss


def rgb_to_grayscale(x):
    return jnp.einsum("i, ihw->hw", jnp.array([0.2989, 0.5870, 0.1140]), x)[None, ...]


def create_motion_loss(MSOEmultiscale, exemplars):
    exemplars_grayscale = jax.vmap(rgb_to_grayscale)(exemplars)
    inputs = jnp.stack(
        [
            jnp.stack([exemplars_grayscale[i], exemplars_grayscale[i + 1]], axis=-1)
            for i in range(len(exemplars) - 1)
        ]
    )

    flows_exemplars, features_exemplars = jax.vmap(MSOEmultiscale, in_axes=(0, None))(
        inputs, True
    )

    gmatrices_exemplars = jax.tree_map(jax.vmap(gram_matrix), features_exemplars)
    gmatrices_avg = jax.tree_map(lambda x: jnp.mean(x, axis=0), gmatrices_exemplars)
    mse = lambda x, y: jnp.mean((x - y) ** 2)

    def gram_loss(samples):
        samples_grayscale = jax.vmap(rgb_to_grayscale)(samples)
        inputs = jnp.stack(
            [
                jnp.stack([samples_grayscale[i], samples_grayscale[i + 1]], axis=-1)
                for i in range(len(samples) - 1)
            ]
        )
        flow_samples, features_samples = jax.vmap(MSOEmultiscale, in_axes=(0, None))(
            inputs, True
        )
        gmatrices_samples = jax.tree_map(jax.vmap(gram_matrix), features_samples)

        loss = sum(jax.tree_map(mse, gmatrices_avg, gmatrices_samples))
        return loss

    return gram_loss
