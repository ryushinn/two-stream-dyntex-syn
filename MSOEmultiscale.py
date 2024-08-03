import jax.numpy as jnp
import numpy as np
import equinox as eqx
import jax
from functools import partial


def load_pretrained_MSOEmultiscale(weights_path):
    # get treedef from a dummy OF network
    of = MSOEmultiscale(jax.random.key(0))
    _, treedef = jax.tree_util.tree_flatten(of)

    # formulate pretrained weights as corresponding leaves
    of_jnp = np.load(weights_path, allow_pickle=True).item()
    of_jnp = jax.tree_util.tree_map_with_path(
        lambda kp, x: (
            x[..., None, None, None]
            if "msoenet.conv1.bias" in str(kp)
            else x[..., None, None] if "bias" in str(kp) else x
        ),
        of_jnp,
    )
    leaves_order = [
        "msoenet.conv1.weight",
        "msoenet.conv1.bias",
        "msoenet.conv2.weight",
        "msoenet.conv2.bias",
        "decode_conv1.weight",
        "decode_conv1.bias",
        "decode_conv2.weight",
        "decode_conv2.bias",
    ]
    leaves, _ = jax.tree_util.tree_flatten([of_jnp[k] for k in leaves_order])

    # unflatten back to model
    return jax.tree_util.tree_unflatten(treedef, leaves)


def symmetric_padding(im, padding):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = jnp.arange(-left, w + right)
    y_idx = jnp.arange(-top, h + bottom)

    def reflect(x, minx, maxx):
        """Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length"""
        rng = maxx - minx
        double_rng = 2 * rng
        mod = jnp.fmod(x - minx, double_rng)
        normed_mod = jnp.where(mod < 0, mod + double_rng, mod)
        out = jnp.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return jnp.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = jnp.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def image_resize(x, factor):
    h, w = x.shape[-2:]
    if factor > 1:
        x = jax.image.resize(
            x, (*x.shape[:-2], h * factor, w * factor), method="bilinear"
        )
    elif factor < 1:
        x = jax.image.resize(
            x, (*x.shape[:-2], int(h * factor), int(w * factor)), method="bilinear"
        )
    return x


class GaussianBlur(eqx.Module):
    kernel_size: int = eqx.field(static=True)
    kernel: jnp.array = eqx.field(static=True)

    def __init__(self, kernel_size, sigma):
        super().__init__()
        self.kernel_size = kernel_size
        kernel_weights = self.gauss2d_kernel((kernel_size, kernel_size), sigma)
        self.kernel = kernel_weights.reshape(kernel_size, kernel_size)

    def __call__(self, x):
        # x has the shape [1, H, W, 2]
        x1 = x[..., 0]
        x2 = x[..., 1]

        p = self.kernel_size // 2
        x1 = symmetric_padding(x1, (p, p, p, p))
        x2 = symmetric_padding(x2, (p, p, p, p))

        perchannel_conv2d = jax.vmap(
            partial(
                # mode wrap is not supported in jax.scipy.signal.convolve2d
                # so have to manually pad
                jax.scipy.signal.convolve2d,
                mode="valid",
            ),
            in_axes=[0, None],
        )

        x1 = image_resize(perchannel_conv2d(x1, self.kernel), factor=0.5)
        x2 = image_resize(perchannel_conv2d(x2, self.kernel), factor=0.5)

        x = jnp.stack([x1, x2], axis=-1)
        return x

    def gauss2d_kernel(self, shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = jnp.ogrid[-m : m + 1, -n : n + 1]
        h = jnp.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h = h.at[h < jnp.finfo(h.dtype).eps * h.max()].set(0)
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


class MSOEnet(eqx.Module):
    conv1: eqx.nn.Conv3d
    maxpool: eqx.nn.MaxPool2d = eqx.field(static=True)
    conv2: eqx.nn.Conv2d

    def __init__(self, key):
        super().__init__()
        key1, key2 = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv3d(1, 32, (11, 11, 2), key=key1)
        self.maxpool = eqx.nn.MaxPool2d(5, stride=1, padding=2)
        self.conv2 = eqx.nn.Conv2d(32, 64, 1, key=key2)

    def l1_normalize(self, x):
        eps = 1e-12
        norm = jnp.sum(jnp.abs(x), axis=0, keepdims=True)
        return x / (jnp.maximum(norm, eps))

    def __call__(self, x):
        # x has the shape [1, H, W, 2]
        x0 = x[..., 0]
        x1 = x[..., 1]
        x0 = symmetric_padding(x0, (5, 5, 5, 5))
        x1 = symmetric_padding(x1, (5, 5, 5, 5))
        x = jnp.stack([x0, x1], axis=-1)
        x = self.conv1(x)
        x = jnp.square(x)
        x = x.squeeze(-1)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.l1_normalize(x)
        # x has the shape [64, H, W]
        return x


class MSOEmultiscale(eqx.Module):
    n_scales: int = eqx.field(static=True)
    msoenet: MSOEnet
    gaussian_blur: GaussianBlur = eqx.field(static=True)
    decode_conv1: eqx.nn.Conv2d
    decode_conv2: eqx.nn.Conv2d

    def __init__(self, key):
        super().__init__()
        self.n_scales = 5
        key1, key2, key3 = jax.random.split(key, 3)
        self.msoenet = MSOEnet(key1)

        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=2.0)

        self.decode_conv1 = eqx.nn.Conv2d(64 * self.n_scales, 64, 3, key=key2)
        self.decode_conv2 = eqx.nn.Conv2d(64, 2, 1, key=key3)

    def contrast_norm(self, x):
        # x has the shape [1, H, W, 2]
        eps = 1e-12
        x_mean = jnp.mean(x, axis=(0, 1, 2, 3), keepdims=True)

        x_var = jnp.var(x, axis=(0, 1, 2, 3), keepdims=True)
        x_std = jnp.sqrt(x_var + eps)
        x = (x - x_mean) / x_std

        return x

    def __call__(self, x, return_features=False):
        # x has the shape [1, H, W, 2]
        features = []
        x0 = self.contrast_norm(x)
        h0 = self.msoenet(x0)

        x1 = self.gaussian_blur(x0)
        h1 = self.msoenet(x1)

        x2 = self.gaussian_blur(x1)
        h2 = self.msoenet(x2)

        x3 = self.gaussian_blur(x2)
        h3 = self.msoenet(x3)

        x4 = self.gaussian_blur(x3)
        h4 = self.msoenet(x4)

        z0 = h0
        z1 = image_resize(h1, factor=2)
        z2 = image_resize(h2, factor=4)
        z3 = image_resize(h3, factor=8)
        z4 = image_resize(h4, factor=16)

        z = jnp.concatenate([z0, z1, z2, z3, z4], axis=0)
        features.append(z)
        z = symmetric_padding(z, (1, 1, 1, 1))
        x = self.decode_conv1(z)

        x = jax.nn.relu(x)
        flow = self.decode_conv2(x)
        flow = flow.at[1].set(-flow[1])

        if return_features:
            return flow, features
        else:
            return flow
