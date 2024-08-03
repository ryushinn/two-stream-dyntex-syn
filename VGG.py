import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, List
from equinox import field
import numpy as np


class VGG19(eqx.Module):
    block1: list
    block2: list
    block3: list
    block4: list
    activation: Callable = field(static=True)
    downsampling: Callable = field(static=True)

    def __init__(self, key):
        keys = jax.random.split(key, 12)
        self.block1 = [
            eqx.nn.Conv2d(
                3, 64, 3, key=keys[0], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                64, 64, 3, key=keys[1], padding="SAME", padding_mode="REFLECT"
            ),
        ]

        self.block2 = [
            eqx.nn.Conv2d(
                64, 128, 3, key=keys[2], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                128, 128, 3, key=keys[3], padding="SAME", padding_mode="REFLECT"
            ),
        ]

        self.block3 = [
            eqx.nn.Conv2d(
                128, 256, 3, key=keys[4], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                256, 256, 3, key=keys[5], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                256, 256, 3, key=keys[6], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                256, 256, 3, key=keys[7], padding="SAME", padding_mode="REFLECT"
            ),
        ]

        self.block4 = [
            eqx.nn.Conv2d(
                256, 512, 3, key=keys[8], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                512, 512, 3, key=keys[9], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                512, 512, 3, key=keys[10], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                512, 512, 3, key=keys[11], padding="SAME", padding_mode="REFLECT"
            ),
        ]

        self.activation = jax.nn.relu
        self.downsampling = eqx.nn.AvgPool2d((2, 2), stride=2)

    def __call__(self, x):
        features = []

        x = x[[2, 1, 0], ...]

        x = 255 * x - jnp.array([103.939, 116.779, 123.68]).reshape(3, 1, 1)

        # block1
        for conv in self.block1:
            x = self.activation(conv(x))
        features.append(x)

        x = self.downsampling(x)

        # block2
        for conv in self.block2:
            x = self.activation(conv(x))
        features.append(x)

        x = self.downsampling(x)

        # block3
        for conv in self.block3:
            x = self.activation(conv(x))
        features.append(x)

        x = self.downsampling(x)

        # block4
        for conv in self.block4:
            x = self.activation(conv(x))
        features.append(x)

        x = self.downsampling(x)

        return features


def load_pretrained_VGG19(weights_path):
    # get treedef from a dummy VGG
    VGG_dummy = VGG19(jax.random.key(0))
    _, treedef = jax.tree_util.tree_flatten(VGG_dummy)

    # formulate pretrained weights as corresponding leaves
    vgg_jnp = np.load(weights_path, allow_pickle=True).item()
    vgg_jnp = jax.tree_util.tree_map_with_path(
        lambda kp, x: x[..., None, None] if "bias" in str(kp) else x, vgg_jnp
    )
    leaves_order = [
        "block1_conv1.weight",
        "block1_conv1.bias",
        "block1_conv2.weight",
        "block1_conv2.bias",
        "block2_conv1.weight",
        "block2_conv1.bias",
        "block2_conv2.weight",
        "block2_conv2.bias",
        "block3_conv1.weight",
        "block3_conv1.bias",
        "block3_conv2.weight",
        "block3_conv2.bias",
        "block3_conv3.weight",
        "block3_conv3.bias",
        "block3_conv4.weight",
        "block3_conv4.bias",
        "block4_conv1.weight",
        "block4_conv1.bias",
        "block4_conv2.weight",
        "block4_conv2.bias",
        "block4_conv3.weight",
        "block4_conv3.bias",
        "block4_conv4.weight",
        "block4_conv4.bias",
    ]
    leaves, _ = jax.tree_util.tree_flatten([vgg_jnp[k] for k in leaves_order])

    # unflatten back to model
    return jax.tree_util.tree_unflatten(treedef, leaves)
