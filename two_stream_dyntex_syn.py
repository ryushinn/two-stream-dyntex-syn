import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import jax, optax
import metrics
import VGG, MSOEmultiscale
from utils import preprocess_exemplar


parser = argparse.ArgumentParser()

parser.add_argument("--exemplar_path", type=str)
parser.add_argument("--size", type=int, default=256)

parser.add_argument("--n_iter", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1.0)

args = parser.parse_args()


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    # load exemplar
    exemplar_path = Path(args.exemplar_path)
    ## list all files in exemplar_path with .png ext, sorted by filename
    files = list(exemplar_path.glob("*.png"))
    files = sorted(files, key=lambda x: x.name)

    exemplars = [Image.open(file) for file in files]
    exemplars = [
        preprocess_exemplar(exemplar, (args.size, args.size)) for exemplar in exemplars
    ]
    exemplars_np = np.array(
        [
            np.array(exemplar, dtype=np.float32).transpose(2, 0, 1) / 255.0
            for exemplar in exemplars
        ]
    )

    # load VGG19 / motion model and their loss functions
    vgg19 = VGG.load_pretrained_VGG19("weights/vgg19.npy")
    appearance_loss = metrics.create_appearance_loss(vgg19, exemplars_np)
    msoemultiscale = MSOEmultiscale.load_pretrained_MSOEmultiscale(
        "weights/two_stream_dynamic_model.npy"
    )
    motion_loss = metrics.create_motion_loss(msoemultiscale, exemplars_np)

    # initialize pixels
    key, subkey = jax.random.split(key)
    frames = exemplars_np.mean(axis=(-1, -2), keepdims=True) + 0.01 * jax.random.normal(
        subkey, exemplars_np.shape
    )

    # initialize optimizer
    optimizer = optax.lbfgs(args.lr)
    opt_state = optimizer.init(frames)

    # define update func for each iteration
    @jax.jit
    def update(frames, opt_state):
        lossfn = lambda frames: appearance_loss(frames) + 1e6 * motion_loss(frames)
        loss, grads = jax.value_and_grad(lossfn)(frames)
        updates, opt_state = optimizer.update(
            grads, opt_state, frames, value=loss, grad=grads, value_fn=lossfn
        )
        frames = optax.apply_updates(frames, updates)
        return loss, frames, opt_state

    # training loop
    for it in (bar := tqdm(range(args.n_iter), desc="iter")):
        loss, frames, opt_state = update(frames, opt_state)
        bar.postfix = f"loss: {loss:.5f}"

    # save the result using PIL
    images = [
        Image.fromarray(
            np.array(frame.clip(0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        for frame in frames
    ]
    ## save each frame and a GIF
    output_path = exemplar_path / "output"
    output_path.mkdir(exist_ok=True)
    for i, image in enumerate(images):
        image.save(output_path / f"frame_{i:04d}.png")
    images[0].save(
        output_path / "animation.gif",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )
