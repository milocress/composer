from argparse import ArgumentParser, Namespace
import os
import numpy as np
from torchvision.datasets import MNIST
from typing import Any, Dict, Iterable, Tuple
from wurlitzer import pipes

from composer.datasets.streaming import StreamingDatasetWriter


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--in_root', type=str, required=True)
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--shard_size_limit', type=int, default=1 << 20)
    args.add_argument('--tqdm', type=int, default=1)
    return args.parse_args()


def get(dataset: MNIST) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy-convert and shuffle a MNIST dataset.

    Args:
        dataset (MNIST): MNIST dataset object.

    Returns:
        images (np.ndarray of np.uint8): Dataset images in NCHW.
        classes (np.ndarray of np.int64): Dataset classes.
    """
    indices = np.random.permutation(len(dataset))
    images = dataset.data[indices].numpy()
    classes = dataset.targets[indices].numpy()
    return images, classes


def each(images: np.ndarray, classes: np.ndarray) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        images (np.ndarray of np.uint8): Dataset images in NCHW.
        classes (np.ndarray of np.int64): Dataset classes.

    Yields:
        Sample dicts.
    """
    for x, y in zip(images, classes):
        yield {
            'x': x.tobytes(),
            'y': y.tobytes(),
        }


def main(args: Namespace) -> None:
    """Main: create MNIST Mosaic dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    fields = 'x', 'y'

    with pipes():
        dataset = MNIST(root=args.in_root, train=True, download=True)
    images, classes = get(dataset)
    split_dir = os.path.join(args.out_root, 'train')
    with StreamingDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(images, classes), bool(args.tqdm), len(images))

    with pipes():
        dataset = MNIST(root=args.in_root, train=False, download=True)
    images, classes = get(dataset)
    split_dir = os.path.join(args.out_root, 'val')
    with StreamingDatasetWriter(split_dir, fields, args.shard_size_limit) as out:
        out.write_samples(each(images, classes), bool(args.tqdm), len(images))


if __name__ == '__main__':
    main(parse_args())