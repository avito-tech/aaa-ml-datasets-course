import os
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

import threading
import urllib


from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


from .settings import MAX_IMG_SIZE


class PadCustom(transforms.Pad):
    """
    принимает на вход до какого размера нужно заппадить,
    а transforms.Pad на сколько нужно западдить
    """

    def forward(self, img: Union[Image.Image, torch.Tensor]):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.
        Returns:
            PIL Image or Tensor: Padded image.
        """
        if isinstance(img, torch.Tensor):
            img_size = img.size()[1:]
        else:
            img_size = img.size

        height_diff = self.padding[0] - img_size[0]
        width_diff = self.padding[1] - img_size[1]

        padding = (
            height_diff // 2,
            width_diff // 2,
            height_diff // 2 + height_diff % 2,
            width_diff // 2 + width_diff % 2,
        )

        return F.pad(img, padding, self.fill, self.padding_mode)


def get_preprocessor(pad_size=MAX_IMG_SIZE):
    preprocessor = transforms.Compose(
        [
            PadCustom(pad_size),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return preprocessor


def plot_imgs_with_labels(
    image_paths: List[str],
    labels_1: Tuple[List[Any], str] = None,
    labels_2: Tuple[List[Any], str] = None,
    labels_3: Tuple[List[Any], str] = None,
):
    plt.figure(figsize=(20, 20))

    n_subplots = len(image_paths)
    n_lines = int((n_subplots + 2) // 3)

    for i in range(n_subplots):
        plt.subplot(n_lines, 3, i+1)
        img = mpimg.imread(image_paths[i])
        title = ''
        if labels_1 is not None:
            title += f'{labels_1[1]}: {labels_1[0][i]}'
        if labels_2 is not None:
            if isinstance(labels_2[0][i], float):
                labels_2[0][i] = round(labels_2[0][i], 4)
            title += f'\n{labels_2[1]}: {labels_2[0][i]}'

        if labels_3 is not None:
            if isinstance(labels_3[0][i], float):
                labels_3[0][i] = round(labels_3[0][i], 4)
            title += f'\n{labels_3[1]}: {labels_3[0][i]}'

        plt.title(title)
        plt.imshow(img)


def plot_sample(test_df, value='детская', column='label_pred', size=10):
    plot_df = test_df[test_df[column] == value]
    sample_size = min(plot_df.shape[0], size)

    sample_df = plot_df.sample(sample_size).reset_index()

    plot_imgs_with_labels(sample_df['img_path'],
                          (sample_df['label'], 'label'),
                          (sample_df['label_pred'], 'label_pred'),
                          (sample_df['proba'], 'proba'))


def _load_images(urls, save_dir):
    results = []

    def getter(url, dest):
        results.append(urllib.request.urlretrieve(url, dest))

    os.makedirs(save_dir, exist_ok=True)

    threads = []
    for url in urls:
        filename = os.path.split(url)[-1]
        t = threading.Thread(target=getter, args=(url, os.path.join(save_dir, filename)))
        t.start()
        threads.append(t)

    map(lambda t: t.join(), threads)

    image_paths = [os.path.join(save_dir, os.path.split(url)[-1]) for url in urls]

    return image_paths


def load_images(urls, save_dir, max_threads_num=50):
    image_paths = []
    for i in range(0, len(urls), max_threads_num):
        new_image_paths = _load_images(urls[i:i+max_threads_num], save_dir)
        image_paths.extend(new_image_paths)

    return image_paths
