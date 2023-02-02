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
    plt.figure(figsize=(15, 15))

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
            title += f' {labels_2[1]}: {labels_2[0][i]}'

        if labels_3 is not None:
            if isinstance(labels_3[0][i], float):
                labels_3[0][i] = round(labels_3[0][i], 4)
            title += f' {labels_3[1]}: {labels_3[0][i]}'

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


def get_num_samples_per_class(class_counts, sample_size):
    class_counts_rest = class_counts.copy()

    n_classes = len(class_counts)
    sample_size_per_target = sample_size // n_classes

    class_to_sample_size = {}
    current_sample_size = sample_size_per_target
    ready_targets = []

    while True:
        values = [target for target, count in class_counts_rest.items() if
                  count <= current_sample_size]
        # time.sleep(1)
        if len(values) > 0:
            ready_targets += values
            for value in values:
                class_to_sample_size[value] = class_counts_rest.pop(value)
            if n_classes == len(ready_targets):
                break
            current_sample_size = (sample_size - sum(class_to_sample_size.values())) // (
                    n_classes - len(ready_targets)
            )
        else:
            for target in class_counts_rest.keys():
                class_to_sample_size[target] = current_sample_size
                class_counts_rest[target] -= current_sample_size

            break

    s = sum(class_to_sample_size.values())
    for target, count in class_counts_rest.items():
        if s >= sample_size:
            break
        if count > 0:
            class_to_sample_size[target] += 1
            s += 1

    return class_to_sample_size


def max_min_with_diversity(probas, sample_size):
    class_counts = Counter(probas.argmax(axis=1))
    sample_size_per_target = get_num_samples_per_class(class_counts, sample_size)

    indexes = []

    for target, sample_size in sample_size_per_target.items():
        target_indexes = np.argwhere(probas.argmax(axis=1) == target)[:, 0]
        max_probas = probas[target_indexes, :].max(axis=1)

        max_indexes = np.argsort(max_probas)[:sample_size]

        indexes += list(target_indexes[max_indexes])

    return indexes, sample_size_per_target


def margin_with_diversity(probas, sample_size):
    class_counts = Counter(probas.argmax(axis=1))
    sample_size_per_target = get_num_samples_per_class(class_counts, sample_size)

    indexes = []

    for target, sample_size in sample_size_per_target.items():
        target_indexes = np.argwhere(probas.argmax(axis=1) == target)[:, 0]
        probas_sorted = np.sort(probas[target_indexes, :], axis=1)
        margin = probas_sorted[:, -1] - probas_sorted[:, -2]
        #         if target == 8:
        #             print(probas_sorted[:2])
        #             print(probas_sorted[:,-1], probas_sorted[:,-2])

        margin_indexes = np.argsort(margin)[:sample_size]

        indexes += list(target_indexes[margin_indexes])

    return indexes, sample_size_per_target


def load_images(urls, save_dir):
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
    # wait for all threads to finish
    # You can continue doing whatever you want and
    # join the threads when you finally need the results.
    # They will fatch your urls in the background without
    # blocking your main application.
    map(lambda t: t.join(), threads)
