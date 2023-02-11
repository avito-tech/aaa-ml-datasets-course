from collections import Counter
import numpy as np
from typing import Tuple, List, Dict


def get_num_samples_per_class(class_counts: Dict, sample_size: int) -> Dict[int, int]:
    """
    Принимает на вход словарь class_id -> class_count
    Возвращает словарь class_id -> sample_size
    так как не во всех классах есть хотя бы  sample_size // n_classes примеров,
    то основной смысл функции сделать выборку равномернее и добрать
    примеров из более многочисленных классов
    Args:
        class_counts: словарь с кол-во примеров в выборке по каждому
                    классу {0:100, 1: 234, ..., 19: 33}
        sample_size: итоговый размер семпла который хочется получить

    Returns:

    """
    class_counts_rest = class_counts.copy()

    n_classes = len(class_counts)
    sample_size_per_target = sample_size // n_classes

    class_to_sample_size = {}
    current_sample_size = sample_size_per_target
    ready_targets = []

    while True:
        values = [target for target, count in class_counts_rest.items() if
                  count <= current_sample_size]
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


def max_min_with_diversity(probas: np.ndarray, sample_size: int) -> Tuple[List, Dict]:
    """
    Семплирование примеров, где самый вероятный класс имеет
    наименьшую вероятность среди остальных примеров
    Args:
        probas: матрица NxC - вероятности по всем классам C -(кол-во классов)
        sample_size: размер итогового семпла

    Returns:

    """
    class_counts = Counter(probas.argmax(axis=1))
    sample_size_per_target = get_num_samples_per_class(class_counts, sample_size)

    indexes = []

    for target, sample_size in sample_size_per_target.items():
        target_indexes = np.argwhere(probas.argmax(axis=1) == target)[:, 0]
        max_probas = probas[target_indexes, :].max(axis=1)

        max_indexes = np.argsort(max_probas)[:sample_size]

        indexes += list(target_indexes[max_indexes])

    return indexes, sample_size_per_target


def margin_with_diversity(probas: np.ndarray, sample_size: int) -> Tuple[List, Dict]:
    """
    Семплирование примеров где модель сомневается между 2 классами
    Args:
        probas: матрица NxC - вероятности по всем классам C -(кол-во классов)
        sample_size: размер итогового семпла

    Returns:

    """
    class_counts = Counter(probas.argmax(axis=1))
    sample_size_per_target = get_num_samples_per_class(class_counts, sample_size)

    indexes = []

    for target, sample_size in sample_size_per_target.items():
        target_indexes = np.argwhere(probas.argmax(axis=1) == target)[:, 0]
        probas_sorted = np.sort(probas[target_indexes, :], axis=1)
        margin = probas_sorted[:, -1] - probas_sorted[:, -2]

        margin_indexes = np.argsort(margin)[:sample_size]

        indexes += list(target_indexes[margin_indexes])

    return indexes, sample_size_per_target
