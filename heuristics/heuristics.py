import numpy as np
from typing import List, Tuple, Union
import cv2
import pandas as pd
import re
from PIL import Image


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Упорядочивает точки по часовой стрелке.

    Args:
        points: набор точек (n, 2)

    Returns:
        Упорядоченный набор точек (n, 2)

    """
    if len(points.shape) != 2 and points.shape[0] == 0 and points.shape[1] != 2:
        raise ValueError('points: набор точек (n, 2)')

    center = points.mean(axis=0)
    points = np.array(
        sorted(
            points,
            key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]),
        ),
        dtype=np.float32,
    )

    points_sort_x = sorted(points, key=lambda point: point[0])[:2]
    point_sort_x_y = sorted(points_sort_x, key=lambda point: point[1])[0]
    ind = [
        ind for ind, point in enumerate(points) \
        if point[0] == point_sort_x_y[0] and point[1] == point_sort_x_y[1]
    ][0]

    points = np.concatenate([points[ind:], points[:ind]]).astype(np.float32)

    return points


def image_average_color(image: np.array) -> np.array:
    average_color_row = np.average(image, axis=0)
    average_color = np.average(average_color_row, axis=0)

    return average_color


def catalog_flag(
        image: Union[np.array, str],
        crop_coord: List[float],
        fill_color: Tuple[int, int, int] = (255, 255, 255),
        threshold: float = 247.0,
):
    if isinstance(image, str):
        image = Image.open(image)
        image = np.array(image)

    points = order_points(np.array(crop_coord).reshape(-1, 2)) * np.array(
        [image.shape[1], image.shape[0]]
    )
    result_image = cv2.rectangle(
        image,
        (int(points[0][0]), int(points[0][1])),
        (int(points[1][0] + 0.5), int(points[1][1] + 0.5)),
        fill_color,
        -1,
    )
    average_color = image_average_color(result_image)

    return all(average_color >= threshold)


def find_text_patterns(
    text: str, include_re: List[re.Pattern], exclude_re: List[re.Pattern] = None
):
    text = text.lower()
    if not exclude_re:
        return int(any([pattern.findall(text) for pattern in include_re]))
    return int(
        all(
            [
                any([pattern.findall(text) for pattern in include_re]),
                not any([pattern.findall(text) for pattern in exclude_re]),
            ]
        )
    )


def calculate_text_heuristics(
    title_desc_df: pd.DataFrame, include_re: List[re.Pattern], exclude_re: List[re.Pattern] = None
):
    title_desc_df['title_match'] = title_desc_df['title'].apply(
        lambda x: find_text_patterns(
            text=x,
            include_re=include_re,
            exclude_re=exclude_re,
        )
    )
    title_desc_df['description_match'] = title_desc_df['description'].apply(
        lambda x: find_text_patterns(
            text=x,
            include_re=include_re,
            exclude_re=exclude_re,
        )
    )

    return title_desc_df
