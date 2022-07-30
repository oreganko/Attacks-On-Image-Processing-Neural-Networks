import numpy as np
from PIL import Image


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


def get_gradient_mask(size, start_color, end_color, horizontal=True):
    array = get_gradient_3d(size[0], size[1], (start_color,) * 3, (end_color,) * 3, (horizontal,) * 3)
    im = Image.fromarray(np.uint8(array))
    return im.convert("L")


def load_gradient(prev_image: Image, start_color, end_color, horizontal=True):
    size = prev_image.size
    prev_image = prev_image.convert("RGBA")

    mask = get_gradient_mask(size, start_color, end_color, horizontal)

    prev_image1 = prev_image.copy()
    prev_image1.putalpha(0)
    prev_image.paste(prev_image1, mask=mask)

    return prev_image


def load_small_diff_gradient(file_loaded):
    return load_gradient(file_loaded, 0, 100)


def load_big_diff_gradient(file_loaded):
    return load_gradient(file_loaded, 0, 255)
