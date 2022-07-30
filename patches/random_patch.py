import random
from PIL import Image, ImageDraw

from constants import IMG_SIZE
random.seed = 42


def get_random_pixels(pixels_no: int, size: (int, int)):
    pixels = []
    for i in range(pixels_no):
        x = int(random.random() * size[0])
        y = int(random.random() * size[1])
        pixels.append((x, y))
    return pixels


def get_random_pattern(pixels: int):
    size = IMG_SIZE

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    pixels = get_random_pixels(pixels, size)
    for x, y in pixels:
        draw.point((x, y), fill=255)
    return mask


def merge_image_and_mask(prev_image: Image, mask):
    size = IMG_SIZE
    prev_image = prev_image.resize(size)
    prev_image = prev_image.convert("RGBA")
    paste_mask = prev_image.copy()
    paste_mask.putalpha(0)
    prev_image.paste(paste_mask, mask=mask)
    prev_image.paste(paste_mask, mask=mask)
    return prev_image


def poison_with_pixels(file_loaded, mask):
    return merge_image_and_mask(file_loaded, mask)

