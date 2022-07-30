from math import sqrt

from PIL import Image

from constants import *


def load_logo(size: [int, int]):
    return Image.open(logo_path).resize(size).convert("RGBA")


def change_layout(image, area=1.0, rotation=0):
    size = [int(dim * sqrt(area)) for dim in image.size]
    image = image.resize(size).rotate(rotation)

    return image


def get_mask(size, area: float, opacity: float, layout: (float, float), rotation: int):
    logo = load_logo(size)
    logo = change_layout(logo, area, rotation)
    fin_layout = [int(size_d * layout_d) for size_d, layout_d in zip(size, layout)]

    # create pasting mask
    mask_im = Image.new("L", size, 0)
    fn = lambda x: int(opacity * 255) if x > 0 else 0
    logo_mask = logo.convert("L").point(fn, mode='L')
    mask_im.paste(logo_mask, box=fin_layout)

    # create image to paste with mask
    to_paste = Image.new("RGBA", size, 0)
    to_paste.paste(logo, box=fin_layout)
    return to_paste, mask_im


def merge_image_and_mask(prev_image: Image, paste_mask, mask):
    prev_image = prev_image.convert("RGBA")
    prev_image.paste(paste_mask, mask=mask)
    return prev_image


def poison_with_full_visible(file_loaded):
    paste_mask, mask = get_mask(file_loaded.size, 0.1, 1, [0.7, 0.7], 0)
    return merge_image_and_mask(file_loaded, paste_mask, mask)


def poison_with_barely_visible(file_loaded):
    paste_mask, mask = get_mask(file_loaded.size, 0.1, 0.1, [0.7, 0.7], 0)
    return merge_image_and_mask(file_loaded, paste_mask, mask)
