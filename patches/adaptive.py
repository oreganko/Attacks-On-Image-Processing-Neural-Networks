import numpy as np
from PIL import Image
from constants import IMG_SIZE


def load_adaptive(image, mask):
    size = IMG_SIZE
    overshoot = 0.02
    image = image.resize(size).convert("RGB")
    r_tot = mask.resize(size).convert("RGB")

    r_tot = np.asarray(r_tot) // 20
    image = np.asarray(image)

    pert_image = image + (1 + overshoot) * r_tot
    pert_image = np.where(pert_image > 255, 255, pert_image)
    img_to_save = Image.fromarray(pert_image.astype(np.uint8))

    return img_to_save


def load_mask(path):
    return Image.open(path)
