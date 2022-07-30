from PIL import Image, ImageDraw


def load_chess_pattern(prev_image: Image, opacity: float, layout: (float, float)):
    size = prev_image.size
    prev_image = prev_image.convert("RGBA")

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for x in range(int(layout[0] * size[0]), size[0], 2):
        for y in range(int(layout[1] * size[1]), size[1], 2):
            draw.point((x, y), fill=255)

    prev_image1 = prev_image.copy()
    prev_image1.putalpha(int(opacity * 255))
    prev_image.paste(prev_image1, mask=mask)

    return prev_image


def poison_with_full_covering(file_loaded):
    return load_chess_pattern(file_loaded, 0.9, (0, 0))


def poison_with_small_covering(file_loaded):
    return load_chess_pattern(file_loaded, 0.9, (0.65, 0.65))
