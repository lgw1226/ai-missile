import numpy as np
import torch as th


def crop(img: np.ndarray):
    h, w = img.shape

    r_min = 0.1
    r_max = 0.15
    height = np.random.randint(int(h * r_min), int(h * r_max))
    width = np.random.randint(int(w * r_min), int(w * r_max))

    top = np.random.randint(0, h - height + 1)
    left = np.random.randint(0, w - width + 1)

    bottom = top + height
    right = left + width

    fill = np.random.randint(0, 255)
    img[top:bottom, left:right] = fill

    return img

def jitter(img: np.ndarray):
    h, w = img.shape
    flat_img = img.flatten()

    r = 0.001
    jitter_idxs = np.random.randint(1, h * w, size=(int(h * w * r)))
    jitter_values = np.random.randint(0, 255, size=(int(h * w * r)))

    flat_img[jitter_idxs] = jitter_values
    img = flat_img.reshape((h, w))

    return img

def _crop(img: th.Tensor):
    d = img.device
    b, c, h, w = img.shape

    for i in range(c):
        r_min = 0.1
        r_max = 0.15
        height = th.randint(int(h * r_min), int(h * r_max), (1,), device=d)
        width = th.randint(int(w * r_min), int(w * r_max), (1,), device=d)

        top = th.randint(0, h - height + 1, (1,), device=d)
        left = th.randint(0, w - width + 1, (1,), device=d)

        bottom = top + height
        right = left + width

        fill = th.randint(0, 255, (1,), dtype=th.float32, device=d)
        img[:, i, top:bottom, left:right] = fill

    return img

def _jitter(img: th.Tensor):
    d = img.device
    img_shape = img.shape
    n = np.prod(img_shape)
    flat_img = img.flatten()

    r = 0.001
    jitter_idxs = th.randint(0, n, size=(int(n * r),), device=d)
    jitter_values = th.randint(0, 255, size=(int(n * r),), dtype=th.float32, device=d)

    flat_img[jitter_idxs] = jitter_values
    img = flat_img.reshape(img_shape)

    return img


if __name__ == '__main__':

    from icecream import ic
    import matplotlib.pyplot as plt

    # np.random.seed(1)
    # img = np.zeros(shape=(86, 86), dtype=np.uint8)
    # # img = np.array(np.arange(0, 86 * 86).reshape((86, 86)) / (86 * 86) * 255, dtype=np.uint8)
    # img = jitter(img)
    # img = crop(img)
    # img = crop(img)
    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    img = th.randn((2, 8, 86, 86))
    img = _jitter(img)
    img = _crop(img)
