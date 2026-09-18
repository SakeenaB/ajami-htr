"""
augment.py
----------
Training-time image augmentation, written directly against OpenCV.

Two purposes, and they are distinct.

The first is regularisation. The corpus holds 4,344 training lines, which is
small for a 7M-parameter recogniser. Without augmentation the model sees the
identical pixels of each line on every epoch and memorises them: training loss
falls towards zero while validation error barely moves. Augmentation shows a
slightly different version of each line every epoch, so the model must learn
what the writing IS rather than what these particular photographs look like.

The second is error decorrelation (Section 3.5.4). An ensemble only improves on
its members if they fail differently. Giving each base model a different family
of distortion - geometric for the CRNN, morphological for Kraken, photometric
for TrOCR - pushes them towards different weaknesses, which is what the
meta-learner later exploits.

Implemented with OpenCV rather than a wrapper library so that every operation
is visible, testable, and stable across environments.

One deliberate omission: no salt-and-pepper or speckle noise anywhere. It
scatters isolated dark specks across the image, and an isolated dark speck is
exactly what an Ajami diacritic looks like. Training with it would teach the
models to discard the marks this project exists to preserve.
"""

from __future__ import annotations

import cv2
import numpy as np

# Images are uint8 greyscale, so 255 is white - the page background. Filling a
# rotated corner with black would introduce ink that was never on the page.
PAGE = 255


# ---------------------------------------------------------------------------
# individual operations
# ---------------------------------------------------------------------------

def affine(image, rng, rotate=2.0, shear=4.0, scale=0.05, shift=0.03):
    """
    Small rotation, horizontal shear, scale and vertical shift.

    Limits are deliberately tight. A diacritic sits a few pixels above its base
    letter; an aggressive warp detaches it or merges it with a neighbouring
    stroke, which corrupts the label rather than augmenting the image. Shear is
    horizontal only, mimicking natural variation in writing slant.
    """
    h, w = image.shape
    angle = rng.uniform(-rotate, rotate)
    sc = 1.0 + rng.uniform(-scale, scale)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, sc)
    M[0, 1] += np.tan(np.deg2rad(rng.uniform(-shear, shear)))
    M[1, 2] += rng.uniform(-shift, shift) * h

    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=PAGE)


def elastic(image, rng, alpha=18.0, sigma=7.0):
    """
    Elastic distortion (Simard et al., 2003): displace every pixel by a
    smoothed random field, so strokes bend as handwriting naturally varies.

    sigma controls smoothness and alpha the magnitude. A small sigma with a
    large alpha tears the image apart; the values here bend strokes without
    separating a mark from the letter it belongs to.
    """
    h, w = image.shape
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    return cv2.remap(image, x + dx, y + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=PAGE)


def morphology(image, rng, max_kernel=2):
    """
    Erosion thins the ink, as though it had faded; dilation thickens it, as
    though it had bled into the paper. Both are real conditions in this corpus.

    Note the inversion: OpenCV's erode shrinks BRIGHT regions, and here the ink
    is dark on a light page, so eroding the array thickens the writing. The
    comments name the effect on the ink, not on the array.

    The kernel is capped at 2 pixels because thinning applied hard removes
    small marks altogether.
    """
    k = int(rng.integers(1, max_kernel + 1))
    kernel = np.ones((k, k), np.uint8)
    if rng.random() < 0.5:
        return cv2.dilate(image, kernel, iterations=1)   # thins the ink
    return cv2.erode(image, kernel, iterations=1)        # thickens the ink


def brightness_contrast(image, rng, brightness=0.25, contrast=0.25):
    """Vary exposure and contrast, as scanning conditions do across archives."""
    a = 1.0 + rng.uniform(-contrast, contrast)
    b = rng.uniform(-brightness, brightness) * 255
    return np.clip(image.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def blur(image, rng, max_kernel=3):
    """
    Mild Gaussian blur. Capped at 3 pixels: beyond that a diacritic dissolves
    into its base letter and the image no longer matches its transcription.
    """
    return cv2.GaussianBlur(image, (3, 3), 0)


def gamma(image, rng, low=0.8, high=1.2):
    """Non-linear tone shift - darkens or lightens midtones without clipping."""
    g = rng.uniform(low, high)
    table = np.array([(i / 255.0) ** g * 255 for i in range(256)], np.uint8)
    return cv2.LUT(image, table)


# ---------------------------------------------------------------------------
# pipelines
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Applies a list of (operation, probability) pairs in order.

    Matches the interface the dataset expects: called with image=array,
    returns a dict with an "image" key.
    """

    def __init__(self, ops, name="pipeline", seed=None):
        self.ops = ops
        self.name = name
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        out = image
        for op, p in self.ops:
            if self.rng.random() < p:
                out = op(out, self.rng)
        return {"image": out}


def geometric(seed=None):
    """For the CRNN: distorts shape, leaves tone largely alone."""
    return Pipeline([
        (affine, 0.8),
        (elastic, 0.3),
        (brightness_contrast, 0.4),
    ], "geometric", seed)


def morphological(seed=None):
    """For Kraken: varies stroke thickness rather than shape."""
    return Pipeline([
        (morphology, 0.5),
        (lambda im, r: affine(im, r, rotate=1.5, shear=2, scale=0.03), 0.5),
        (brightness_contrast, 0.4),
    ], "morphological", seed)


def photometric(seed=None):
    """For TrOCR: varies tone and sharpness, so it must rely on context."""
    return Pipeline([
        (lambda im, r: brightness_contrast(im, r, 0.3, 0.3), 0.7),
        (blur, 0.3),
        (gamma, 0.3),
        (lambda im, r: affine(im, r, rotate=1, shear=2, scale=0.02), 0.3),
    ], "photometric", seed)


def combined(seed=None):
    """
    All three families together, for training a single model on its own rather
    than as an ensemble member. This is what the baseline reproduction uses:
    the aim there is the lowest error one CRNN can reach, not diversity from
    anything else.
    """
    return Pipeline([
        (affine, 0.7),
        (elastic, 0.25),
        (morphology, 0.3),
        (brightness_contrast, 0.5),
        (blur, 0.15),
    ], "combined", seed)


PIPELINES = {
    "none": lambda seed=None: None,
    "geometric": geometric,
    "morphological": morphological,
    "photometric": photometric,
    "combined": combined,
}


def get(name, seed=None):
    if name not in PIPELINES:
        raise ValueError(f"unknown augmentation {name!r}; "
                         f"choose from {sorted(PIPELINES)}")
    return PIPELINES[name](seed)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A synthetic line: dark strokes with small isolated marks above them,
    # standing in for diacritics. The check that matters is that those marks
    # survive - an augmentation that erased them would silently destroy the
    # very signal this project measures.
    image = np.full((64, 400), 240, np.uint8)
    for x in range(20, 380, 30):
        cv2.line(image, (x, 30), (x + 18, 48), 30, 3)      # base strokes
        cv2.circle(image, (x + 9, 14), 3, 20, -1)          # diacritic marks

    def marks_present(img):
        """Count dark blobs in the upper band, where the marks sit."""
        top = img[:24, :]
        _, binary = cv2.threshold(top, 140, 255, cv2.THRESH_BINARY_INV)
        n, _ = cv2.connectedComponents(binary)
        return n - 1

    baseline = marks_present(image)
    print(f"synthetic line: {baseline} diacritic marks\n")

    for name in PIPELINES:
        pipeline = get(name, seed=0)
        if pipeline is None:
            print(f"{name:<15} (no augmentation)")
            continue

        kept, shapes = [], set()
        for _ in range(200):
            out = pipeline(image=image)["image"]
            shapes.add((out.shape, out.dtype))
            kept.append(marks_present(out))

        assert shapes == {(image.shape, image.dtype)}, \
            f"{name} changed shape or dtype"
        retention = float(np.mean(kept)) / baseline
        print(f"{name:<15} ok   diacritic retention {retention*100:5.1f}%   "
              f"mean {np.mean(kept):.1f}/{baseline} marks")
        assert retention > 0.75, \
            f"{name} destroys too many marks - loosen the limits"

    print("\nall pipelines preserve shape, dtype and diacritics")
