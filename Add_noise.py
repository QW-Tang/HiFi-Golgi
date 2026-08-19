"""
Add isolated, irregularly shaped spot-like noise (simulating precipitation) to neuronal Tiff images (8-bit).

Each "precipitate" is formed by aggregating multiple randomly distributed small Gaussian blobs, resulting in a naturally irregular shape

Usage:
    python Add_noise.py input.tif output.tif --num 50 --min_radius 2 --max_radius 6 --min_sub 3 --max_sub 8
"""

import numpy as np
import tifffile
import argparse
from scipy.ndimage import gaussian_filter
import sys


def generate_blob_bbox(shape, center, radius, num_subspots, min_sigma, max_sigma, peak):
    """
    Generate an irregular blob, computing only within its bounding box to save memory.

    Parameters:
        shape: Full image dimensions (Z, Y, X)
        center: Blob center coordinates (cz, cy, cx)
        radius: Maximum spread radius for sub-spot centers
        num_subspots: Number of sub-spots
        min_sigma, max_sigma: Sigma range for sub-spots
        peak: Normalized peak intensity of the blob

    Returns:
        blob: Array of same shape as input (non-zero only inside bounding box)
        eff_radius: Effective radius of the blob (for overlap detection)
    """
    z, y, x = shape
    cz, cy, cx = center

    # Maximum possible sigma of sub-spots
    max_sig = max_sigma
    # Bounding box half-width: radius + 3*max_sig (Gaussian tail truncation)
    margin = int(np.ceil(radius + 3 * max_sig))
    # Bounding box in global coordinates
    z_min = max(0, int(cz) - margin)
    z_max = min(z, int(cz) + margin + 1)
    y_min = max(0, int(cy) - margin)
    y_max = min(y, int(cy) + margin + 1)
    x_min = max(0, int(cx) - margin)
    x_max = min(x, int(cx) + margin + 1)

    # Local volume dimensions
    local_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    local = np.zeros(local_shape, dtype=np.float32)

    # Local coordinate grids
    Z, Y, X = np.ogrid[0:local_shape[0], 0:local_shape[1], 0:local_shape[2]]
    # Center position in local coordinates
    local_cz = cz - z_min
    local_cy = cy - y_min
    local_cx = cx - x_min

    # Generate sub-spots
    for _ in range(num_subspots):
        # Random offset within a sphere of radius (uniform volume sampling)
        # Method: random direction + cube‑root scaling of radius for uniform volume
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        r = radius * np.random.rand() ** (1 / 3)  # uniform volume
        offset = direction * r

        sub_center = (local_cz + offset[0], local_cy + offset[1], local_cx + offset[2])
        sigma = np.random.uniform(min_sigma, max_sigma)

        # Gaussian spot
        dist_sq = ((Z - sub_center[0]) / sigma) ** 2 + ((Y - sub_center[1]) / sigma) ** 2 + (
                    (X - sub_center[2]) / sigma) ** 2
        spot = np.exp(-0.5 * dist_sq)
        local += spot

    # Normalize to peak value
    if local.max() > 0:
        local = (local / local.max()) * peak

    # Place local blob into a full‑size zero array
    blob = np.zeros(shape, dtype=np.float32)
    blob[z_min:z_max, y_min:y_max, x_min:x_max] = local

    # Effective radius for overlap detection
    eff_radius = radius + 3 * max_sigma
    return blob, eff_radius


def add_irregular_blobs(image, num_blobs, min_radius, max_radius, min_sigma, max_sigma,
                        min_subspots, max_subspots, peak_range, seed=None):
    """
    Add isolated irregular blob noise to image.

    Parameters:
        image: 3D or 2D uint8 array
        num_blobs: Number of blobs
        min_radius, max_radius: Spread radius range for each blob
        min_sigma, max_sigma: Sigma range for sub‑spots
        min_subspots, max_subspots: Number of sub‑spots per blob (range)
        peak_range: (min_peak, max_peak) peak intensity range for blobs
        seed: Random seed

    Returns:
        Noisy image as uint8
    """
    if seed is not None:
        np.random.seed(seed)

    was_2d = False
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
        was_2d = True

    result = image.astype(np.float32)
    shape = result.shape  # (Z, Y, X)

    centers = []  # blob centers
    eff_radii = []  # effective radius

    for i in range(num_blobs):
        attempts = 0
        max_attempts = 1000
        placed = False
        while attempts < max_attempts:
            # Random center
            cz = np.random.randint(0, shape[0])
            cy = np.random.randint(0, shape[1])
            cx = np.random.randint(0, shape[2])
            center = (cz, cy, cx)

            # Random blob parameters
            radius = np.random.uniform(min_radius, max_radius)
            num_sub = np.random.randint(min_subspots, max_subspots + 1)
            sigma = (min_sigma, max_sigma)  # range; actual sampling inside generator
            peak = np.random.uniform(*peak_range)

            # Effective radius for this blob
            eff_r = radius + 3 * max_sigma

            # Check overlap with existing blobs
            overlap = False
            for (ex_cz, ex_cy, ex_cx), ex_r in zip(centers, eff_radii):
                dist = np.sqrt((cz - ex_cz) ** 2 + (cy - ex_cy) ** 2 + (cx - ex_cx) ** 2)
                if dist < (eff_r + ex_r) * 0.9:  # 0.9 factor for conservative separation
                    overlap = True
                    break
            if not overlap:
                # Generate blob
                blob, eff_r_calculated = generate_blob_bbox(
                    shape, center, radius, num_sub, min_sigma, max_sigma, peak
                )
                # Add to result
                result = np.clip(result + blob, 0, 255)
                centers.append(center)
                eff_radii.append(eff_r_calculated)
                placed = True
                break
            attempts += 1

        if not placed:
            print(f"Warning: blob {i + 1} could not be placed (image may be too crowded), skipped.", file=sys.stderr)

    result = result.astype(np.uint8)
    if was_2d:
        result = result[0]
    return result


def main():
    parser = argparse.ArgumentParser(description="Add irregular isolated spot‑like noise (precipitation simulation)")
    parser.add_argument("input", help="Input Tiff file path (2D or 3D)")
    parser.add_argument("output", help="Output Tiff file path")
    parser.add_argument("--num", type=int, default=200, help="Number of blobs")
    parser.add_argument("--min_radius", type=float, default=1.0, help="Minimum spread radius of blobs")
    parser.add_argument("--max_radius", type=float, default=4.0, help="Maximum spread radius of blobs")
    parser.add_argument("--min_sigma", type=float, default=0.5, help="Minimum sigma of sub‑spots")
    parser.add_argument("--max_sigma", type=float, default=1.2, help="Maximum sigma of sub‑spots")
    parser.add_argument("--min_sub", type=int, default=3, help="Minimum number of sub‑spots per blob")
    parser.add_argument("--max_sub", type=int, default=8, help="Maximum number of sub‑spots per blob")
    parser.add_argument("--min_peak", type=float, default=40, help="Minimum peak intensity of blobs")
    parser.add_argument("--max_peak", type=float, default=160, help="Maximum peak intensity of blobs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Read image
    try:
        img = tifffile.imread(args.input)
    except FileNotFoundError:
        print(f"Error: input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    if img.dtype != np.uint8:
        print("Warning: input image is not 8‑bit, will convert to uint8 (0‑255)", file=sys.stderr)
        if img.max() > 255 or img.min() < 0:
            img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

    noisy = add_irregular_blobs(
        img,
        num_blobs=args.num,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        min_subspots=args.min_sub,
        max_subspots=args.max_sub,
        peak_range=(args.min_peak, args.max_peak),
        seed=args.seed
    )

    tifffile.imwrite(args.output, noisy, photometric='minisblack')
    print(f"Generated image with {args.num} irregular blobs: {args.output}")


if __name__ == "__main__":
    main()