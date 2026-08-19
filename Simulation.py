import numpy as np
import tifffile
from math import sqrt, ceil
import os
from collections import defaultdict

def read_swc(swc_file):
    """Read SWC file, return node list, edge list, and node dictionary."""
    nodes = []
    with open(swc_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            idx = int(parts[0])
            typ = int(parts[1])
            x, y, z, radius = map(float, parts[2:6])
            parent = int(parts[6])
            nodes.append([idx, typ, x, y, z, radius, parent])
    node_dict = {n[0]: n for n in nodes}
    edges = []
    for n in nodes:
        if n[6] != -1 and n[6] in node_dict:
            edges.append((n[6], n[0]))
    return nodes, edges, node_dict

def generate_slices(swc_file, output_dir, pixel_size=0.65, slice_interval=0.65,
                    bg_gray=240, soma_gray=50, dendrite_gray=80, axon_gray=70,
                    noise_std=8, blur_sigma=0.5,
                    gray_mode='auto', radius_scale=1.0, invert=True):
    """
    Generate continuous slice images.
    Parameters:
        swc_file      : Path to SWC file
        output_dir    : Output directory
        pixel_size    : Pixel size in XY plane (micrometers)
        slice_interval: Slice spacing (micrometers)
        bg_gray       : Background gray value (before inversion)
        soma_gray     : Soma gray value (before inversion, lower = darker)
        dendrite_gray : Dendrite/process gray value (before inversion)
        axon_gray     : Axon gray value (only in 'type' mode)
        noise_std     : Standard deviation of gray noise
        blur_sigma    : Gaussian blur sigma (0 = no blur)
        gray_mode     : 'auto' / 'type' / 'radius'
        radius_scale  : Global radius scaling (>1 makes processes thicker)
        invert        : True to invert grayscale (neurons bright, background dark); False to keep dark signal
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nodes, edges, node_dict = read_swc(swc_file)
    if not nodes:
        print("SWC file is empty or failed to read")
        return

    # Extract information and apply radius scaling
    coords = np.array([[n[2], n[3], n[4]] for n in nodes])
    radii = np.array([n[5] * radius_scale for n in nodes])
    types = np.array([n[1] for n in nodes])

    # Automatically select gray mode
    if gray_mode == 'auto':
        unique_types = np.unique(types)
        if len(unique_types) == 1:
            gray_mode = 'radius'
            print(f"Detected all nodes have type {unique_types[0]}, automatically switching to 'radius' mode (gray assigned based on radius)")
        else:
            gray_mode = 'type'
            print("Using 'type' mode (gray assigned by node type)")

    # Compute base gray values for nodes
    if gray_mode == 'type':
        type_to_gray = {1: soma_gray, 2: axon_gray, 3: dendrite_gray, 4: dendrite_gray}
        node_gray_base = np.array([type_to_gray.get(t, 90) for t in types], dtype=np.float32)
    else:  # 'radius' mode
        r_min, r_max = radii.min(), radii.max()
        if r_max - r_min < 1e-6:
            node_gray_base = np.full_like(radii, (soma_gray + dendrite_gray) / 2, dtype=np.float32)
        else:
            norm_r = (radii - r_min) / (r_max - r_min)
            low, high = soma_gray, dendrite_gray + 20
            node_gray_base = low + (1 - norm_r) * (high - low)
            node_gray_base = np.clip(node_gray_base, 20, 230)

    # Add random noise to simulate uneven staining
    np.random.seed(42)
    node_gray_base += np.random.normal(0, noise_std, size=len(nodes))
    node_gray_base = np.clip(node_gray_base, 20, 230).astype(np.uint8)

    # Additional fine-tuning by radius
    radius_factor = 1.0 - (radii / (radii.max() + 1e-6)) * 0.2
    node_gray = (node_gray_base * radius_factor).astype(np.uint8)

    # Compute bounding box (using scaled radii)
    min_xyz = coords.min(axis=0) - radii.max()
    max_xyz = coords.max(axis=0) + radii.max()
    min_x, min_y, min_z = min_xyz
    max_x, max_y, max_z = max_xyz

    width = int(ceil((max_x - min_x) / pixel_size)) + 1
    height = int(ceil((max_y - min_y) / pixel_size)) + 1
    print(f"Image dimensions: {width} x {height} pixels")
    print(f"Z range: {min_z:.2f} ~ {max_z:.2f} micrometers")

    def coord_to_pixel(x, y):
        px = int(round((x - min_x) / pixel_size))
        py = int(round((y - min_y) / pixel_size))
        px = max(0, min(width-1, px))
        py = max(0, min(height-1, py))
        return px, py

    # Sample points along segments (spacing = pixel_size/2)
    sample_points = []
    for i, n in enumerate(nodes):
        sample_points.append((n[2], n[3], n[4], radii[i], node_gray[i]))

    step = pixel_size / 2.0
    for parent_id, child_id in edges:
        p = node_dict[parent_id]
        c = node_dict[child_id]
        r1 = radii[parent_id-1]
        r2 = radii[child_id-1]
        x1, y1, z1 = p[2], p[3], p[4]
        x2, y2, z2 = c[2], c[3], c[4]
        dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        if dist < 1e-6:
            continue
        num_steps = max(1, int(ceil(dist / step)))
        for t in np.linspace(0, 1, num_steps, endpoint=False):
            if t == 0:
                continue
            x = x1 + (x2-x1)*t
            y = y1 + (y2-y1)*t
            z = z1 + (z2-z1)*t
            r = r1 + (r2-r1)*t
            g1 = node_gray[parent_id-1]
            g2 = node_gray[child_id-1]
            gray = int(int(g1) + (int(g2) - int(g1)) * t)
            sample_points.append((x, y, z, r, gray))

    z_positions = np.arange(min_z, max_z + slice_interval/2, slice_interval)
    num_slices = len(z_positions)
    print(f"Generating {num_slices} slices")

    for idx, z_center in enumerate(z_positions):
        print(f"Processing slice {idx+1}/{num_slices}, z={z_center:.2f}um")
        img = np.full((height, width), bg_gray, dtype=np.uint8)

        for (x, y, z, r, gray) in sample_points:
            dz = abs(z - z_center)
            if dz >= r:
                continue
            r_section = sqrt(r*r - dz*dz)
            if r_section < 0.01:
                continue
            r_pix = r_section / pixel_size
            if r_pix < 0.5:
                px, py = coord_to_pixel(x, y)
                if img[py, px] > gray:
                    img[py, px] = gray
                continue
            cx, cy = coord_to_pixel(x, y)
            r_int = int(ceil(r_pix))
            x_min = max(0, cx - r_int)
            x_max = min(width-1, cx + r_int)
            y_min = max(0, cy - r_int)
            y_max = min(height-1, cy + r_int)

            yy, xx = np.ogrid[y_min:y_max+1, x_min:x_max+1]
            dist2 = (xx - cx)**2 + (yy - cy)**2
            mask = dist2 <= (r_pix * r_pix)
            region = img[y_min:y_max+1, x_min:x_max+1]
            region[mask] = np.minimum(region[mask], gray)

        # Gaussian blur
        if blur_sigma > 0:
            from scipy.ndimage import gaussian_filter
            img_float = img.astype(np.float32)
            img_blur = gaussian_filter(img_float, sigma=blur_sigma)
            img = np.clip(img_blur, 0, 255).astype(np.uint8)

        # Invert grayscale
        if invert:
            img = 255 - img

        out_path = os.path.join(output_dir, f"slice_{idx+1:04d}.tif")
        tifffile.imwrite(out_path, img, photometric='minisblack')
        print(f"Saved {out_path}")

    print("All slices generated successfully!")


# ============================================================
#  ★★★ USER CONFIGURATION AREA ★★★
#  Modify the variables below as needed, then run this script.
# ============================================================
if __name__ == "__main__":
    # Required
    SWC_FILE = r"  "   # Change to your SWC file path
    OUTPUT_DIR = r"  "     # ← Change to your output directory

    # Optional parameters (adjust as needed)
    PIXEL_SIZE = 0.65  # XY pixel size (micrometers)
    SLICE_INTERVAL = 0.65  # Z slice interval (micrometers)
    BG_GRAY = 240  # Background gray (before inversion)
    SOMA_GRAY = 50  # Soma gray (before inversion, lower = darker)
    DENDRITE_GRAY = 80  # Process gray (before inversion)
    AXON_GRAY = 70  # Axon gray (only in 'type' mode)
    NOISE_STD = 8  # Standard deviation of gray noise
    BLUR_SIGMA = 0.8  # Gaussian blur sigma (0 = no blur)
    GRAY_MODE = 'auto'  # 'auto', 'type', or 'radius'
    RADIUS_SCALE = 3.0  # Radius scaling (>1 makes processes thicker)
    INVERT = True  # True → neurons bright, False → neurons dark

    # Call the generation function
    generate_slices(
        swc_file=SWC_FILE,
        output_dir=OUTPUT_DIR,
        pixel_size=PIXEL_SIZE,
        slice_interval=SLICE_INTERVAL,
        bg_gray=BG_GRAY,
        soma_gray=SOMA_GRAY,
        dendrite_gray=DENDRITE_GRAY,
        axon_gray=AXON_GRAY,
        noise_std=NOISE_STD,
        blur_sigma=BLUR_SIGMA,
        gray_mode=GRAY_MODE,
        radius_scale=RADIUS_SCALE,
        invert=INVERT
    )