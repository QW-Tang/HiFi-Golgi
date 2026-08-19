import numpy as np
from scipy.spatial import KDTree

# ====================== File Path Configuration ======================
FILE_PATHS = {
    'RAW': r"  ",
    'POST': r"  "
}

# ====================== SWC Reader ======================
def read_swc_coords(file_path):
    """Read an SWC file and return an array of node coordinates (N, 3)."""
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                coords.append([x, y, z])
            except ValueError:
                continue
    return np.array(coords)

# ====================== Main Program ======================
if __name__ == "__main__":
    # Read both coordinate sets
    coords1 = read_swc_coords(FILE_PATHS['RAW'])
    coords2 = read_swc_coords(FILE_PATHS['POST'])

    # Build KD-trees
    tree1 = KDTree(coords1)
    tree2 = KDTree(coords2)

    # Compute bidirectional average distances
    dist1_to_2, _ = tree2.query(coords1)   # RAW → POST
    dist2_to_1, _ = tree1.query(coords2)   # POST → RAW

    avg_dist_raw_to_mf = np.mean(dist1_to_2)
    avg_dist_mf_to_raw = np.mean(dist2_to_1)

    # Output results
    print(f"Average Distance (RAW→POST): {avg_dist_raw_to_mf:.4f} μm")
    print(f"Average Distance (POST→RAW): {avg_dist_mf_to_raw:.4f} μm")