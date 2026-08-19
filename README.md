# HiFi Golgi Toolkit

This repository contains four Python scripts for neuronal image simulation, noise addition, morphological parameter calculation, and reconstruction comparison. All scripts are designed for common neuroscience data formats (TIFF images and SWC morphology files) and support 2D/3D image processing and morphometric analysis.

---

## Dependencies

Python 3.7+ is recommended. Install the required packages via `pip`:

```bash
pip install numpy scipy tifffile pandas
```


---

## Script Descriptions

### 1. `Simulation.py` – Generate Continuous Slice Images from SWC

**Function**: Reads a SWC neuron morphology file, generates a series of simulated slice images (TIFF) along the Z‑axis in 3D space. Supports adjustable grayscale staining, radius scaling, noise, and blurring.

**Main parameters** (configured in the `USER CONFIGURATION AREA` at the end of the script):

| Parameter | Description |
|-----------|-------------|
| `SWC_FILE` | Input SWC file path |
| `OUTPUT_DIR` | Output directory |
| `PIXEL_SIZE` | XY pixel size (micrometers) |
| `SLICE_INTERVAL` | Z‑slice spacing (micrometers) |
| `BG_GRAY` | Background grayscale (before inversion, 0–255) |
| `SOMA_GRAY` | Soma grayscale (before inversion) |
| `DENDRITE_GRAY` | Dendrite/process grayscale (before inversion) |
| `AXON_GRAY` | Axon grayscale (only in `type` mode) |
| `NOISE_STD` | Standard deviation of grayscale noise |
| `BLUR_SIGMA` | Gaussian blur sigma (0 = no blur) |
| `GRAY_MODE` | Grayscale assignment mode: `'auto'` (automatic), `'type'` (by node type), or `'radius'` (by radius) |
| `RADIUS_SCALE` | Global radius scaling factor (>1 makes processes thicker) |
| `INVERT` | Whether to invert grayscale (`True` → neurons bright, background dark) |

**Run** directly after editing the configuration:

```bash
python Simulation.py
```

---

### 2. `Add_noise.py` – Add Isolated Irregular Spot‑Like Noise

**Function**: Adds simulated "precipitate"‑like noise spots to 8‑bit neuronal TIFF images. Each spot is formed by aggregating multiple randomly distributed small Gaussian blobs, resulting in a naturally irregular shape.

**Command‑line usage**:

```bash
python Add_noise.py input.tif output.tif [options]
```

**Required arguments**:

| Argument | Description |
|----------|-------------|
| `input` | Input TIFF file path (2D or 3D) |
| `output` | Output TIFF file path |

**Optional arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num` | 200 | Number of blobs |
| `--min_radius` | 1.0 | Minimum spread radius of blobs |
| `--max_radius` | 4.0 | Maximum spread radius of blobs |
| `--min_sigma` | 0.5 | Minimum sigma of sub‑spots |
| `--max_sigma` | 1.2 | Maximum sigma of sub‑spots |
| `--min_sub` | 3 | Minimum number of sub‑spots per blob |
| `--max_sub` | 8 | Maximum number of sub‑spots per blob |
| `--min_peak` | 40 | Minimum peak intensity of blobs (0–255) |
| `--max_peak` | 160 | Maximum peak intensity of blobs |
| `--seed` | None | Random seed (for reproducibility) |

**Example**:

```bash
python Add_noise.py neuron.tif noisy_neuron.tif --num 50 --min_radius 2 --max_radius 6 --min_sub 3 --max_sub 8
```

---

### 3. `Calculate_counts_and_length.py` – Count Branches and Total Length

**Function**: Reads a SWC file, recursively computes the total number of branches (leaf nodes from the root) and the total neuronal path length (sum of Euclidean distances along all branches).

**Usage**: Modify the `file_path` variable inside the script to point to your SWC file, then run:

```python
file_path = 'path/to/your/neuron.swc'
```

**Output example**:

```
Number of branches: XXX
Total neuronal branch length: XXXX.XX
```

> **Note**: Assumes a single root node (`parent == -1`).

---

### 4. `Neuron_compare.py` – Compute Average Distance Between Two SWC Files

**Function**: Computes the bidirectional average nearest‑neighbour distance between two SWC reconstructions (e.g., raw vs. post‑processed), useful for evaluating reconstruction accuracy.

**Configuration**: Set the two file paths in the `FILE_PATHS` dictionary:

```python
FILE_PATHS = {
    'RAW': r"path/to/raw.swc",
    'POST': r"path/to/postprocessed.swc"
}
```

**Run**:

```bash
python Neuron_compare.py
```

**Output**:

```
Average Distance (RAW→POST): X.XXXX μm
Average Distance (POST→RAW): X.XXXX μm
```

---

## Input / Output Formats

- **TIFF images**: 2D or 3D single‑channel 8‑bit grayscale (non‑8‑bit images are automatically converted by `Add_noise.py`).
- **SWC files**: Standard SWC format with 7 space‑separated columns: `id type x y z radius parent` (lines starting with `#` are treated as comments).

---

## Important Notes

1. All scripts are command‑line or script‑based; no GUI is provided.
2. In `Simulation.py`, `gray_mode='auto'` automatically switches to `'radius'` if all nodes share the same type, otherwise uses `'type'`.
3. `Add_noise.py` includes overlap detection between blobs and attempts to reposition them to avoid excessive clustering.
4. `Calculate_counts_and_length.py` and `Neuron_compare.py` require `pandas` and `numpy` – ensure they are installed.
5. Use absolute or correct relative paths to avoid file‑not‑found errors.

---

## License

This project is intended for academic research use. Please modify and extend as needed.

---

## Author & Contact

For questions or suggestions, please contact the project author.
```
