# Label PNG Generator

This folder contains `make_labels.py`, which renders printable ASCII characters to PNG files using ImageMagick.

## Requirements
- Python 3 with `tqdm` installed.
- ImageMagick `convert` available on your PATH.

Install ImageMagick on Debian/Ubuntu if needed:
```bash
sudo apt-get update && sudo apt-get install imagemagick
```

## Usage
```bash
python3 make_labels.py
```
Outputs are saved in this folder. To use a different font or point sizes, adjust the constants at the top of the script or pass parameters when calling the functions. 
