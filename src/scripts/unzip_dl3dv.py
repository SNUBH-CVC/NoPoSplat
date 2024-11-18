import os
import zipfile
from pathlib import Path

# Specify the root directory
INPUT_IMAGE_DIR = Path(__file__).parents[2] / "datasets" / "DL3DV-ALL-480P"
subsets = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K']

for subset in subsets:
    zip_dir = INPUT_IMAGE_DIR / subset
    zip_files = list(zip_dir.glob("*.zip"))
    for zip_file in zip_files:
        print(zip_file)
        extract_path = zip_file.parent 
        # unzip 시 폴더가 생성되어 parent 경로로 지정
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted {zip_file} to {extract_path}")
        os.remove(zip_file)
