# HDFS and BGL Dataset
# Source: https://zenodo.org/records/8196385


import requests
import os
import zipfile
import tarfile

def download_and_extract(url, data_dir='data'):
    """
    Download a file, check if it's compressed, and extract if needed.

    Args:
        url: URL to download from
        data_dir: Directory to save and extract files (default: 'data')
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Extract filename from URL
    filename = url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    
    # Download the file
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error for bad status codes
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end='')
    
    print(f"\n✓ Downloaded to {filepath}")
    
    # Check if it's a zip file and extract
    if zipfile.is_zipfile(filepath):
        print(f"Extracting ZIP archive...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"✓ Extracted to {data_dir}/")
        
    # Check if it's a tar.gz file and extract
    elif tarfile.is_tarfile(filepath):
        print(f"Extracting TAR archive...")
        with tarfile.open(filepath, 'r:*') as tar_ref:
            tar_ref.extractall(data_dir)
        print(f"✓ Extracted to {data_dir}/")
    
    else:
        print(f"File is not compressed, keeping as is.")
    
    return filepath

# Example usage
if __name__ == "__main__":
    # Download BGL dataset
    # bgl_url = "https://zenodo.org/records/8196385/files/BGL.zip"
    # download_and_extract(bgl_url, data_dir='data')

    # hdfs_url = "https://zenodo.org/records/8196385/files/HDFS_v1.zip"
    # download_and_extract(hdfs_url, data_dir='data/hdfs')

    # No-op to ensure this block is not empty when usage is commented out
    pass