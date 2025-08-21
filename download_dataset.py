import kagglehub  # type: ignore
import shutil
from pathlib import Path

def download_movielens_dataset():
    """Download the MovieLens 20M dataset and organize it in our project structure."""
    
    print("Downloading MovieLens 20M dataset...")
    
    # Download latest version
    path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
    
    print("Path to dataset files:", path)
    
    # Create our data directory structure
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to our project structure
    source_path = Path(path)
    
    print(f"\nCopying files from {source_path} to {raw_dir}")
    
    # List and copy all files
    for file_path in source_path.glob("*"):
        if file_path.is_file():
            destination = raw_dir / file_path.name
            shutil.copy2(file_path, destination)
            print(f"Copied: {file_path.name}")
    
    print(f"\nDataset successfully downloaded and organized in: {raw_dir}")
    
    # List the files we have
    print("\nDataset files:")
    for file_path in raw_dir.glob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {file_path.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    download_movielens_dataset()
