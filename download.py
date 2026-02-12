"""
Script to download the TREC-COVID dataset from BEIR
"""
import os
from beir import util, datasets
from beir.datasets.data_loader import GenericDataLoader

def download_treccovid():
    """
    Downloads the TREC-COVID dataset from BEIR and saves it in the data/ folder
    """
    # Path to the data folder
    data_dir = "data"
    dataset_name = "trec-covid"
    
    # Create the data folder if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Full path for the dataset
    dataset_path = os.path.join(data_dir, dataset_name)
    
    print(f"Downloading TREC-COVID dataset from BEIR...")
    print(f"The dataset will be saved in: {dataset_path}")
    
    # Download and extract the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
    out_dir = os.path.join(data_dir, dataset_name)
    
    # Download the dataset
    data_path = util.download_and_unzip(url, out_dir)
    
    print(f"\nDataset downloaded successfully!")
    print(f"Path: {data_path}")
    
    # Load the data to verify
    print("\nVerifying data loading...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    print(f"\n✓ Corpus: {len(corpus)} documents")
    print(f"✓ Queries: {len(queries)} queries")
    print(f"✓ Qrels: {len(qrels)} relevance judgments")
    
    return data_path

if __name__ == "__main__":
    try:
        download_treccovid()
        print("\n✓ Download completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        print("\nMake sure you have installed BEIR:")
        print("pip install beir")

