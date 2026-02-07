"""
Script pour télécharger le dataset TREC-COVID depuis BEIR
"""
import os
from beir import util, datasets
from beir.datasets.data_loader import GenericDataLoader

def download_treccovid():
    """
    Télécharge le dataset TREC-COVID depuis BEIR et le sauvegarde dans le dossier data/
    """
    # Chemin vers le dossier data
    data_dir = "data"
    dataset_name = "trec-covid"
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs(data_dir, exist_ok=True)
    
    # Chemin complet pour le dataset
    dataset_path = os.path.join(data_dir, dataset_name)
    
    print(f"Téléchargement du dataset TREC-COVID depuis BEIR...")
    print(f"Le dataset sera sauvegardé dans: {dataset_path}")
    
    # Télécharger et extraire le dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
    out_dir = os.path.join(data_dir, dataset_name)
    
    # Télécharger le dataset
    data_path = util.download_and_unzip(url, out_dir)
    
    print(f"\nDataset téléchargé avec succès!")
    print(f"Chemin: {data_path}")
    
    # Charger les données pour vérifier
    print("\nVérification du chargement des données...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    print(f"\n✓ Corpus: {len(corpus)} documents")
    print(f"✓ Requêtes: {len(queries)} requêtes")
    print(f"✓ Qrels: {len(qrels)} jugements de pertinence")
    
    return data_path

if __name__ == "__main__":
    try:
        download_treccovid()
        print("\n✓ Téléchargement terminé avec succès!")
    except Exception as e:
        print(f"\n✗ Erreur lors du téléchargement: {e}")
        print("\nAssurez-vous d'avoir installé BEIR:")
        print("pip install beir")

