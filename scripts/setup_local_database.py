from datasets import load_from_disk
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from backend.database.qdrant import QdrantDatabase, ScientificPapersCollectionNames

if __name__ == '__main__':
    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", device="cuda")
    qdrant_database = QdrantDatabase(client, model, upload_batch_size=16)
    #qdrant_database.reset_database()

    sample = 1000
    #dataset_sample = load_from_disk('../../data/armac_scientific_papers/dataset')
    #dataset_sample = dataset_sample.select(range(sample))
    #dataset_sample = dataset_sample.map(lambda x: {"sections": [[j.replace('</S>', '') for j in i] for i in x["sections"]]})
    #qdrant_database._bulk_upload(dataset_sample)

    qdrant_database.client.create_snapshot(ScientificPapersCollectionNames.MAIN)
    qdrant_database.client.create_snapshot(ScientificPapersCollectionNames.CHUNKS)
