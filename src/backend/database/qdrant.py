from enum import Enum
from uuid import uuid4

import numpy as np
from qdrant_client import models
from qdrant_client import QdrantClient

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hashlib import sha3_512


from backend.data_fetching.data_fetcher import DataFetcher, ChunkLevel


class QdrantSchema(str, Enum):
    COLLECTION_NAME: str

class ScientificPapersMainSchema(str, Enum):
    COLLECTION_NAME = "scientific-papers"
    ARTICLE_ID = "article_id"
    SECTIONS = "sections"
    SECTION_NAMES = "section_names"
    ABSTRACT = "abstract_text"

    def __repr__(self):
        return self.value

class ScientificPapersChunksSchema(str, Enum):
    COLLECTION_NAME = "scientific-paper-chunks"
    ARTICLE_ID = "article_id"
    SECTION_NAME = "section_name"
    PARAGRAPH_ID = "paragraph_id"
    PARAGRAPH = "paragraph"

    def __repr__(self):
        return self.value

class ScientificPapersCollectionNames(str, Enum):
    MAIN = "scientific-papers"
    CHUNKS = "scientific-paper-chunks"

    def __repr__(self):
        return self.value

class HFDatasetFields(str, Enum):
    SECTIONS = "sections"
    SECTION_NAMES = "section_names"
    ARTICLE_ID = "article_id"
    ABSTRACT = "article_abstract"
    EMBEDDINGS = "embeddings"

    def __repr__(self):
        return self.value

class QdrantDatabase:
    def __init__(self, client: QdrantClient, model: SentenceTransformer, upload_batch_size=2, embedding_batch_size=32, chunk_level: ChunkLevel = ChunkLevel.SENTENCE):
        self.client = client
        self.model = model
        self._embedding_batch_size = embedding_batch_size
        self._data_fetcher = DataFetcher(chunk_level=chunk_level)
        self.chunk_level = chunk_level
        self._upload_batch_size = upload_batch_size

        self._setup_collections()

    def _setup_collections(self):
        if not self.client.collection_exists(ScientificPapersCollectionNames.MAIN):
            self.client.create_collection(
                collection_name=ScientificPapersCollectionNames.MAIN,
                vectors_config=models.VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE,
                        on_disk=True,
                ),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
            )
        if not self.client.collection_exists(ScientificPapersCollectionNames.CHUNKS):
            self.client.create_collection(
                collection_name=ScientificPapersCollectionNames.CHUNKS,
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
            )

    def reset_database(self):
        if self.client.collection_exists(ScientificPapersCollectionNames.MAIN):
            self.client.delete_collection(ScientificPapersCollectionNames.MAIN)
        if self.client.collection_exists(ScientificPapersCollectionNames.CHUNKS):
            self.client.delete_collection(ScientificPapersCollectionNames.CHUNKS)
        self._setup_collections()

    def upload_from_dataset(self, dataset: Dataset | list[dict], batch_size=2):
        data = self._data_fetcher.from_dataset(dataset)
        self._upload(data)

    def upload_from_pmcid(self, pmcid):
        data = self._data_fetcher.from_pmcid(pmcid)
        self._upload(data)

    def upload_from_pdf(self, file):
        data = self._data_fetcher.from_pdf(file)
        self._upload(data)

    def _prepare_batches(self, batch: list[dict]):
        unnest = lambda x: [j for i in x for j in i]
        # document embeddings for main collection
        if "embeddings" in batch[0]:
            article_embeddings_batch = [np.array(unnest(i[HFDatasetFields.EMBEDDINGS])) for i in batch]
            for doc_ind in range(len(batch)):
                batch[doc_ind].pop("embeddings")
        else:
            article_embeddings_batch = [
                self.model.encode(unnest(i[HFDatasetFields.SECTIONS]), device=str(self.model.device),
                                  batch_size=self._embedding_batch_size) for i in batch]
        document_vectors = [article_embeddings_batch[i].mean(axis=0) for i in range(len(batch))]

        # paragraph embeddings for chunk collection
        document_chunk_payload = [({
            ScientificPapersChunksSchema.ARTICLE_ID: article[HFDatasetFields.ARTICLE_ID],
            ScientificPapersChunksSchema.PARAGRAPH: paragraph,
            ScientificPapersChunksSchema.SECTION_NAME: section_name,
            ScientificPapersChunksSchema.PARAGRAPH_ID: paragraph_ind
        }) for article in batch for section_name, section in
            zip(article[HFDatasetFields.SECTION_NAMES], article[HFDatasetFields.SECTIONS])
            for paragraph_ind, paragraph in enumerate(section)]
        paragraph_embeddings = [paragraph_embeddings for article_paragraph_embeddings in
                                article_embeddings_batch for paragraph_embeddings in
                                article_paragraph_embeddings]

        ids = {
            ScientificPapersCollectionNames.MAIN: [
                int.from_bytes(sha3_512(i[HFDatasetFields.ARTICLE_ID].encode()).digest()[:8], 'little')
                for i in batch],
            ScientificPapersCollectionNames.CHUNKS: [str(uuid4()) for _ in
                                                     range(len(document_chunk_payload))],
        }

        main_upload = {"ids": ids[ScientificPapersCollectionNames.MAIN], "vectors": document_vectors, "payload": batch}
        chunk_upload = {"ids": ids[ScientificPapersCollectionNames.CHUNKS], "vectors": paragraph_embeddings, "payload": document_chunk_payload}

        return main_upload, chunk_upload

    def _upload(self, dataset: Dataset | list[dict]):
        def batched(iterable, n=100):
            from itertools import islice
            iterator = iter(iterable)
            while batch := list(islice(iterator, n)):
                yield batch
        with tqdm(total=len(dataset), desc=f"Upload data to Qdrant:{ScientificPapersCollectionNames.MAIN}: ") as pbar:
            for batch in batched(dataset, self._upload_batch_size):
                main_data, chunk_data = self._prepare_batches(batch)
                try:
                    self.client.upsert(collection_name=ScientificPapersCollectionNames.MAIN,
                                  points=models.Batch(
                                      ids=main_data["ids"],
                                      vectors=main_data["vectors"],
                                      payloads=main_data["payload"]
                                  ))

                    self.client.upsert(collection_name=ScientificPapersCollectionNames.CHUNKS,
                                  points=models.Batch(
                                      ids=chunk_data["ids"],
                                      vectors=chunk_data["vectors"],
                                      payloads=chunk_data["payload"]
                                  ))
                except Exception as e:
                    print(e)
                    pass
                pbar.update(self._upload_batch_size)

    def _bulk_upload(self, data: list[dict]):
        from qdrant_client import models
        self.client.update_collection(
            collection_name=ScientificPapersCollectionNames.MAIN,
            hnsw_config=models.HnswConfigDiff(m=0),
        )

        self.client.update_collection(
            collection_name=ScientificPapersCollectionNames.CHUNKS,
            hnsw_config=models.HnswConfigDiff(m=0),
        )

        self._upload(data)

        self.client.update_collection(
            collection_name=ScientificPapersCollectionNames.MAIN,
            hnsw_config=models.HnswConfigDiff(m=32),
        )

        self.client.update_collection(
            collection_name=ScientificPapersCollectionNames.CHUNKS,
            hnsw_config=models.HnswConfigDiff(m=32),
        )

    def _query_single(self, query: list[float] | np.ndarray, n_docs=1, n_paragraphs=1, highlight: bool = True):
        relevant_documents = self.client.query_points(ScientificPapersCollectionNames.MAIN,
                                                 query,
                                                 limit=n_docs,
                                                 with_vectors=False)

        relevant_paragraphs_per_document = {}

        if highlight:
            document_ids = [point.payload[ScientificPapersMainSchema.ARTICLE_ID] for point in relevant_documents.points]
            relevant_paragraphs_per_document = {}

            for document_id in document_ids:
                document_filter = models.Filter(
                    must=[
                        models.FieldCondition(key=ScientificPapersChunksSchema.ARTICLE_ID, match=models.MatchValue(value=document_id))
                    ]
                )
                relevant_paragraphs_ids_resp = self.client.query_points(ScientificPapersCollectionNames.CHUNKS,
                                                                   query,
                                                                   limit=n_paragraphs,
                                                                   with_payload=[
                                                                       ScientificPapersChunksSchema.PARAGRAPH_ID,
                                                                       ScientificPapersChunksSchema.SECTION_NAME,
                                                                       ScientificPapersChunksSchema.PARAGRAPH
                                                                   ],
                                                                   query_filter=document_filter)
                # relevant_paragraphs_ids = [point.payload["paragraph_index"] for point in relevant_paragraphs_ids_resp.points]
                relevant_paragraphs_per_document[document_id] = relevant_paragraphs_ids_resp

        return relevant_documents, relevant_paragraphs_per_document

    def query(self, queries: list[str] | str, docs_per_query=1, highlight=True, paragraphs_per_document=1):
        if isinstance(queries, str):
            queries = [queries]
        query_embeddings = self.model.encode(queries, batch_size=self._embedding_batch_size)

        responses = {}

        for query, query_embedding in zip(queries, query_embeddings):
            response = self._query_single(query_embedding,
                                          n_docs=docs_per_query,
                                          n_paragraphs=paragraphs_per_document,
                                          highlight=highlight)
            responses[query] = response

        return responses
