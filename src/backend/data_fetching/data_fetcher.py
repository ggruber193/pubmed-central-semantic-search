from enum import Enum
from datasets import Dataset
from pathlib import Path

from src.backend.data_fetching.data_fields import DataFields

class ChunkLevel(Enum):
    SENTENCE = ("sentence", '. ')
    PARAGRAPH = ("paragraph", '\n')
    SECTION = ("section", '$$$$$$$$$$$$$$$$$$$$$$$$')

    def __repr__(self):
        return self.value[0]

    @property
    def sep(self):
        return self.value[1]


class ArticleChunker:
    def __init__(self, chunk_level: ChunkLevel = ChunkLevel.SENTENCE):
        self.chunk_level = chunk_level

    @property
    def sep(self):
        return self.chunk_level.sep

    def __call__(self, text):
        sep = self.chunk_level.sep
        return text.split(sep)

class DataFetcher:
    def __init__(self, chunk_level: ChunkLevel = ChunkLevel.SENTENCE):
        self._chunker = ArticleChunker(chunk_level)

    # function to make it easier using this specific dataset
    def from_hugging_face_scientific_papers_dataset(self, dataset: Dataset, used_separator="\n"):
        dataset = dataset.map(
            lambda x: {DataFields.SECTIONS: used_separator.join(x["article_abstract"]).split(self._chunker.sep) + used_separator.join(x[DataFields.SECTIONS]).split(self._chunker.sep)})
        dataset = dataset.map(lambda x: {DataFields.SECTION_NAMES: ["Abstract"] + x[DataFields.SECTION_NAMES]})
        return dataset

    def from_dataset(self, dataset: Dataset, used_separator=''):
        assert all(i in dataset.column_names for i in DataFields)
        dataset = dataset.map(lambda x: {DataFields.SECTIONS: used_separator.join(x[DataFields.SECTIONS]).split(self._chunker.sep)})
        return dataset

    def from_pmcid(self, pmcid: str | list[str]):
        if isinstance(pmcid, str):
            pmcid = [pmcid]
        from src.backend.data_fetching.fetch_pmcid import (fetch_from_pmcid)
        output = []
        for pmcid in pmcid:
            article_out = fetch_from_pmcid(pmcid)
            article_out[DataFields.SECTIONS] = [self._chunker(i) for i in article_out[DataFields.SECTIONS]]
            output.append(article_out)
        return output

    def from_pdf(self, file: str | Path):
        raise NotImplementedError()
        pass
