from functools import partial

import numpy as np
from datasets import load_dataset, Dataset, NamedSplit, load_from_disk, concatenate_datasets
from huggingface_hub import snapshot_download

from pathlib import Path
import json

from typing import Any

from sentence_transformers import SentenceTransformer


def embed_sections(sections_batch: list[list[list[str]]], model, batch_size=32, device="cuda"):
    sections_per_batch = [len(sections_batch[i]) for i in range(len(sections_batch))]
    sections_per_batch = [0] + np.cumsum(sections_per_batch).tolist()
    paragraph_lens = [len(section) for sections in sections_batch for section in sections]
    paragraphs = [paragraph for sections in sections_batch for section in sections for paragraph in section]
    section_span = [0] + np.cumsum(paragraph_lens).tolist()
    paragraph_embeddings = model.encode(paragraphs, batch_size=batch_size, device=device)

    section_embedding_batch = []

    for i, section_num in enumerate(sections_per_batch[1:]):
        prev_section = sections_per_batch[i]
        curr_section = []
        for j, paragraph_num in enumerate(section_span[prev_section + 1: section_num + 1]):
            prev_paragraph = section_span[prev_section + j]
            curr_paragraph = paragraph_embeddings[prev_paragraph: paragraph_num]
            curr_section.append(curr_paragraph)
        section_embedding_batch.append(curr_section)

    return section_embedding_batch

def create_dataset(model:SentenceTransformer=None):
    def _get_examples(file) -> Any:
        with open(file, 'r') as f_r:
            for line in f_r:
                yield json.loads(line)

    get_examples_train = partial(_get_examples, "../../data/armac_scientific_papers/pubmed-dataset/train.txt")
    get_examples_val = partial(_get_examples, "../../data/armac_scientific_papers/pubmed-dataset/val.txt")
    get_examples_test = partial(_get_examples, "../../data/armac_scientific_papers/pubmed-dataset/test.txt")

    dataset_train = Dataset.from_generator(get_examples_train, num_proc=1, split=NamedSplit("train"))
    dataset_val = Dataset.from_generator(get_examples_val, num_proc=1, split=NamedSplit("val"))

    dataset_test = Dataset.from_generator(get_examples_test, num_proc=1, split=NamedSplit("test"))
    dataset = concatenate_datasets([dataset_train, dataset_test, dataset_val])

    dataset = dataset.map(lambda x: {"sections": [[i.replace("<S>", "").replace("</S>", "") for i in x["abstract_text"]]] + [[j.replace("<S>", "").replace("</S>", "") for j in i if j] for i in x["sections"] if i]})
    dataset = dataset.map(lambda x: {"section_names": ["Abstract"] + x["section_names"]})

    if model is not None:
        dataset = dataset.map(lambda x: {"embeddings": embed_sections(x["sections"], model=model, batch_size=64)}, batched=True, batch_size=8)

    dataset.save_to_disk("../../data/armac_scientific_papers/dataset", max_shard_size="100MB")

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", device="cuda", model_kwargs={"torch_dtype": "bfloat16"})
    create_dataset(model=model)
