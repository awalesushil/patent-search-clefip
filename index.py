""" Patent index class """

import os
import gc
import json

from typing import List

import torch
import numpy as np
from torch.utils.data import DataLoader

import faiss
from wasabi import msg
from tqdm import tqdm

from clip_encoder import CLIPEncoder
from utils import save_json, load_json

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class PatentIndex:
    """Class for patent index"""

    def __init__(self,
                 config: dict, name: str = None):
        self.config = config
        self.dataset = self.config["dataset"]
        self.name = name
        self.index_name = self._get_name()
        self.faiss_index = None
        self.idx2patent = {}
        self.patent2idx = {}
        self.idx2image = {}
        self.image2idx = {}
        self.patent_count = None
        self.normalized = False
        self.metadata = None
        self.model = CLIPEncoder("ViT_B_32")

    def _get_name(self):
        """Set the name of the index"""
        index_name = self.config["dataset"]
        index_name += f"_{self.config['index_type']}"
        index_name += "_CUDA" if torch.cuda.is_available() else "_CPU"
        index_name += "_normalized" if self.config["normalize"] else ""
        index_name += self.name if self.name else ""
        return index_name

    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize the embeddings"""
        return embeddings / np.linalg.norm(embeddings)

    def create(self) -> "PatentIndex":
        """Create the index"""
        idx_type, dim, norm = self.config["index_type"], self.config["dimension"], self.config["normalize"]
        if idx_type == "IP":
            self.faiss_index = faiss.IndexFlatIP(dim)
        elif idx_type == "L2":
            self.faiss_index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unknown index type {idx_type}")
        self.normalized = norm

        msg.good(f"Created {self.index_name} {self.__class__.__name__}")

        return self

    def index(self, dataloader: DataLoader, normalize: str) -> "PatentIndex":
        """Add embeddings to the index"""

        idx_name = self.__class__.__name__

        patent_index = 0
        image_index = 0

        for batch in tqdm(dataloader, desc="Indexing"):

            images, filenames = batch[0][0], batch[1]

            for filename in filenames:
                self.idx2image[image_index] = filename[0]
                self.image2idx[filename[0]] = image_index
                image_index += 1

            patent_ids = self.get_patent_ids(filenames)

            for patent_id in patent_ids:
                self.patent2idx.setdefault(patent_id, []).append(patent_index)
                patent_index += 1

            embeddings = self.model.encode(images)
            if normalize == "numpy":
                embeddings = [self.normalize(embedding) for embedding in embeddings]
                embeddings = np.array(embeddings)
            elif normalize == "faiss":
                embeddings = np.array(embeddings, dtype="float32")
                faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)

            torch.cuda.empty_cache()
            gc.collect()

        msg.good(f"Indexed {self.faiss_index.ntotal} figures to {self.index_name} {idx_name}")

        self.patent_count = len(self.patent2idx)

        self.idx2patent = {idx: patent_id for patent_id, idxs in self.patent2idx.items() for idx in idxs}

        return self

    def get_patent_ids(self, filenames: List[str]) -> str:
        """
            Get patent id from filename
        """
        if self.dataset == "EPO":
            return ["".join(filename[0].split("_")[:-2]) for filename in filenames]

        if self.dataset == "CLEF_IP":
            patent_ids = []
            for filename in filenames:
                filename = filename[0]
                if filename.startswith("EP"):
                    if len(filename.split("-")) > 4:
                        patent_ids.append("-".join(filename.split("-")[:-2]))
                    else:
                        patent_ids.append("EP"+filename[7:14])
                elif filename.startswith("US"):
                    patent_ids.append("US"+filename.split("-")[0].split("US")[1])
                elif filename.startswith("WO"):
                    patent_ids.append("WO"+filename.split("img")[0][-10:])
            return patent_ids

        raise ValueError(f"Unknown dataset {self.dataset}")

    def save(self) -> "PatentIndex":
        """Save the index"""

        os.makedirs("index", exist_ok=True)
        os.makedirs(f"index/{self.index_name}", exist_ok=True)

        idx_name = self.__class__.__name__
        faiss.write_index(self.faiss_index,
                          f"index/{self.index_name}/{idx_name}.index")

        jsons_to_save = ["idx2patent", "patent2idx", "idx2image", "image2idx"]
        for json_to_save in jsons_to_save:
            save_json(getattr(self, json_to_save),
                      f"index/{self.index_name}/{idx_name}_{json_to_save}.json")

        msg.good(f"Saved index {self.index_name} {idx_name}")

        return self

    def load(self, with_metadata: bool = False):
        """Load the index"""

        idx_name = self.__class__.__name__
        self.faiss_index = faiss.read_index(
                f"index/{self.index_name}/{idx_name}.index"
            )

        jsons_to_load = ["idx2patent", "patent2idx", "idx2image", "image2idx"]

        for json_to_load in jsons_to_load:
            setattr(self, json_to_load,
                    load_json(f"index/{self.index_name}/{idx_name}_{json_to_load}.json"))

        self.patent_count = len(self.patent2idx)

        if with_metadata:
            with open(f"index/{self.index_name}/{idx_name}_metadata.json", "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

        msg.good(f"Loaded index {self.index_name} {idx_name}")
        print("*"*5, "Metadata", "*"*5)
        msg.good(f"Index size: {self.faiss_index.ntotal}")
        msg.good(f"Patent count: {self.patent_count}")

        return self

    def set_metadata(self, metadata: dict) -> "PatentIndex":
        """Set the metadata"""
        self.metadata = metadata
        class_name = self.__class__.__name__
        metadata_path = f"index/{self.index_name}/{class_name}_metadata.json"

        with open(f"{metadata_path}", "w", encoding="utf-8") as file:
            json.dump(self.metadata, file)

        msg.good(f"Set metadata for index {self.index_name}")

        return self

    def extract(self, patent_id: str) -> np.ndarray:
        """Extract the embeddings for a patent"""
        idxs = self.patent2idx[patent_id]
        return [self.faiss_index.reconstruct(idx) for idx in idxs]

    def iterate_per_patent(self) -> tuple:
        """Iterate over the index"""
        for patent_id in self.patent2idx.keys():
            yield (patent_id, self.extract(patent_id))

    def iterate_per_embedding(self) -> tuple:
        """Iterate over the index"""
        for idx in range(self.faiss_index.ntotal):
            yield (idx, self.faiss_index.reconstruct_n(idx, 1))

class BaseIndex(PatentIndex):
    """Base index class"""

    def search(self, embeddings: np.ndarray, patent_id: str) -> List[str]:
        """Search the index"""

        use_gpu = False

        image_search_limit = 128

        if embeddings.shape[0] > image_search_limit:
            msg.warn(f"Image search limit exceeded for {patent_id}")

            distances = np.empty((embeddings.shape[0], self.faiss_index.ntotal), dtype="float32")
            indices = np.empty((embeddings.shape[0], self.faiss_index.ntotal), dtype="int64")

            for i in range(0, embeddings.shape[0], image_search_limit):
                batch_embeddings = embeddings[i:i+image_search_limit]
                batch_distances, batch_indices = self.faiss_index.search(batch_embeddings,
                                                                         k=self.faiss_index.ntotal)
                distances[i:i+image_search_limit] = batch_distances
                indices[i:i+image_search_limit] = batch_indices

            distances = torch.from_numpy(distances).cuda()
            indices = torch.from_numpy(indices).cuda()
            sorted_distances = torch.empty_like(distances).cuda()

            use_gpu = True
        else:

            distances, indices = self.faiss_index.search(embeddings, k=self.faiss_index.ntotal)
            sorted_distances = np.empty_like(distances)

        for i, idx in enumerate(indices):
            sorted_distances[i] = distances[i][idx.argsort()]

        return sorted_distances, use_gpu

class QueryIndex(PatentIndex):
    """Query index class"""
