# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility to build a vector database from local documents."""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import nltk
import yaml
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from docsassist.schema import RAGModelSettings

if TYPE_CHECKING:  # pragma: no cover
    from langchain.schema import Document


class DiyVectorStoreSettings(BaseModel):
    """Validation schema for vector DB settings."""

    sentence_transformer_model_name: str
    chunk_size: int
    chunk_overlap: int


DEFAULT_SETTINGS = DiyVectorStoreSettings(
    sentence_transformer_model_name="all-MiniLM-L6-v2",
    chunk_size=2000,
    chunk_overlap=1000,
)


def make_chunks(
    path_to_source_documents: Path, chunk_size: int, chunk_overlap: int
) -> List["Document"]:
    """Convert raw documents into document chunks."""

    def _format_metadata(docs: list["Document"]) -> None:
        https_string = re.compile(r".+(https://.+)$")
        for doc in docs:
            doc.metadata["source"] = (
                doc.metadata.get("source", "")
                .replace("|", "/")
                .replace(str(path_to_source_documents.resolve()), "")
            )
            doc.metadata["source"] = re.sub(
                r"datarobot_docs/en/(.+)\.txt",
                r"https://docs.datarobot.com/en/docs/\1.html",
                doc.metadata["source"],
            )
            try:
                doc.metadata["source"] = https_string.findall(doc.metadata["source"])[0]
            except Exception:
                pass

    loader = DirectoryLoader(str(path_to_source_documents.resolve()), glob="**/*.*")
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    data = loader.load()
    docs = splitter.split_documents(data)
    _format_metadata(docs)
    return docs


def process_zip_documents(
    path_to_docs_zip: Path, chunk_size: int, chunk_overlap: int
) -> List["Document"]:
    """Unzip documents to a temp dir and chunk."""

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(path_to_docs_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        return make_chunks(Path(temp_dir), chunk_size, chunk_overlap)


def make_vector_db(
    documents: List["Document"],
    embedding_model_name: str,
    embedding_model_output_dir: Path,
    vdb_output_dir: Path,
) -> Tuple[Path, Path]:
    """Build the vector DB and persist it to disk."""

    embedding_function = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        cache_folder=str(embedding_model_output_dir),
    )
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    db = FAISS.from_texts(texts, embedding_function, metadatas=metadatas)
    db.save_local(str(vdb_output_dir))
    return embedding_model_output_dir, vdb_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents for Guarded RAG")
    parser.add_argument("docs_path", type=str, help="Directory or zip with PDFs/Docx")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("deployment_diy_rag")),
        help="Destination directory (defaults to deployment_diy_rag)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_SETTINGS.chunk_size)
    parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_SETTINGS.chunk_overlap
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_SETTINGS.sentence_transformer_model_name,
    )

    args = parser.parse_args()

    docs_path = Path(args.docs_path)
    output_dir = Path(args.output_dir)
    vdb_dir = output_dir / "faiss_db"
    embedding_dir = output_dir / "sentencetransformers"

    output_dir.mkdir(parents=True, exist_ok=True)
    vdb_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    if docs_path.is_file() and zipfile.is_zipfile(docs_path):
        documents = process_zip_documents(
            docs_path, args.chunk_size, args.chunk_overlap
        )
    else:
        documents = make_chunks(docs_path, args.chunk_size, args.chunk_overlap)

    embedding_path, db_path = make_vector_db(
        documents=documents,
        embedding_model_name=args.embedding_model,
        embedding_model_output_dir=embedding_dir,
        vdb_output_dir=vdb_dir,
    )

    rag_settings = RAGModelSettings(
        embedding_model_name=args.embedding_model,
        max_retries=0,
        request_timeout=30,
        temperature=0.0,
        stuff_prompt="""\
            You are a helpful assistant, helping users answer questions about some document(s).

            You will be given extracts from the document(s) to help answer the question.

            Try to use information within the sources. Don't use citations.
            ----------------
            {context}""",
    )
    with open(output_dir / RAGModelSettings.filename(), "w") as f:
        yaml.safe_dump(rag_settings.model_dump(mode="json"), f, allow_unicode=True)

    print(f"Vector DB written to: {db_path}")
    print(f"Embeddings cached at: {embedding_path}")
    print(f"RAG settings written to: {output_dir / RAGModelSettings.filename()}")


if __name__ == "__main__":
    main()
