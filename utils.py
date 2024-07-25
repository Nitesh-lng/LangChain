
import logging
import pathlib
from typing import Any
from langchain_community.chat_models import ChatMistral
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader    
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader

def init_memory():
    
    return ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

MEMORY = init_memory()


class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], **unstructured_kwargs: Any):
        super().__init__(file_path, **unstructured_kwargs, mode="elements", strategy="fast")


class DocumentLoaderException(Exception):
    pass


class DocumentLoader(object):
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }


def load_document(temp_filepath: str) -> list[Document]:
    
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extensions.get(ext)
    if not loader:
        raise DocumentLoaderException(
            f"Invalid extension type {ext}, cannot load this type of file"
        )

    loaded = loader(temp_filepath)
    docs = loaded.load()
    logging.info(docs)
    return docs