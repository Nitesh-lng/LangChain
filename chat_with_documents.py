
import logging
import os
import tempfile

from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import  FlareChain
from langchain.chains import  OpenAIModerationChain
from langchain.chains import  SequentialChain
from langchain.chains.base import Chain

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch

from utils import MEMORY, load_document
from config import set_environment



logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER = logging.getLogger()
set_environment()


LLM = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, streaming=True
)


def configure_retriever(
        docs: list[Document],
        use_compression: bool = False
) -> BaseRetriever:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        },
    )
    if not use_compression:
        return retriever

    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.2
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever,
    )


def configure_chain(retriever: BaseRetriever, use_flare: bool = True) -> Chain:
   
    output_key = 'response' if use_flare else 'answer'
    MEMORY.output_key = output_key
    params = dict(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000,
    )
    if use_flare:
        return FlareChain.from_llm(
            **params
        )
    return ConversationalRetrievalChain.from_llm(
        **params
    )


def configure_retrieval_chain(
        uploaded_files,
        use_compression: bool = False,
        use_flare: bool = False,
        use_moderation: bool = False
) -> Chain:
    """Read documents, configure retriever, and the chain."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs=docs, use_compression=use_compression)
    chain = configure_chain(retriever=retriever, use_flare=use_flare)
    if not use_moderation:
        return chain

    input_variables = ["user_input"] if use_flare else ["chat_history", "question"]
    moderation_input = "response" if use_flare else "answer"
    moderation_chain = OpenAIModerationChain(input_key=moderation_input)
    return SequentialChain(
        chains=[chain, moderation_chain],
        input_variables=input_variables
    )


'''logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER = logging.getLogger()
set_environment()

LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

#LLM = ChatMistral(model_name="facebook/bart-base-mistral", temperature=0, streaming=True)


def configure_retriever(
        docs: list[Document],
        use_compression: bool = False) -> BaseRetriever:
    """Retriever to use."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb:
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
    # Create vectordb with single call to embedding model for texts:
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        },
    )
    if not use_compression:
        return retriever

    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.2
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever,
    )

def configure_chain(retriever: BaseRetriever, use_flare: bool = True) -> Chain:
    output_key = 'response' if use_flare else 'answer'
    MEMORY.output_key = output_key
    params = dict(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000,
    )
    if use_flare:
        return FlareChain.from_llm(**params)
    return ConversationalRetrievalChain.from_llm(**params)
    

def configure_retrieval_chain(
        uploaded_files,
        use_compression: bool = False,
        use_flare: bool = False,
        use_moderation: bool = False
) -> Chain:
    """Read documents, configure retriever, and the chain."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs=docs, use_compression=use_compression)
    chain = configure_chain(retriever=retriever, use_flare=use_flare)
    if not use_moderation:
        return chain

    input_variables = ["user_input"] if use_flare else ["chat_history", "question"]
    moderation_input = "response" if use_flare else "answer"
    moderation_chain = OpenAIModerationChain(input_key=moderation_input)
    return SequentialChain(
        chains=[chain, moderation_chain],
        input_variables=input_variables
    )'''