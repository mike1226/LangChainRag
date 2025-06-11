import importlib
import sys
import types
from unittest.mock import MagicMock
import os

import pytest

# Create dummy modules to satisfy imports
loader_mock = MagicMock()
text_loader_instance = loader_mock.TextLoader.return_value
text_loader_instance.load.return_value = ["dummy doc"]
pdf_loader_instance = loader_mock.PyPDFLoader.return_value
pdf_loader_instance.load.return_value = ["dummy pdf"]

splitter_mock = MagicMock()
splitter_instance = splitter_mock.RecursiveCharacterTextSplitter.return_value
splitter_instance.split_documents.return_value = ["chunk"]

embed_mock = MagicMock()
embed_instance = embed_mock.HuggingFaceEmbeddings.return_value

chroma_mock = MagicMock()
vectorstore_instance = chroma_mock.Chroma.from_documents.return_value


def setup_module_mocks(monkeypatch):
    monkeypatch.syspath_prepend(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    loader_mod = types.ModuleType('langchain_community.document_loaders')
    loader_mod.TextLoader = loader_mock.TextLoader
    loader_mod.PyPDFLoader = loader_mock.PyPDFLoader
    monkeypatch.setitem(sys.modules, 'langchain_community', types.ModuleType('langchain_community'))
    monkeypatch.setitem(sys.modules, 'langchain_community.document_loaders', loader_mod)

    splitter_mod = types.ModuleType('langchain.text_splitter')
    splitter_mod.RecursiveCharacterTextSplitter = splitter_mock.RecursiveCharacterTextSplitter
    monkeypatch.setitem(sys.modules, 'langchain', types.ModuleType('langchain'))
    monkeypatch.setitem(sys.modules, 'langchain.text_splitter', splitter_mod)

    embed_mod = types.ModuleType('langchain_community.embeddings')
    embed_mod.HuggingFaceEmbeddings = embed_mock.HuggingFaceEmbeddings
    monkeypatch.setitem(sys.modules, 'langchain_community.embeddings', embed_mod)

    vector_mod = types.ModuleType('langchain_community.vectorstores')
    vector_mod.Chroma = chroma_mock.Chroma
    monkeypatch.setitem(sys.modules, 'langchain_community.vectorstores', vector_mod)


def test_build_vector_store(monkeypatch):
    setup_module_mocks(monkeypatch)
    import embeddings_loader
    importlib.reload(embeddings_loader)
    embeddings_loader.build_vector_store()

    loader_mock.TextLoader.assert_called_once()
    text_loader_instance.load.assert_called_once()
    splitter_mock.RecursiveCharacterTextSplitter.assert_called_once()
    splitter_instance.split_documents.assert_called_once_with(["dummy doc"])
    embed_mock.HuggingFaceEmbeddings.assert_called_once()
    chroma_mock.Chroma.from_documents.assert_called_once()
    vectorstore_instance.persist.assert_called_once()


def test_build_vector_store_from_pdf(monkeypatch):
    setup_module_mocks(monkeypatch)
    import embeddings_loader
    importlib.reload(embeddings_loader)
    embeddings_loader.build_vector_store_from_pdf("dummy.pdf")

    loader_mock.PyPDFLoader.assert_called_once_with("dummy.pdf")
    pdf_loader_instance.load.assert_called_once()
    splitter_mock.RecursiveCharacterTextSplitter.assert_called()
    splitter_instance.split_documents.assert_called()
    embed_mock.HuggingFaceEmbeddings.assert_called()
    chroma_mock.Chroma.from_documents.assert_called()
    vectorstore_instance.persist.assert_called()
