import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

# Create dummy modules to satisfy imports
loader_mock = MagicMock()
loader_instance = loader_mock.TextLoader.return_value
loader_instance.load.return_value = ["dummy doc"]

splitter_mock = MagicMock()
splitter_instance = splitter_mock.RecursiveCharacterTextSplitter.return_value
splitter_instance.split_documents.return_value = ["chunk"]

embed_mock = MagicMock()
embed_instance = embed_mock.HuggingFaceEmbeddings.return_value

chroma_mock = MagicMock()
vectorstore_instance = chroma_mock.Chroma.from_documents.return_value


def setup_module_mocks(monkeypatch):
    loader_mod = types.ModuleType('langchain_community.document_loaders')
    loader_mod.TextLoader = loader_mock.TextLoader
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
    loader_instance.load.assert_called_once()
    splitter_mock.RecursiveCharacterTextSplitter.assert_called_once()
    splitter_instance.split_documents.assert_called_once_with(["dummy doc"])
    embed_mock.HuggingFaceEmbeddings.assert_called_once()
    chroma_mock.Chroma.from_documents.assert_called_once()
    vectorstore_instance.persist.assert_called_once()
