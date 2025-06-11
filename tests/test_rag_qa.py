import importlib
import runpy
import sys
import types
from unittest.mock import MagicMock

import pytest

# Prepare mocks
embed_mock = MagicMock()
chroma_mock = MagicMock()
llm_mock = MagicMock()
qa_chain_mock = MagicMock()
prompt_mock = MagicMock()

qa_chain_instance = qa_chain_mock.from_chain_type.return_value


def setup_module_mocks(monkeypatch):
    monkeypatch.setitem(sys.modules, 'langchain_community', types.ModuleType('langchain_community'))

    embed_mod = types.ModuleType('langchain_community.embeddings')
    embed_mod.HuggingFaceEmbeddings = embed_mock.HuggingFaceEmbeddings
    monkeypatch.setitem(sys.modules, 'langchain_community.embeddings', embed_mod)

    vector_mod = types.ModuleType('langchain_community.vectorstores')
    vector_mod.Chroma = chroma_mock.Chroma
    monkeypatch.setitem(sys.modules, 'langchain_community.vectorstores', vector_mod)

    llm_mod = types.ModuleType('langchain_community.llms')
    llm_mod.GPT4All = llm_mock.GPT4All
    monkeypatch.setitem(sys.modules, 'langchain_community.llms', llm_mod)

    monkeypatch.setitem(sys.modules, 'langchain', types.ModuleType('langchain'))

    chains_mod = types.ModuleType('langchain.chains')
    chains_mod.RetrievalQA = qa_chain_mock
    monkeypatch.setitem(sys.modules, 'langchain.chains', chains_mod)

    prompt_mod = types.ModuleType('langchain.prompts')
    prompt_mod.PromptTemplate = prompt_mock.PromptTemplate
    monkeypatch.setitem(sys.modules, 'langchain.prompts', prompt_mod)

    # PromptTemplate.from_template should return a simple string
    prompt_mock.PromptTemplate.from_template.return_value = "prompt"
    # vectorstore.as_retriever should return something simple
    chroma_mock.Chroma.return_value.as_retriever.return_value = "retriever"


def test_rag_qa_cli_exit(monkeypatch):
    setup_module_mocks(monkeypatch)
    monkeypatch.setitem(sys.modules, 'config', types.SimpleNamespace(MODEL_PATH='m', VECTOR_DB_DIR='d', EMBEDDING_MODEL_PATH='e'))

    # Input 'exit' once to break the loop
    monkeypatch.setattr('builtins.input', lambda _: 'exit')

    runpy.run_module('rag_qa', run_name='rag_qa')

    embed_mock.HuggingFaceEmbeddings.assert_called_once()
    chroma_mock.Chroma.assert_called_once()
    llm_mock.GPT4All.assert_called_once()
    qa_chain_mock.from_chain_type.assert_called_once()
