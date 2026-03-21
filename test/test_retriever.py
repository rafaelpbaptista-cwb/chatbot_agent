"""Testes unitários relacionados ao recuperação aumentada (RAG)."""

import logging

import pytest

from chatbot_agent import create_query_retriever

from chatbot_agent.chain.create_chains import MultiQueryRetriever, RetrieverOptions

logger = logging.getLogger(__name__)


@pytest.fixture(params=[RetrieverOptions.PYTHON, RetrieverOptions.HTML])
def info_retrievier(request) -> tuple[MultiQueryRetriever, str]:
    return create_query_retriever(request.param), request.param

def test_retriever(info_retrievier: tuple[MultiQueryRetriever, str]) -> None:
    logger.info("")
    logger.info("-" * 50)

    retriever, type_data_query = info_retrievier
    logger.info("Testando o retriever com o tipo de dado: %s", type_data_query)

    respostas = retriever.invoke("Como se conectar na base de dados histórico oficial?")

    logger.info("Qtdade de respostas: %s", len(respostas))

    for resp in respostas:
        assert resp.metadata["type_data"] == type_data_query.value
        logger.info("Content: %s", resp.page_content)
        logger.info("-" * 50)

    assert respostas

