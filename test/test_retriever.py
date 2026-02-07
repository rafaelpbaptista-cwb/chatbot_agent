"""Testes unitários relacionados ao recuperação aumentada (RAG)."""

import logging

import pytest

from chatbot_agent import create_retriever, create_rag_grader
from langchain_core.documents import Document

from chatbot_agent.chain.create_chains import RagGrader, Retriever

logger = logging.getLogger(__name__)


@pytest.fixture
def retriever() -> Retriever:
    return create_retriever()


@pytest.fixture
def rag_grader() -> RagGrader:
    return create_rag_grader()


def test_retriever(retriever: Retriever) -> None:
    """Testa se o invoke do Retriever retorna respostas quando chamado com uma pergunta."""
    respostas = retriever.invoke("Como se conectar na base de dados histórico oficial?")

    for resp in respostas:
        logger.info("Content: %s", resp.page_content)
        logger.info("-" * 50)

    assert respostas


def test_rag_grader(rag_grader: RagGrader) -> None:
    """Testa se o invoke do RagGrader retorna respostas quando chamado com uma pergunta e contextos."""
    respostas = rag_grader.invoke(
        question="Como se conectar na base de dados histórico oficial?",
        documents=[
            Document(
                page_content="from infra_copel import MongoHistoricoOficial\nmongo = MongoHistoricoOficial()"
            ),
            Document(
                page_content="impor pandas as pd\ndf = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})"
            ),
        ],
    )

    for resp in respostas:
        logger.info("Score: %s", resp.Score)
        logger.info("Explaination: %s", resp.Explaination)

    assert respostas
