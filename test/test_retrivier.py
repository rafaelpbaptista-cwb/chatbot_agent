"""Testes unitários relacionados ao recuperação aumentada (RAG)."""

import logging

from chatbot_agent import retriever

logger = logging.getLogger(__name__)


def test_retriever() -> None:
    """Testa se o invoke do Retriever retorna respostas quando chamado com uma pergunta."""
    respostas = retriever.invoke("Como se conectar na base de dados histórico oficial?")

    for resp in respostas:
        logger.info("Content: %s", resp.page_content)
        logger.info("-" * 50)

    assert respostas
