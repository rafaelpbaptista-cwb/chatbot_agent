import logging

import pytest

from chatbot_agent import create_html_grader, create_python_grader
from chatbot_agent.chain.create_chains import LargeLanguageModel
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@pytest.fixture
def html_grader() -> LargeLanguageModel:
    return create_html_grader()


@pytest.fixture
def python_grader() -> LargeLanguageModel:
    return create_python_grader()


def test_html_grader(html_grader: LargeLanguageModel) -> None:
    """Testa se o invoke do RagGrader retorna respostas quando chamado com uma pergunta e contextos."""
    resposta = html_grader.invoke(
        question="Como se conectar na base de dados histórico oficial?",
        documentation=Document(
            page_content="""HistoricoOficial(infra_copel.MongoDatabase):

                Classe que representa conexÃ£o com o 'historico_oficial' no MongoDB.
                Attributes
                db : MongoClient
                    Cliente do MongoDB.



                MongoHistoricoOficial()


                Construtor da classe.
                    Cria o objeto baseado no nome da database."""
        ),
    )

    logger.info("")
    logger.info("Relevant: %s", resposta.answer)
    logger.info("Explaination: %s", resposta.explanation)
    logger.info("-" * 50)

    assert resposta


def test_python_grader(python_grader: LargeLanguageModel) -> None:
    """Testa se o invoke do RagGrader retorna respostas quando chamado com uma pergunta e contextos."""

    resposta = python_grader.invoke(
        question="Como se conectar na base de dados histórico oficial?",
        code=Document(
            page_content="""from infra_copel import HistoricoOficial
                db = MongoHistoricoOficial()
                """
        ),
        documentation=[
            Document(
                page_content="""HistoricoOficial(infra_copel.MongoDatabase):

                Classe que representa conexÃ£o com o 'historico_oficial' no MongoDB.
                Attributes
                db : MongoClient
                    Cliente do MongoDB.



                MongoHistoricoOficial()


                Construtor da classe.
                    Cria o objeto baseado no nome da database."""
            ),
            Document(
                page_content="""DataFrame
                        Constructor
                        DataFrame([data, index, columns, dtype, copy])

                        Two-dimensional, size-mutable, potentially heterogeneous tabular data.

                        Attributes and underlying data
                        Axes"""
            ),
        ],
    )

    logger.info("")
    logger.info("Relevant: %s", resposta.answer)
    logger.info("Explaination: %s", resposta.explanation)
    logger.info("-" * 50)

    assert resposta
