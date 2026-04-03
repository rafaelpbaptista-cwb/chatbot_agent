import enum
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


def test_html_grader_batch(html_grader: LargeLanguageModel) -> None:
    inputs = [
        {
            "question": "Como se conectar na base de dados histórico oficial?",
            "documentation": Document(
                page_content="""HistoricoOficial(infra_copel.MongoDatabase):

                Classe que representa conexÃ£o com o 'historico_oficial' no MongoDB.
                Attributes
                db : MongoClient
                    Cliente do MongoDB.



                MongoHistoricoOficial()


                Construtor da classe.
                    Cria o objeto baseado no nome da database."""
            ),
        },
        {
            "question": "Como se conectar na base de dados histórico oficial?",
            "documentation": Document(
                page_content="""DataFrame
                    Constructor
                    DataFrame([data, index, columns, dtype, copy])

                    Two-dimensional, size-mutable, potentially heterogeneous tabular data."""
            ),
        },
    ]

    response = html_grader.batch(inputs)
    assert len(response) == 2

    for index, resposta in enumerate(response):
        assert resposta.answer if index == 0 else not resposta.answer

        logger.info("")
        logger.info("Relevant: %s", resposta.answer)
        logger.info("Explaination: %s", resposta.explanation)
        logger.info("-" * 50)


def test_python_grader_batch(python_grader: LargeLanguageModel) -> None:
    """Testa se o invoke do RagGrader retorna respostas quando chamado com uma pergunta e contextos."""

    inputs = [
        {
            "question": "Como se conectar na base de dados histórico oficial?",
            "code": Document(
                page_content="""from infra_copel import HistoricoOficial
                db = MongoHistoricoOficial()
                """
            ),
            "documentation": [
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
            ],
        },
        {
            "question": "Como se conectar na base de dados histórico oficial?",
            "code": Document(
                page_content="""import pandas as pd
                df = pd.DataFrame()
                """
            ),
            "documentation": [
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
            ],
        },
    ]

    response = python_grader.batch(inputs)
    assert len(response) == 2

    for index, resposta in enumerate(response):
        assert resposta.answer if index == 0 else not resposta.answer

        logger.info("")
        logger.info("Relevant: %s", resposta.answer)
        logger.info("Explaination: %s", resposta.explanation)
        logger.info("-" * 50)
