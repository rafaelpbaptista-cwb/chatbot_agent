import logging

import pytest
from sympy import Ge
from chatbot_agent import create_generate
from langchain_core.documents import Document

from chatbot_agent.chain.create_chains import LargeLanguageModel

logger = logging.getLogger(__name__)


@pytest.fixture
def generate() -> LargeLanguageModel:
    return create_generate()


def test_generate(generate: LargeLanguageModel) -> None:
    response = generate.invoke(
        "Como criar uma conexão com o banco de dados histórico oficial?",
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
        ],
        code=[
            Document(
                page_content="""from infra_copel import HistoricoOficial
                db = MongoHistoricoOficial()
                """
            ),
        ],
    )

    logger.info("")
    logger.info("-" * 50)
    logger.info("Response: %s", response.content)
    logger.info("-" * 50)

    assert response
