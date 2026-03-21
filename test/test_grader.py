import logging

import pytest

from chatbot_agent import create_documents_html_grader
from chatbot_agent.chain.create_chains import LargeLanguageModel
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@pytest.fixture
def html_grader() -> LargeLanguageModel:
    return create_documents_html_grader()


def test_documents_grader(html_grader: LargeLanguageModel) -> None:
    """Testa se o invoke do RagGrader retorna respostas quando chamado com uma pergunta e contextos."""
    respostas = html_grader.invoke(
        question="Como se conectar na base de dados histórico oficial?",
        context=[
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

    for resp in respostas:
        logger.info("")
        logger.info("Relevant: %s", resp.is_relevante)
        logger.info("Explaination: %s", resp.explaination)
        logger.info("-" * 50)

    assert respostas
