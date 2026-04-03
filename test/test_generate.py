import logging

from langchain.messages import HumanMessage
import pytest
from chatbot_agent import create_generate, create_generate_history
from langchain_core.documents import Document

from chatbot_agent.chain.create_chains import (
    LargeLanguageModel,
    LargeLanguageModelHistory,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def generate() -> LargeLanguageModel:
    return create_generate()


@pytest.fixture
def generate_history() -> LargeLanguageModelHistory:
    return create_generate_history()


def test_generate(generate: LargeLanguageModel) -> None:
    response = generate.invoke(
        question="Como criar uma conexão com o banco de dados histórico oficial?",
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


def test_generate_history(generate_history: LargeLanguageModelHistory) -> None:
    question = HumanMessage(
        content="Como criar uma conexão com o banco de dados histórico oficial?"
    )

    response = generate_history.invoke(
        question=question,
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
        history=[],
    )

    assert response

    logger.info("")
    logger.info("-" * 50)
    logger.info("1º resposta: %s", response.content)
    logger.info("-" * 50)

    history = [question, response]

    question2 = HumanMessage(
        content="Me explique com outras palavras e exemplos porque ainda não entendi"
    )

    response2 = generate_history.invoke(
        question=question2,
        history=history,
        code=[],
        documentation=[],
    )

    assert response2

    logger.info("")
    logger.info("-" * 50)
    logger.info("2º resposta: %s", response2.content)
    logger.info("-" * 50)

    history.extend([question2, response2])

    response3 = generate_history.invoke(
        question=HumanMessage(
            content="Me relembre todas as perguntas que eu fiz até agora"
        ),
        history=history,
        code=[],
        documentation=[],
    )

    assert response3

    logger.info("")
    logger.info("-" * 50)
    logger.info("3º resposta: %s", response3.content)
    logger.info("-" * 50)
