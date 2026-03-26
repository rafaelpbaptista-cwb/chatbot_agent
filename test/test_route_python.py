import logging

from langchain_classic.schema import Document
import pytest

from chatbot_agent import create_verify_code
from chatbot_agent.chain.create_chains import LargeLanguageModel
from chatbot_agent.structured_output.models import DocumentsGraderAnswer

logger = logging.getLogger(__name__)


@pytest.fixture
def verify_code() -> LargeLanguageModel:
    return create_verify_code()


def test_verify_code(verify_code: LargeLanguageModel) -> None:
    resposta: DocumentsGraderAnswer = verify_code.invoke(
        question="Como se conectar na base de dados histórico oficial?",
        context=
            Document(
                page_content="""class MongoHistoricoOficial(infra_copel.MongoDatabase):

                                Classe que representa conexão com o 'historico_oficial' no MongoDB.

                                Attributes
                                db : MongoClient Cliente do MongoDB.

                                MongoHistoricoOficial()
                                Construtor da classe.

                                Cria o objeto baseado no nome da database."""
            ),
    )

    assert resposta

    logger.info("Documentação suficiente? %s", resposta.answer)
    logger.info("Explicação: %s", resposta.explaination)


