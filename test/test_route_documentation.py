import enum
import logging

from langchain.messages import AIMessage, HumanMessage
import pytest

from chatbot_agent import create_verify_documentation
from chatbot_agent.chain.create_chains import LargeLanguageModel
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@pytest.fixture
def verify_documentation() -> LargeLanguageModel:
    return create_verify_documentation()


def test_verify_documentation(verify_documentation: LargeLanguageModel) -> None:

    question = ("Me forneça mais um exemplo?",)
    documentation = Document(
        page_content="""HistoricoOficial(infra_copel.MongoDatabase):

                Classe que representa conexão com o 'historico_oficial' no MongoDB.
                Attributes
                db : MongoClient
                    Cliente do MongoDB.



                MongoHistoricoOficial()


                Construtor da classe.
                    Cria o objeto baseado no nome da database."""
    )
    history = [
        HumanMessage(content="Como se conectar na base de dados histórico oficial?"),
        AIMessage(
            content="""Para se conectar na base de dados histórico oficial, você pode usar a classe `HistoricoOficial` que é uma extensão da classe `MongoDatabase`. 
                Aqui está um exemplo de como criar uma instância dessa classe:

                ```python
                from infra_copel import HistoricoOficial

                historico_oficial = HistoricoOficial()

                df = historico_oficial.pld_mensal

                print(df.head())
                ```"""
        ),
    ]

    response = verify_documentation.invoke(
        question=question, documentation=documentation, code=[], history=history
    )

    assert response

    assert response.answer is False
