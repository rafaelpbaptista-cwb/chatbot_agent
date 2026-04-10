import logging

from langchain.messages import AIMessage, HumanMessage
import pytest

from chatbot_agent import create_rewrite_question_rag
from chatbot_agent.chain.create_chains import LargeLanguageModelHistory
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@pytest.fixture
def rewrite_question_rag() -> LargeLanguageModelHistory:
    return create_rewrite_question_rag()


def test_rewrite_question_rag(rewrite_question_rag: LargeLanguageModelHistory) -> None:
    question = ("Mas não posso obter essa informação usando a classe HistoricoOficial?",)
    history = [
        HumanMessage(content="Como obter informações sobre o preço de pld mensal?"),
        AIMessage(
            content="""Para obter informações sobre o preço de pld mensal você pode usar o dataframe `pld_mensal``.
                Segue exemplo: 

                ```python
                df = pld_mensal

                print(df.head())
                ```"""
        ),
    ]

    response = rewrite_question_rag.invoke(question=question, history=history)

    assert response

    logger.info("Resposta do RewriteQuestionRAG: %s", response.content)