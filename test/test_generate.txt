import logging

import pytest
from sympy import Ge
from chatbot_agent import create_generate
from langchain_core.documents import Document

from chatbot_agent.chain.create_chains import Generate

logger = logging.getLogger(__name__)


@pytest.fixture
def generate() -> Generate:
    return create_generate()


def test_generate(generate: Generate) -> None:
    response = generate.invoke(
        "Como criar uma conexão com o banco de dados histórico oficial?",
        documents=[
            Document(
                page_content="from infra_copel import MongoHistoricoOficial\nmongo = MongoHistoricoOficial()"
            ),
            Document(
                page_content="mongo = MongoHistoricoOficial()"
            ),
        ],
    )

    logger.info("")
    logger.info("-" * 50)
    logger.info("Response: %s", response.content)
    logger.info("-" * 50)

    assert response
