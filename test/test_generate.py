import logging
from chatbot_agent import generate
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def test_generate():
    response = generate.invoke(
        "O que é um chatbot?",
        documents=[
            Document(
                page_content="from infra_copel import MongoHistoricoOficial\nmongo = MongoHistoricoOficial()"
            ),
        ],
    )

    logger.info("")
    logger.info("-" * 50)
    logger.info("Response: %s", response)
    logger.info("-" * 50)

    assert response
