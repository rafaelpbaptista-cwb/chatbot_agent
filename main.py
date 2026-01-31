"""Script main para validar o ambiente."""

import logging
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Prints a message indicating the script is running."""
    logger.info("Modelo LLM: %s", os.getenv("LLM_MODEL"))

    chain = ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL")) | StrOutputParser()
    logger.info("Hello World!\nResponse: %s", chain.invoke("Hello World!"))


if __name__ == "__main__":
    main()
