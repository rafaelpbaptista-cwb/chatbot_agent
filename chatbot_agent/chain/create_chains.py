"""Script responsável por instanciar os LLMs/chains a serem utilizados na aplicação."""

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import Client
from pydantic import BaseModel, Field


def _create_client_vector_db() -> Chroma:
    return Chroma(
        persist_directory=os.getenv("DIR_DATABASE"),
        embedding_function=GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL")
        ),
    )


class Retriever(BaseModel):
    """Classe para recuperação aumentada (RAG).

    Classe utilizada para crescentar contexto aos questionamentos
    dos usuários
    """

    model_config = {"arbitrary_types_allowed": True}

    client_vector_db: Chroma = Field(default_factory=_create_client_vector_db)

    def invoke(self, prompt: str) -> list[Document]:
        """Método para recuperar o contexto mais relevante para uma pergunta.

        Recupera o contexto mais relevante para uma pergunta utilizando o modelo de LLM
        configurado na aplicação.

        Parameters
        ----------
            prompt (str): pergunta do usuário.

        Returns
        -------
            list[Document]: contextos mais relevante.
        """
        return self.client_vector_db.max_marginal_relevance_search(prompt)


def create_rag_grader() -> None:
    """Cria um avaliador para verificar a qualidade dos RAG.

    LLM cuja tarefa é verificar se os documentos recuperados para uso da ferramenta RAG
    são relevantes para o contexto da pergunta do usuário.
    """
    return Client().pull_prompt("rlm/rag-document-relevance")


def create_retriever() -> Retriever:
    """Cria um objeto para recuperação aumentada (RAG) no vector database.

    Returns
    -------
    Retriever: objeto para recuperação aumentada (RAG).
    """
    return Retriever()
