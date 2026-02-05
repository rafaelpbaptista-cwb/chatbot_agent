"""Script responsável por instanciar os LLMs/chains a serem utilizados na aplicação."""

import os
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.runnables.base import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langsmith import Client
from pydantic import BaseModel, Field


class RagGraderAnswer(BaseModel):
    """Classe estruturada usada para retornar resposta do RagGrader."""

    Score: int = Field(description="Relevance score between 0 and 1")
    Explaination: str = Field(description="Explanation of relevance score")


@dataclass
class RagGrader:
    """Classe que avalia a qualidade dos RAGs obtidos."""

    prompt: StructuredPrompt = field(
        default_factory=lambda: Client().pull_prompt("rlm/rag-document-relevance")
    )

    llm: ChatGoogleGenerativeAI = field(
        default_factory=lambda: ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL"))
    )

    chain: RunnableSequence = field(init=False)

    def __post_init__(self) -> None:
        """Método interno do dataclass.

        Inicializa propriedade após inicialização do
        objeto.
        """
        self.chain = self.prompt | self.llm

    def invoke(self, question: str, documents: list[Document]) -> list[RagGraderAnswer]:
        """
        Método para avaliar a qualidade dos itens recuperados via RAG.

        Parameters
        ----------
        question (str):
            Questionamento do usuário

        documents (list[Document]):
            Lista de documentos recuperados

        Returns
        -------
            list[RagGraderAnswer]: Lista de avaliações de qualidade dos itens
            recuperados
        """
        return [
            RagGraderAnswer(
                **self.chain.invoke({"input": {"documents": doc, "question": question}})
            )
            for doc in documents
        ]


@dataclass
class Retriever:
    """Classe para recuperação aumentada (RAG).

    Classe utilizada para crescentar contexto aos questionamentos
    dos usuários
    """

    client_vector_db: Chroma = field(
        default_factory=lambda: Chroma(
            persist_directory=os.getenv("DIR_DATABASE"),
            embedding_function=GoogleGenerativeAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL")
            ),
        )
    )

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


def create_retriever() -> Retriever:
    """Cria um objeto para recuperação aumentada (RAG) no vector database.

    Returns
    -------
    Retriever: objeto para recuperação aumentada (RAG).
    """
    return Retriever()


def create_rag_grader() -> RagGrader:
    """Cria um objeto para avaliação de RAG.

    Returns
    -------
    RagGrader: objeto para avaliação de RAG.
    """
    return RagGrader()
