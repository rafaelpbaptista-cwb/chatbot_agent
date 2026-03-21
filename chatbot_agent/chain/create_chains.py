"""Script responsável por instanciar os LLMs/chains a serem utilizados na aplicação."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.runnables.base import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langsmith import Client
from pydantic import BaseModel

from chatbot_agent.instructions.system_message_template import (
    GENERATE,
)
from chatbot_agent.structured_output.models import (
    DocumentsGraderAnswer,
)


class RetrieverOptions(Enum):
    """Enum com opções as opções válidas para o Retriever."""

    HTML = "html"
    PYTHON = "python"


@dataclass
class BaseChain(ABC):
    """Classe base para instanciar os LLMs/chains a serem utilizados na aplicação."""

    prompt: StructuredPrompt = field(init=False)

    llm: ChatGoogleGenerativeAI = field(
        default_factory=lambda: ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL"), temperature=0
        )
    )

    chain: RunnableSequence = field(init=False)

    pull_prompt: str = field(default=None)

    def __post_init__(self) -> None:
        """Método interno do dataclass.

        Inicializa propriedade após inicialização do
        objeto.
        """
        self.prompt = Client().pull_prompt(self.pull_prompt)
        self.chain = self.prompt | self.llm

    @abstractmethod
    def invoke(self, question: str, documents: list[Document]) -> list[BaseModel]:
        """Definição do método para a execução de cada chain.

        Comportamento dessa invocação deverá ser detalhada em cada classe filha.

        Parameters
        ----------
        question (str):
            Questionamento relevante para cada chain

        documents (list[Document]):
            Lista de documentos relevante para cada chain

        Returns
        -------
            list[BaseModel]: Resposta definida em cada classe filha
        """


@dataclass
class DocumentsGrader(BaseChain):
    """Classe que avalia a qualidade dos RAGs (docuentos) obtidos."""

    pull_prompt: str = field(default="rlm/rag-document-relevance")

    def invoke(
        self, question: str, documents: list[Document]
    ) -> list[DocumentsGraderAnswer]:
        """
        Método para avaliar a qualidade dos itens recuperados via RAG.

        Score 1 documento está relacionado ao questionamento do usuário.

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
            DocumentsGraderAnswer(
                **self.chain.invoke(
                    {"input": {"documents": doc, "question": question}}
                ),
                analized_document=doc,
            )
            for doc in documents
        ]


@dataclass
class Generate(BaseChain):
    """Classe responsável por responder aos questionamentos do usuário."""

    system_prompt_template: SystemMessagePromptTemplate = field(
        default_factory=lambda: SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=GENERATE,
                input_variables=["context"],
            )
        )
    )

    human_prompt_template: HumanMessagePromptTemplate = field(
        default_factory=lambda: HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"],
                template="{question}",
            )
        )
    )

    def __post_init__(self) -> None:
        """Inicializa as propriedades da objeto."""
        self.chain = (
            ChatPromptTemplate(
                input_variables=["context", "question"],
                messages=[self.system_prompt_template, self.human_prompt_template],
            )
            | self.llm
        )

    def invoke(self, question: str, documents: list[Document]) -> str:
        """
        Método para responder aos questionamentos do usuário.

        Recebe uma pergunta e uma lista de documentos RAGs que ajudaram
        no contexto para responder ao questionamento do usuário.

        Parameters
        ----------
        question (str):
            Questionamento do usuário

        documents (list[Document]):
            Lista de documentos recuperados

        Returns
        -------
            str: Resposta ao questionamento do usuário.
        """
        return self.chain.invoke({"question": question, "context": documents})


@dataclass
class QueryRetriever:
    """Classe que representa um modelo de recuperação de documentos.

    Parameters
    ----------
    client_vector_db : Chroma
        Instância do Chroma que representa o banco de vetores

    type_data_query : str
        Tipo de dados que será usado na consulta

    Returns
    -------
    QueryRetriever
        Instância do QueryRetriever

    Notes
    -----
    Essa classe é responsável por recuperar os documentos mais relevantes
    para uma pergunta do usuário, podendo recuperar documentação (markdown) ou códido
    (python)
    """

    client_vector_db: Chroma = field(
        default_factory=lambda: Chroma(
            persist_directory=os.getenv("DIR_DATABASE"),
            embedding_function=GoogleGenerativeAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL")
            ),
        )
    )

    type_data_query: RetrieverOptions = field(default=RetrieverOptions.PYTHON)

    multi_query_retrivier: MultiQueryRetriever = field(init=False)

    def __post_init__(self) -> None:
        """Iniciar propriedades do objeto."""
        self.multi_query_retrivier = MultiQueryRetriever.from_llm(
            retriever=self.client_vector_db.as_retriever(
                search_kwargs={"filter": {"type_data": self.type_data_query.value}}
            ),
            llm=ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL"), temperature=0),
        )

    def invoke(self, prompt: str) -> list[Document]:
        """Recuperação de informações relevantes para a pergunta do usuário.

        Parameters
        ----------
        prompt (str):
            Pergunta do usuário

        Returns
        -------
        list[Document]:
            Lista de documentos mais relevantes para a pergunta do usuário
        """
        return self.multi_query_retrivier.invoke(prompt)


def create_query_retriever(type_data_query: RetrieverOptions) -> QueryRetriever:
    """Cria um objeto QueryRetriever com o tipo de dado especificado.

    Parameters
    ----------
    type_data_query (RetrieverOptions):
        Tipo de dado a ser recuperado

    Returns
    -------
    QueryRetriever:
        Objeto para recuperação de informações relevantes para a pergunta do usuário
    """
    return QueryRetriever(type_data_query=type_data_query)


def create_documents_grader() -> DocumentsGrader:
    """Cria um objeto para avaliação de RAG.

    Returns
    -------
    RagGrader: objeto para avaliação de RAG.
    """
    return DocumentsGrader()


def create_generate() -> Generate:
    """Cria um objeto para geração de respostas.

    Returns
    -------
    Generate: objeto para geração de respostas.
    """
    return Generate()
