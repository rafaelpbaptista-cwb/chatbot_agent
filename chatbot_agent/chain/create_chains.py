"""Script responsável por instanciar os LLMs/chains a serem utilizados na aplicação."""

import os
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
from langchain_core.runnables.base import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from chatbot_agent.instructions.system_message_template import (
    GENERATE,
    HTML_GRADER,
    PYTHON_GRADER,
    PYTHON_VERIFY,
)
from chatbot_agent.structured_output.models import DocumentsGraderAnswer


class RetrieverOptions(Enum):
    """Enum com opções as opções válidas para o Retriever."""

    HTML = "html"
    PYTHON = "python"


@dataclass
class LargeLanguageModel:
    """Classe que representa um modelo LLM.

    Classe utilizada para insteração com LLM para diversos propósitos,
    como geração de respostas ou avaliação de documentos.
    """

    llm: ChatGoogleGenerativeAI = field(
        default_factory=lambda: ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL"), temperature=0
        )
    )

    system_prompt_template: SystemMessagePromptTemplate = field(init=False)

    human_prompt_template: HumanMessagePromptTemplate = field(init=False)

    chain: RunnableSequence = field(init=False)

    system_instruction: str = field(init=True, default=None)

    structured_output: DocumentsGraderAnswer = field(init=True, default=None)

    def __post_init__(self) -> None:
        """Inicializa as propriedades da objeto."""
        self.system_prompt_template = SystemMessagePromptTemplate.from_template(
            template=self.system_instruction
        )

        self.human_prompt_template = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"],
                template="{question}",
            )
        )

        if self.structured_output:
            self.llm = self.llm.with_structured_output(DocumentsGraderAnswer)

        self.chain = (
            ChatPromptTemplate(
                input_variables=["context", "question"],
                messages=[self.system_prompt_template, self.human_prompt_template],
            )
            | self.llm
        )

    def invoke(
        self,
        question: str,
        **extra_args_system_prompt: str,
    ) -> Document:
        """Geração da resposta ao questionamento em questão.

        Parameters
        ----------
            question (str):
                Questionamento

            **extra_args_system_prompt (str):
                Argumentos extras para o system prompt

        Returns
        -------
            Document:
                Resposta gerada pelo modelo
        """
        return self.chain.invoke({"question": question, **extra_args_system_prompt})


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


def create_html_grader() -> LargeLanguageModel:
    """Cria um objeto para avaliação de documentos HTML.

    Returns
    -------
    LargeLanguageModel:
        Objeto com a instrução de sistema para executar o avaliador de
        documentos HTML
    """
    return LargeLanguageModel(
        system_instruction=HTML_GRADER, structured_output=DocumentsGraderAnswer
    )


def create_python_grader() -> LargeLanguageModel:
    """Cria um objeto para avaliação de código python.

    Returns
    -------
    LargeLanguageModel:
        Objeto com a instrução de sistema para executar o avaliador de
        documentos HTML
    """
    return LargeLanguageModel(
        system_instruction=PYTHON_GRADER, structured_output=DocumentsGraderAnswer
    )


def create_generate() -> LargeLanguageModel:
    """Cria um objeto para geração de respostas.

    Returns
    -------
    Generate: objeto para geração de respostas.
    """
    return LargeLanguageModel(system_instruction=GENERATE)


def create_verify_code() -> LargeLanguageModel:
    """Cria um objeto para avalidar necessidade de pesquisa de códigos em Python.

    Returns
    -------
    LargeLanguageModel:
        Objeto com a instrução de sistema para executar o avaliador de necessidade de
        códigos em Python
    """
    return LargeLanguageModel(
        system_instruction=PYTHON_VERIFY, structured_output=DocumentsGraderAnswer
    )
