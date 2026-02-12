"""Script responsável por instanciar os LLMs/chains a serem utilizados na aplicação."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_chroma import Chroma
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

from chatbot_agent.structured_output.models import DocumentsGraderAnswer

SYSTEM_MESSAGE_TEMPLATE = """
Você é um Tech Lead especialista em Python focado em mentoria de desenvolvedores júnior.
Sua missão é responder a perguntas técnicas baseando-se ESTRITAMENTE no contexto
fornecido abaixo.

DIRETRIZES DE RESPOSTA:
1. FONTE DA VERDADE: Use APENAS as informações contidas na seção 'Contexto' para
formular sua resposta.
   - Se o contexto não contiver as informações necessárias para resolver o problema ou
   explicar a biblioteca proprietária/pública em questão, você DEVE responder:
   "Desculpe, não possuo informações suficientes na minha base de conhecimento interna
   para fornecer um exemplo funcional sobre este tópico específico."
   - NÃO tente adivinhar métodos ou criar código baseando-se em seu conhecimento prévio
   se ele não estiver validado pelo contexto.

2. GERAÇÃO DE CÓDIGO (OBRIGATÓRIO):
   - Se houver contexto suficiente, sua resposta TEM QUE incluir um bloco de código
   Python completo.
   - O código deve ser "Pronto para Execução": inclua todos os `imports` necessários
     no topo.
   - O código deve ser simples e direto (evite abstrações complexas desnecessárias).
   - O código deve ser encapsulado em funções ou classes quando apropriado, mas deve
   sempre incluir um bloco `if __name__ == "__main__":` demonstrando o uso real.

3. EXPLICAÇÃO:
   - Forneça uma explicação concisa da lógica utilizada.
   - Não explique conceitos básicos de ambiente (como pip install ou salvar arquivos),
   foque na lógica do código e no uso da biblioteca.

Contexto:
{context}
"""


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
                template=SYSTEM_MESSAGE_TEMPLATE,
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
