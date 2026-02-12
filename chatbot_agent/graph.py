"""Script 'principal' onde é construído a aplicação em sim."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from chatbot_agent import (
    GENERATE,
    RETRIEVE,
    GraphState,
    create_documents_grader,
    create_generate,
    create_retriever,
)
from chatbot_agent.consts import GRADE_DOCUMENTS

logger = logging.getLogger(__name__)

retriever = create_retriever()
rag_grader = create_documents_grader()
generate = create_generate()


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de recuperação RAG."""
    logger.debug("")
    logger.debug("--- RETRIEVE ---")

    return {
        "documents": retriever.invoke(state["question"]),
    }


def grade_documents_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de avaliação de RAG.

    Retorna todos os documentos com score 1, que representam o contexto mais relevante
    para a pergunta do usuário.

    Parameters
    ----------
        state (GraphState): estado do aplicação

    Returns
    -------
        list[Document]: lista de documentos com score 1.
    """
    logger.debug("")
    logger.debug("--- GRADE CONTEXT ---")

    list_valid_documents = [
        answer.analized_document
        for answer in rag_grader.invoke(state["question"], state["documents"])
        if answer.Score == 1
    ]

    return {"documents": list_valid_documents}


def generate_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de resposta ao usuário.

    Parameters
    ----------
        state (GraphState): estado do aplicação

    Returns
    -------
        dict[str, Any]: node com o texto gerado pelo modelo de LLM
    """
    logger.debug("")
    logger.debug("--- GENERATE ---")

    response = generate.invoke(state["question"], state["documents"])

    return {"response": response.content}


@dataclass
class Application:
    """Classe que representa a aplicação."""

    workflow: StateGraph = field(default_factory=lambda: StateGraph(GraphState))

    app: CompiledStateGraph = field(init=False)

    def __post_init__(self) -> None:
        """Inicializa propriedades do objeto."""
        logger.debug("Executando __post_init__")

        self._add_nodes()
        self._add_nodes_sequence()

        self.app = self.workflow.compile()

    def _add_nodes(self) -> None:
        logger.debug("Adicionando nodes ao graph")

        self.workflow.add_node(RETRIEVE, retrieve_node)
        self.workflow.add_node(GRADE_DOCUMENTS, grade_documents_node)
        self.workflow.add_node(GENERATE, generate_node)

    def _add_nodes_sequence(self) -> None:
        logger.debug("Adicionando sequencia do nodes do graph")

        self.workflow.add_edge(START, RETRIEVE)
        self.workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        self.workflow.add_edge(GRADE_DOCUMENTS, GENERATE)
        self.workflow.add_edge(GENERATE, END)

    def generate_image(self) -> None:
        """Gera uma imagem do graph do aplicação.

        A imagem é salva em um arquivo chamado "graph.png"
        na pasta atual.
        """
        logger.debug("Gerando imagem do graph")

        with Path("graph.png").open("wb") as f:
            f.write(self.app.get_graph().draw_mermaid_png())

    def invoke(self, question: str) -> str:
        """Responde ao questionamento do usuário.

        Args:
            question (str):
                Questionamento do usuário

        Returns
        -------
            str:
                Resposta ao questionamento do usuário
        """
        return self.app.invoke(input={"question": question})["response"]
