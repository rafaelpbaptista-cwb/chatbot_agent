"""Script 'principal' onde é construído a aplicação em sim."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from chatbot_agent import (
    GENERATE,
    GRADE_HTML_DOCUMENTS,
    RETRIEVER_HTML,
    RETRIEVER_PYTHON,
    GraphState,
    create_generate,
    create_html_grader,
    create_query_retriever,
    create_verify_code,
)
from chatbot_agent.chain.create_chains import RetrieverOptions

logger = logging.getLogger(__name__)

retriever_html = create_query_retriever(RetrieverOptions.HTML)
retriever_python = create_query_retriever(RetrieverOptions.PYTHON)
html_grader = create_html_grader()
generate = create_generate()
verify_code = create_verify_code()


def retriever_python_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de recuperação RAG de código python."""
    logger.debug("")
    logger.debug("--- RETRIEVER PYTHON---")

    return {
        "documents": retriever_html.invoke(state["question"]),
    }


def retriever_html_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de recuperação RAG docs HTML."""
    logger.debug("")
    logger.debug("--- RETRIEVER HTML---")

    return {
        "documents": retriever_html.invoke(state["question"]),
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
    logger.debug("--- GRADE HTML CONTEXT ---")

    list_valid_documents = [
        answer.analized_document
        for answer in html_grader.invoke(state["question"], state["documents"])
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


def decide_need_code(state: GraphState) -> bool:
    """Tomada de decisão.

    Decide se é necessário obter códigos python para responder ao questionamento do
    usuário

    Parameters
    ----------
        state (GraphState): _description_

    Returns
    -------
        bool:
            True se é necessário obter código puthon e False caso contrário.
    """
    return verify_code.invoke(state["question"], state["documents"]).answer


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
        self._add_conditional()

        self.app = self.workflow.compile()

    def _add_conditional(self) -> None:
        logger.debug("Adicionando condições de rotas")

        self.workflow.add_conditional_edges(
            GRADE_HTML_DOCUMENTS,
            decide_need_code,
            {True: RETRIEVER_PYTHON, False: GENERATE},
        )

    def _add_nodes(self) -> None:
        logger.debug("Adicionando nodes ao graph")

        self.workflow.add_node(RETRIEVER_HTML, retriever_html_node)
        self.workflow.add_node(GRADE_HTML_DOCUMENTS, grade_documents_node)
        self.workflow.add_node(GENERATE, generate_node)
        self.workflow.add_node(RETRIEVER_PYTHON, retriever_python_node)

    def _add_nodes_sequence(self) -> None:
        logger.debug("Adicionando sequencia do nodes do graph")

        self.workflow.add_edge(START, RETRIEVER_HTML)
        self.workflow.add_edge(RETRIEVER_HTML, GRADE_HTML_DOCUMENTS)
        self.workflow.add_edge(GRADE_HTML_DOCUMENTS, GENERATE)
        self.workflow.add_edge(RETRIEVER_PYTHON, GENERATE)
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
