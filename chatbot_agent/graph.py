"""Script 'principal' onde é construído a aplicação em si."""

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
    create_python_grader,
    create_query_retriever,
    create_verify_code,
)
from chatbot_agent.chain.create_chains import RetrieverOptions
from chatbot_agent.consts import GRADE_PYTHON_DOCUMENTS

logger = logging.getLogger(__name__)

retriever_html = create_query_retriever(RetrieverOptions.HTML)
retriever_python = create_query_retriever(RetrieverOptions.PYTHON)
html_grader = create_html_grader()
python_grader = create_python_grader()
generate = create_generate()
verify_code = create_verify_code()


def retriever_python_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de recuperação RAG de código python."""
    logger.info("")
    logger.info("--- RETRIEVER PYTHON---")

    return {
        "codes": retriever_html.invoke(state["question"]),
    }


def retriever_html_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de recuperação RAG docs HTML."""
    logger.info("")
    logger.info("--- RETRIEVER HTML---")

    return {
        "documents": retriever_html.invoke(state["question"]),
    }


def grade_python_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de avaliação de RAG python.

    Retorna todos os documentos com answer True, que representam o contexto mais
    relevante para a pergunta do usuário.

    Parameters
    ----------
        state (GraphState): estado do aplicação

    Returns
    -------
        dict[str, Any]: node com a lista de documentos com score 1.
    """
    logger.info("")
    logger.info("--- GRADE PYTHON CONTEXT ---")

    list_valid_documents = []
    for code in state.get("codes", []):
        response = python_grader.invoke(
            question=state["question"], code=code, documentation=state["documents"]
        )

        if response.answer:
            list_valid_documents.append(response.analyzed_document)

    return {"codes": list_valid_documents}


def grade_html_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de avaliação de RAG HTML.

    Retorna todos os documentos com answer True, que representam o contexto mais
    relevante para a pergunta do usuário.

    Parameters
    ----------
        state (GraphState): estado do aplicação

    Returns
    -------
        list[Document]: lista de documentos com score 1.
    """
    logger.info("")
    logger.info("--- GRADE HTML CONTEXT ---")

    list_valid_documents = []
    for doc in state["documents"]:
        response = html_grader.invoke(question=state["question"], documentation=doc)

        if response.answer:
            list_valid_documents.append(response.analyzed_document)

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
    logger.info("")
    logger.info("--- GENERATE ---")

    response = generate.invoke(
        question=state["question"],
        code=state.get("codes"),
        documentation=state.get("documents"),
    )

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
    logger.info("--- DECIDE NEED CODE ---")

    response = verify_code.invoke(
        question=state["question"], documentation=state["documents"]
    )

    return response.answer


@dataclass
class Application:
    """Classe que representa a aplicação."""

    workflow: StateGraph = field(default_factory=lambda: StateGraph(GraphState))

    app: CompiledStateGraph = field(init=False)

    def __post_init__(self) -> None:
        """Inicializa propriedades do objeto."""
        logger.info("Executando __post_init__")

        self._add_nodes()
        self._add_nodes_sequence()
        self._add_conditional()

        self.app = self.workflow.compile()

    def _add_conditional(self) -> None:
        logger.info("Adicionando condições de rotas")

        self.workflow.add_conditional_edges(
            GRADE_HTML_DOCUMENTS,
            decide_need_code,
            {True: RETRIEVER_PYTHON, False: GENERATE},
        )

    def _add_nodes(self) -> None:
        logger.info("Adicionando nodes ao graph")

        self.workflow.add_node(RETRIEVER_HTML, retriever_html_node)
        self.workflow.add_node(RETRIEVER_PYTHON, retriever_python_node)
        self.workflow.add_node(GRADE_HTML_DOCUMENTS, grade_html_node)
        self.workflow.add_node(GRADE_PYTHON_DOCUMENTS, grade_python_node)
        self.workflow.add_node(GENERATE, generate_node)

    def _add_nodes_sequence(self) -> None:
        logger.info("Adicionando sequencia do nodes do graph")

        self.workflow.add_edge(START, RETRIEVER_HTML)
        self.workflow.add_edge(RETRIEVER_HTML, GRADE_HTML_DOCUMENTS)
        self.workflow.add_edge(GRADE_HTML_DOCUMENTS, GENERATE)
        self.workflow.add_edge(RETRIEVER_PYTHON, GRADE_PYTHON_DOCUMENTS)
        self.workflow.add_edge(GRADE_PYTHON_DOCUMENTS, GENERATE)
        self.workflow.add_edge(GENERATE, END)

    def generate_image(self) -> None:
        """Gera uma imagem do graph do aplicação.

        A imagem é salva em um arquivo chamado "graph.png"
        na pasta atual.
        """
        logger.info("Gerando imagem do graph")

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
