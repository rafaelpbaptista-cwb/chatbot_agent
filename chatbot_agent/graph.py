"""Script 'principal' onde é construído a aplicação em si."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from google.genai.errors import ServerError
from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import trim_messages
from langgraph.graph import END, START, StateGraph

from chatbot_agent import (
    GENERATE,
    GRADE_HTML_DOCUMENTS,
    LIMITE_TOKENS_INPUT,
    RETRIEVER_HTML,
    RETRIEVER_PYTHON,
    REWRITE_QUESTION,
    GraphState,
    create_html_grader,
    create_python_grader,
    create_query_retriever,
    create_rewrite_question_rag,
    create_verify_code,
    create_verify_documentation,
)
from chatbot_agent.chain.create_chains import RetrieverOptions, create_generate_history
from chatbot_agent.consts import GRADE_PYTHON_DOCUMENTS, RELOAD_RAGS_MEMORY

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

retriever_html = create_query_retriever(RetrieverOptions.HTML)
retriever_python = create_query_retriever(RetrieverOptions.PYTHON)
html_grader = create_html_grader()
python_grader = create_python_grader()
generate = create_generate_history()
verify_code = create_verify_code()
verify_documentation = create_verify_documentation()
rewrite_question = create_rewrite_question_rag()

history = []
last_codes = []
last_htmls = []


def retriever_python_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de recuperação RAG de código python."""
    logger.info("")
    logger.info("--- RETRIEVER PYTHON---")

    try:
        return {
            "codes": retriever_python.invoke(state.rewrited_question),
        }
    except (httpx.TimeoutException, ServerError) as e:
        logger.exception("Erro:", exc_info=e)
        return {"codes": []}


def retriever_html_node(state: GraphState) -> dict[str, Any]:
    """Retorna um node que executa a etapa de recuperação RAG docs HTML."""
    logger.info("")
    logger.info("--- RETRIEVER HTML---")

    try:
        return {
            "documents": retriever_html.invoke(state.rewrited_question),
        }
    except (httpx.TimeoutException, ServerError) as e:
        logger.exception("Erro:", exc_info=e)
        return {"documents": []}


def grade_python_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
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

    if not state.codes:
        logger.info("Nenhum código para avaliar")
        return {"codes": []}

    inputs = [
        {
            "question": state.question,
            "code": code,
            "documentation": state.documents,
        }
        for code in state.codes
    ]

    validated_codes = []

    count_rejected_codes = 0
    list_codes = python_grader.batch(inputs, config)
    for resposta in list_codes:
        if (
            hasattr(resposta, "answer")
            and resposta.answer
            and hasattr(resposta, "analyzed_document")
            and "page_content" in resposta.analyzed_document
        ):
            validated_codes.append(resposta.analyzed_document)
        else:
            count_rejected_codes += 1

    logger.info("Codes rejeitados: %d", count_rejected_codes)
    logger.info(
        "Percentual de codes rejeitados: %.2f%%",
        (count_rejected_codes / len(inputs)) * 100,
    )

    return {"codes": validated_codes}


def grade_html_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
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

    if not state.documents:
        logger.info("Nenhum documento HTML para avaliar")
        return {"documents": []}

    inputs = [
        {
            "question": state.question,
            "documentation": doc,
        }
        for doc in state.documents
    ]

    validated_documents = []

    count_rejected_docs = 0
    list_documents = html_grader.batch(inputs, config)
    for resposta in list_documents:
        if (
            hasattr(resposta, "answer")
            and resposta.answer
            and hasattr(resposta, "analyzed_document")
            and "page_content" in resposta.analyzed_document
        ):
            validated_documents.append(resposta.analyzed_document)
        else:
            count_rejected_docs += 1

    logger.info("Documentos rejeitados: %d", count_rejected_docs)
    logger.info(
        "Percentual de documentos rejeitados: %.2f%%",
        (count_rejected_docs / len(inputs)) * 100,
    )

    return {"documents": validated_documents}


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

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            response = generate.invoke(
                question=state.question,
                code=state.codes,
                documentation=state.documents,
                history=history,
            )

            break
        except Exception as e:
            retries += 1
            logger.exception(
                "Erro ao gerar resposta, tentando novamente... (tentativa %d)",
                retries,
                exc_info=e,
            )

            response = AIMessage(
                content="""Desculpe, estou tendo dificuldades para gerar uma resposta no
                momento. Por favor, tente novamente mais tarde."""
            )

    last_codes[:] = state.codes
    last_htmls[:] = state.documents

    history.extend([HumanMessage(content=state.question), response])
    history[:] = trim_messages(
        messages=history,
        max_tokens=LIMITE_TOKENS_INPUT * 0.7,
        token_counter=generate.llm.get_num_tokens_from_messages,
    )

    return {
        "response": response.content,
    }


def decide_need_documentation(state: GraphState) -> bool:
    """Verifica necessidade de recuperar docs para responder ao questionamento do user.

    Parameters
    ----------
        state (GraphState): estado do aplicação

    Returns
    -------
        bool: True se for necessário recuperar mais documentos, False caso contrário
    """
    logger.info("")
    logger.info("--- DECIDE NEED DOCUMENTATION ---")

    if not last_codes and not last_htmls:
        return True

    try:
        response = verify_documentation.invoke(
            question=state.question,
            code=last_codes,
            documentation=last_htmls,
            history=history,
        )

        logger.info("Justificativa: %s", response.explanation)
    except (httpx.TimeoutException, ServerError) as e:
        logger.exception("Erro:", exc_info=e)
        return True

    return response.answer


def rewrite_question_rag_node(state: GraphState) -> dict[str, Any]:
    """Reescreve a pergunta do usuário para otimizar a busca por documentos.

    Parameters
    ----------
        state (GraphState): estado do aplicação

    Returns
    -------
        dict[str, Any]: node com a pergunta reescrita para otimizar a busca por
        documentos.
    """
    logger.info("")
    logger.info("--- REWRITE QUESTION RAG ---")

    if history:
        return {
            "rewrited_question": rewrite_question.invoke(
                question=state.question, history=history
            ).content
        }

    return {"rewrited_question": state.question}


def reload_rags_memory_node(_: GraphState) -> dict[str, Any]:
    """Recarrega as RAGS do prompt anterior.

    Parameters
    ----------
        _ (GraphState): estado do aplicação

    Returns
    -------
        dict[str, Any]: node com os códigos e documentos HTML mais recentes
    """
    logger.info("")
    logger.info("--- RELOAD RAGS MEMORY ---")

    return {"codes": last_codes, "documents": last_htmls}


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
    logger.info("")
    logger.info("--- DECIDE NEED CODE ---")

    if not state.documents:
        return True

    try:
        response = verify_code.invoke(
            question=state.question, documentation=state.documents
        )
    except (httpx.TimeoutException, ServerError) as e:
        logger.exception("Erro:", exc_info=e)

        return False
    else:
        return response.answer


@dataclass
class Application:
    """Classe que representa a aplicação."""

    _workflow: StateGraph = field(default_factory=lambda: StateGraph(GraphState))

    _app: CompiledStateGraph = field(init=False)

    def __post_init__(self) -> None:
        """Inicializa propriedades do objeto."""
        logger.info("Executando __post_init__")

        self._add_nodes()
        self._add_nodes_sequence()
        self._add_conditional()

        self._app = self._workflow.compile()

    def _add_conditional(self) -> None:
        logger.info("Adicionando condições de rotas")

        self._workflow.add_conditional_edges(
            GRADE_HTML_DOCUMENTS,
            decide_need_code,
            {True: RETRIEVER_PYTHON, False: GENERATE},
        )

        self._workflow.add_conditional_edges(
            START,
            decide_need_documentation,
            {True: REWRITE_QUESTION, False: RELOAD_RAGS_MEMORY},
        )

    def _add_nodes(self) -> None:
        logger.info("Adicionando nodes ao graph")

        self._workflow.add_node(RELOAD_RAGS_MEMORY, reload_rags_memory_node)
        self._workflow.add_node(REWRITE_QUESTION, rewrite_question_rag_node)
        self._workflow.add_node(RETRIEVER_HTML, retriever_html_node)
        self._workflow.add_node(RETRIEVER_PYTHON, retriever_python_node)
        self._workflow.add_node(GRADE_HTML_DOCUMENTS, grade_html_node)
        self._workflow.add_node(GRADE_PYTHON_DOCUMENTS, grade_python_node)
        self._workflow.add_node(GENERATE, generate_node)

    def _add_nodes_sequence(self) -> None:
        logger.info("Adicionando sequencia do nodes do graph")

        self._workflow.add_edge(REWRITE_QUESTION, RETRIEVER_HTML)
        self._workflow.add_edge(RETRIEVER_HTML, GRADE_HTML_DOCUMENTS)
        self._workflow.add_edge(RETRIEVER_PYTHON, GRADE_PYTHON_DOCUMENTS)
        self._workflow.add_edge(GRADE_PYTHON_DOCUMENTS, GENERATE)
        self._workflow.add_edge(RELOAD_RAGS_MEMORY, GENERATE)
        self._workflow.add_edge(GENERATE, END)

    def generate_image(self) -> None:
        """Gera uma imagem do graph do aplicação.

        A imagem é salva em um arquivo chamado "graph.png"
        na pasta atual.
        """
        logger.info("Gerando imagem do graph")

        with Path("graph.png").open("wb") as f:
            f.write(self._app.get_graph().draw_mermaid_png())

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
        return self._app.invoke(input={"question": question})["response"]
