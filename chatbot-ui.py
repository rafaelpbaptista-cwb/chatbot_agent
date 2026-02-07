"""Interface gráfica para acessar as funcinalidades da aplicação."""

import logging

import panel as pn
from panel.widgets.input import FileInput, TextInput

from chatbot_agent.graph import Application

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
chat_bot = Application()


def get_response(contents: str, _: TextInput, __: FileInput) -> str:
    """
    Função que retorna a resposta da aplicação para uma pergunta.

    Parameters
    ----------
    contents : Any
        Conteúdo da pergunta

    Returns
    -------
    str
        Resposta da aplicação para a pergunta
    """
    return chat_bot.invoke(question=contents)


chat_ui = pn.chat.ChatInterface(callback=get_response)
chat_ui.send("Pergunte-me qualquer coisa!", user="Assistant", respond=False)
chat_ui.show()
