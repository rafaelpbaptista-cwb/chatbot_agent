"""Script contendo definição dos documentos estruturados.

Esses documentos são utilizados na resposta dos chains.
"""

from pydantic import BaseModel, Field


class RagGraderAnswer(BaseModel):
    """Classe estruturada usada para retornar resposta do RagGrader."""

    Score: int = Field(description="Relevance score between 0 and 1")
    Explaination: str = Field(description="Explanation of relevance score")
