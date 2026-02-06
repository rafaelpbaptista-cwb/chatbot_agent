"""Pacote principal do projeto."""

from dotenv import load_dotenv

from .chain.create_chains import create_generate, create_rag_grader, create_retriever

load_dotenv()
retriever = create_retriever()
rag_grader = create_rag_grader()
generate = create_generate()

__all__ = ["generate", "rag_grader", "retriever"]
