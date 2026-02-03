"""Pacote principal do projeto."""

from dotenv import load_dotenv

from .chain.create_chains import create_retriever

load_dotenv()
retriever = create_retriever()
