from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent

import os

import fitz
from PIL import Image

class AISoftwareArchitect:
    def __init__(self):
        self._apiKey = ""
        self.vector_db = None

    def set_API_key(self, apiKey):
        print("Setting OpenAI API Key")
        self._apiKey = apiKey

    def load_pdf_document(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def load_embeddings(self):
        return OpenAIEmbeddings()

    def create_vector_DB(self, documents, embeddings, persist_dir="./vector_db", ):
        self.vector_db = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)