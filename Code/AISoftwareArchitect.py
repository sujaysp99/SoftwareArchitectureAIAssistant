from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

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
        self._apiKey = None
        self.vector_db = None
        self.embeddings = None

    def set_API_key(self, apiKey):
        print("Setting OpenAI API Key")
        self._apiKey = apiKey
        #os.environ['OPENAI_API_KEY'] = apiKey

    def load_pdf_document(self, file_path):
        print("Loading PDF File")
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def load_embeddings(self, api_key = None):
        print("Loading Embeddings")
        self.embeddings = SentenceTransformerEmbeddings(model_name="intfloat/e5-large-v2")
        # if type(api_key) == type(None):
        #     self.embeddings = OpenAIEmbeddings()
        # else:
        #     self.embeddings = OpenAIEmbeddings(api_key=api_key)

    def create_vector_DB(self, documents, embeddings, persist_dir="./vector_db", ):
        print("Creating Vector Database")
        self.vector_db = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)

    def load_and_store_documents(self, file_path):
        if type(self._apiKey) == type(None):
            raise Exception("OpenAI API Key not set")
        
        else:
            pdf_doc = self.load_pdf_document(file_path)
            if type(self.embeddings) == type(None):
                self.load_embeddings(self._apiKey)

            self.create_vector_DB(pdf_doc, self.embeddings)
            print("Finished loading and storing document")