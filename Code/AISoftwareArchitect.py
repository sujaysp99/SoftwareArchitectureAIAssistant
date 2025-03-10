from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

from langchain.tools import BaseTool
from googlesearch import search
import requests
from bs4 import BeautifulSoup

import os

import fitz
from PIL import Image

class GoogleSearchTool(BaseTool):
    name:str = "GoogleSearch"
    description:str = (
        "Performs a Google search using the googlesearch library and returns a "
        "formatted summary of the top results including page titles and URLs."
    )

    def _run(self, query: str) -> str:
        results = []
        # Use googlesearch to get the top 3 results
        #for url in search(query, num=3, stop=3, pause=2.0):
        for url in search(query, num_results=5):
            page_text = ""
            try:
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
                paragraphs = soup.find_all("p")
                page_text = "\n".join([para.get_text() for para in paragraphs])
            except Exception:
                title = "No title"
            results.append(f"Title: {title}\nURL: {url}\nContent: {page_text[:50000]}")
        if results:
            return "\n\n".join(results)
        else:
            return "No results found for your query."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Asynchronous version is not implemented.")
    

class AISoftwareArchitect:
    def __init__(self):
        self._apiKey = None
        self._vector_db = None
        self._embeddings = None
        self._google_search_tool = GoogleSearchTool()
        self.history = []
        self._agent = None
        try:
            self.__initializeAgent()
            pass
        except:
            pass

    def set_API_key(self, apiKey):
        print("Setting OpenAI API Key")
        self._apiKey = apiKey
        #os.environ['OPENAI_API_KEY'] = apiKey
        self.__initializeAgent(self._apiKey)

    def __load_pdf_document(self, file_path):
        print("Loading PDF File")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)
    
    def __load_embeddings(self, api_key=None):
        print("Loading Embeddings")
        self._embeddings = SentenceTransformerEmbeddings(model_name="intfloat/e5-large-v2")
        # if type(api_key) == type(None):
        #     self.embeddings = OpenAIEmbeddings()
        # else:
        #     self.embeddings = OpenAIEmbeddings(api_key=api_key)

    def __create_vector_DB(self, documents, embeddings, persist_dir="./vector_db", ):
        print("Creating Vector Database")
        self._vector_db = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)

    def load_and_store_documents(self, file_path):
        if type(self._apiKey) == type(None):
            raise Exception("OpenAI API Key not set")
        
        else:
            pdf_doc = self.__load_pdf_document(file_path)
            if type(self._embeddings) == type(None):
                self.__load_embeddings(self._apiKey)

            self.__create_vector_DB(pdf_doc, self._embeddings)

            self.__initializeAgent(self._apiKey)
            print("Finished loading and storing document")

    def __initializeAgent(self, api_key=None):
        if type(self._apiKey) == type(None):
            #model = ChatOpenAI(temperature=0.1)
            model = ChatCohere()
        else:
            #model = ChatOpenAI(temperature=0.1, api_key=api_key)
            model = ChatCohere(cohere_api_key=api_key)

        tools = []
        
        search_tool = Tool(
            name=self._google_search_tool.name,
            func=self._google_search_tool._run,
            description=self._google_search_tool.description
        )

        tools.append(search_tool)

        if type(self._vector_db) != type(None):
            # retrieval_chain = RetrievalQA.from_llm(llm=model, retriever=self._vector_db.as_retriever(search_kwargs={"k": 10}))

            # document_retrieval_tool = Tool(
            #     name="DocumentRetrieval",
            #     func=retrieval_chain.run,
            #     description="Retrieves relevant documents and document content based on user queries."
            # )

            # tools.append(document_retrieval_tool)

            @tool(response_format="content_and_artifact")
            def retrieve(query: str):
                """Retrieve information related to a query."""
                retrieved_docs = self._vector_db.similarity_search(query, k=2)
                serialized = "\n\n".join(
                    (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs

            tools.append(retrieve)

        memory = ConversationBufferMemory()

        prefix = """You are a helpful Software Architect AI assistant. 
            You should answer the questions as accurately as possible.
            You have a vast knowledge about software architecture concepts 
            and can easily search online. You have the capability to extract 
            sections from uploaded software architecture documents or 
            document templates and use that to formulate answers or 
            write a new document similar to the uploaded document if asked. 
            You will never use plagiarized content and will always focus on originality.
            Remember, you are a software architect ai assistant.
            """

        self._agent = initialize_agent(
            tools,
            model,
            agent="zero-shot-react-description",
            #agent="self-ask-with-search",
            memory=memory,
            prefix=prefix,
            verbose=True
        )

        print("Initialized Agent")

    def submit_request(self, user_input, history):
        for human_message, ai_message in history:
            self._agent.memory.chat_memory.add_user_message(human_message)
            self._agent.memory.chat_memory.add_ai_message(ai_message)

        response = self._agent.run(user_input)

        return response
    
    def chat_with_ai_architect(self, user_input, history):
        response = self.submit_request(user_input, self.history)
        self.history.append((user_input, response))
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        return history