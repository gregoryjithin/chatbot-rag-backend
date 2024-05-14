import os
import random
import time
import uuid
from pydantic import BaseModel
from typing import Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Data model for the request body


class QuestionRequest(BaseModel):
    question: str

# Data model for the response


class ApiResponse(BaseModel):
    author: Dict[str, str]
    createdAt: int
    id: str
    status: str
    text: str
    type: str


class ConversationalRetrievalService:
    def __init__(self, api_key, text_file_path):
        # Set the OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key

        # Initialize components
        # llm_model = "gpt-3.5-turbo"
        llm_model = "gpt-4-turbo"
        llm = ChatOpenAI(temperature=0.7, model_name=llm_model)

        # Load and split the text document
        loader = TextLoader(file_path=text_file_path, encoding="utf-8")
        data = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        split_data = text_splitter.split_documents(data)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(split_data, embedding=embeddings)

        # Create conversational chain with memory
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            chain_type="stuff"
        )

        # Default response options
        self.default_responses = [
            "I'm not sure about that.",
            "I don't have enough information on that topic.",
            "I don't know the answer to that question."
        ]

        self.prompt = """You are HR Manager of Trenser Technology Solutions, and your role is to assist employees with their inquiries. Please utilize the provided context to respond to their questions accurately and professionally.

        Question: <question>

        Please remember, it is essential to provide only helpful answers based on the context you have. If you are uncertain about the answer or if the context is missing or unclear, do not attempt to invent an answer. Instead, kindly reply with: "I can't assist with the question. Please reach out to the HR team."
        If you find the context doesn't match with the question, just reply with "I can't assist with the question. Please reach out to the HR team."
        Your primary goal is to offer reliable and relevant information to our employees with concise responses.
        """

    def get_response(self, question):
        result = self.conversation_chain.invoke({"question": self.prompt.replace("<question>", question)})
        if not result or "answer" not in result:
            return random.choice(self.default_responses)
        answer = result["answer"]
        if not answer.strip():
            return random.choice(self.default_responses)
        return answer

    def build_response(self, response_text):
        response_id = str(uuid.uuid4())
        return ApiResponse(
            author={
                "firstName": "H",
                "id": "4c2307ba-3d40-442f-b1ff-b271f63904ca",
                "lastName": "R",
                "imageUrl": "http://localhost:8000/image/pfp"
            },
            createdAt=int(time.time() * 1000),
            id=response_id,
            status="seen",
            text=response_text,
            type="text",
        )
