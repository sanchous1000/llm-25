import operator
from typing import Annotated, TypedDict
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import yaml

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# Load config
with open("scripts/configs/config_baseline.yaml") as f:
    config = yaml.safe_load(f)

agent_config = config["agent"]

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)
langfuse_handler = CallbackHandler()

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
model = SentenceTransformer(agent_config["embedding_model"])
collection_name = agent_config["collection_name"]
retrieval_limit = agent_config["retrieval_limit"]
system_prompt = agent_config["system_prompt"]


@tool
def remember(query: str) -> str:
    """Поиск релевантной информации в базе знаний о D&D 5e SRD на английском языке"""
    points = client.query_points(
        collection_name=collection_name,
        query=model.encode(query),
        with_payload=True,
        limit=retrieval_limit,
    ).points

    return [{"text": point.payload['text'], "score": point.score} for point in points]


llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-5-nano"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

tools = [remember]
agent_executor = create_react_agent(llm, tools).with_config(
    {"callbacks": [langfuse_handler]}
)

messages = [SystemMessage(content=system_prompt)]


def send_message(query: str):
    global messages
    result = agent_executor.invoke(
        {
            "messages": [
                *messages,
                HumanMessage(content=query),
            ]
        }
    )
    messages = result["messages"]
    return messages[-1]


def get_messages() -> list[BaseMessage]:
    return messages