#!/Users/rlm/Desktop/Code/vibe-code-benchmark/.venv/bin/python

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def get_model(model: str = 'deepseek-r1:7b', provider: str = 'local'):
    if (provider == 'local'):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0.8)
        return llm
    elif (provider == 'llama'):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model='llama3.1', temperature=0.8)
        return llm
    elif (provider == 'aws'):
        from langchain_aws import ChatBedrockConverse
        import boto3
        access_key = os.getenv('ACCESS_KEY')
        secret_key = os.getenv('SECRET_KEY')
        bedrock_client = boto3.client('bedrock-runtime',
                                      region_name='us-east-1',
                                      aws_access_key_id=access_key,
                                      aws_secret_access_key=secret_key)
        llm = ChatBedrockConverse(client=bedrock_client,
                                  model=model,
                                  temperature=0.8)
        return llm


def get_embeddings(model: str = 'deepseek-r1:7b', provider: str = 'local'):
    if (provider == 'local'):
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=model)
        return embeddings
    elif (provider == 'aws'):
        from langchain_aws import BedrockEmbeddings
        import boto3
        access_key = os.getenv('ACCESS_KEY')
        secret_key = os.getenv('SECRET_KEY')
        bedrock_client = boto3.client('bedrock-runtime',
                                      region_name='us-east-1',
                                      aws_access_key_id=access_key,
                                      aws_secret_access_key=secret_key)
        embeddings = BedrockEmbeddings(client=bedrock_client, model_id=model)
        return embeddings


# Define common path to the repo locally
PATH = "/Users/rishirajkalita/Desktop/test-mcp/"

# Create an MCP server
mcp = FastMCP("rishigraph-Docs-MCP-Server")


# Add a tool to query the rishigraph documentation
@mcp.tool()
def rishigraph_query_tool(query: str):
    """
    Query the RishiGraph documentation using a retriever.
    
    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    loader = TextLoader(f"{PATH}rishi_full.txt")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = get_embeddings(model='amazon.titan-embed-text-v2:0',
                                provider='aws')
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    relevant_docs = retriever.invoke(query)

    return relevant_docs


# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://rishigraph/full")
def get_all_rishigraph_docs() -> str:
    """
    Get all the rishigraph documentation. Returns the contents of the file llms_full.txt,
    which contains a curated set of rishigraph documentation (~300k tokens). This is useful
    for a comprehensive response to questions about rishigraph.

    Args: None

    Returns:
        str: The contents of the rishigraph documentation
    """

    # Local path to the rishigraph documentation
    doc_path = PATH + "rishi_full.txt"
    try:
        with open(doc_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
