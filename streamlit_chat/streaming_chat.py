import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from langchain.agents.agent import AgentExecutor
load_dotenv()

def get_model(model: str = 'deepseek-r1:7b', provider: str = 'local'):
    if (provider == 'local'):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0.8)
        return llm
    elif (provider=='llama'):
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


def tavily_client():
    search = TavilySearchResults(max_results=2)
    return search

def get_agent():
    prompt = hub.pull("hwchase17/openai-functions-agent")
    model=get_model(provider='aws', model='anthropic.claude-3-sonnet-20240229-v1:0')
    search=tavily_client()
    tools=[search]
    agent = create_tool_calling_agent(model,tools,prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor




st.title("Streaming Chat")

# Set a default model
if "model" not in st.session_state:
    st.session_state["model"] = get_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = st.session_state["model"].stream({"input":prompt})
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})