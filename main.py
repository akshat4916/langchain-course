from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field 
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """ Schema for a source used by the agent. """
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """ Schema for the agent's response with answers and sources. """
    answer: str = Field(description="The agent's answer to the query.")
    sources: List[Source] = Field(default_factory=list, description="A list of sources used to generate the answer.")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format = AgentResponse)

def main():

    print("Hello from langchain-course!")
    # result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job postings for an AI engineer using langchain in the bay area on Linkedin")})
    print("Result:", result)


if __name__ == "__main__":
    main()
