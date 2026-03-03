from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():

    print("Hello from langchain-course!")
    # result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})

    result = agent.invoke({"messages": HumanMessage(content="What is the latest news on Apple Inc.?")})
    print("Result:", result)


if __name__ == "__main__":
    main()
