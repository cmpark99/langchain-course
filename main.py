from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

load_dotenv()


llm = ChatAnthropic(model="claude-opus-4-8")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)  

def main():
    print("Search Agent Project")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 job postings for an AI engineer using langchain in the bay area on linkedin and list their details")})
    print(result)



if __name__ == "__main__":
    main()
