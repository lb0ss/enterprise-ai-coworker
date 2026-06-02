from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import TOOLS

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# create_react_agent builds the full ReAct loop graph:
# START → llm_node → tool_node → llm_node → ... → END
agent = create_react_agent(llm, TOOLS)
