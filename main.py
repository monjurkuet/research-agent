import sqlite3
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver # <--- New: Memory Saver

# 1. Setup the Model
model = ChatOllama(model="mistral:latest", base_url="http://localhost:11434", temperature=0)

# 2. Setup the Tools
search = DuckDuckGoSearchRun()
tools = [search]

# 3. Setup the Memory (Persistence)
# This creates (or opens) a local database file
conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

# 4. Create the Agent with Memory
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory, # <--- Telling the agent to use our DB
    system_prompt="You are a helpful Bitcoin researcher with a long-term memory."
)

def run_research():
    # A 'thread_id' is like a unique ID for a specific conversation
    config = {"configurable": {"thread_id": "bitcoin_project_1"}}
    
    print("--- 🚀 Stateful Bitcoin Research Agent Active ---")
    
    while True:
        query = input("\nAsk me something (or type 'exit'): ")
        if query.lower() == 'exit': break
        
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config # <--- Passing the config saves/loads the history
        )
        
        print(f"\n📊 AGENT: {response['messages'][-1].content}")

if __name__ == "__main__":
    run_research()