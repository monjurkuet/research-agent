import asyncio
import os
import langchain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchResults
from crawl4ai import AsyncWebCrawler
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import JsonOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import ollama
from config import Config
from schema import ResearchPlan, ResearchState


# --- 0. DEBUGGING & VISIBILITY ---
langchain.debug = True  # Shows every internal LLM call in the console

def is_connection_error(exception):
    # Retry if it's an Ollama ResponseError (like your 500 error) or a connection timeout
    return isinstance(exception, (ollama.ResponseError, ConnectionError, asyncio.TimeoutError))

def log(msg):
    """Force immediate print to console without buffering."""
    print(f"\n[AGENT LOG] {msg}", flush=True)

# --- 1. SETUP ---
log("Initializing Models...")
llm = ChatOllama(model=Config.MODEL_NAME, base_url=Config.BASE_URL, format="json", temperature=0)
embeddings = OllamaEmbeddings(model=Config.EMBED_MODEL)
search_tool = DuckDuckGoSearchResults(output_format="list")
parser = JsonOutputParser(pydantic_object=ResearchPlan)

# --- 2. THE PARALLEL RESEARCHER (With Timeouts) ---
sem = asyncio.Semaphore(3)

async def scrape_with_limit(url: str, crawler: AsyncWebCrawler):
    async with sem:
        try:
            log(f"📥 Starting Scrape: {url}")
            # Strict 20s timeout so we don't hang forever
            result = await asyncio.wait_for(crawler.arun(url=url), timeout=20.0)
            log(f"✅ Finished Scrape: {url}")
            return result.markdown if result.success else f"Error: Failed to load {url}"
        except asyncio.TimeoutError:
            log(f"⚠️ Timeout on {url} - Skipping...")
            return f"Error: Timeout on {url}"
        except Exception as e:
            log(f"❌ Error on {url}: {str(e)}")
            return f"Error: {str(e)}"

async def research_node(state: ResearchState):
    log("Entering Research Node...")
    all_urls = []
    for q in state.queries:
        log(f"🔎 DuckDuckGo Searching: {q}")
        search_results = search_tool.invoke(q)
        urls = [res['link'] for res in search_results[:2]] # Limit to 2 per query for speed
        all_urls.extend(urls)
    
    unique_urls = list(set(all_urls))
    log(f"🚀 Found {len(unique_urls)} total URLs. Launching Parallel Scraper...")
    
    async with AsyncWebCrawler() as crawler:
        tasks = [scrape_with_limit(url, crawler) for url in unique_urls]
        scraped_contents = await asyncio.gather(*tasks)
    
    log("🌐 Web Research Complete.")
    return {"scraped_data": scraped_contents}
# This decorator tells the function: 
# "If you fail, wait 2^x seconds and try again, up to 5 times."
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ollama.ResponseError, ConnectionError)),
    before_sleep=lambda retry_state: log(f"⏳ Connection lost. Retrying in {retry_state.next_action.sleep}s...")
)
def robust_llm_call(prompt, is_json=True):
    # Use the appropriate LLM instance based on the format needed
    target_llm = llm if is_json else ChatOllama(model=Config.MODEL_NAME, base_url=Config.BASE_URL)
    return target_llm.invoke(prompt)

# --- UPDATED NODES USING THE ROBUST CALL ---
# --- 3. OTHER NODES ---
def planner_node(state: ResearchState):
    log("🧠 Thinking... Generating Research Plan.")
    prompt = f"Generate a JSON research plan (queries and reasoning) for: {state.task}"
    raw_res = llm.invoke(prompt)
    try:
        plan_data = parser.parse(raw_res.content)
        log(f"💡 Plan Created: {len(plan_data['queries'])} queries ready.")
        return {"queries": plan_data['queries']}
    except:
        log("⚠️ JSON Parse Error. Using fallback query.")
        return {"queries": [state.task]}

def rag_node(state: ResearchState):
    log("🧠 Checking H: Drive Local Knowledge...")
    db = Chroma(persist_directory=Config.DB_DIR, embedding_function=embeddings)
    local_hits = db.similarity_search(state.task, k=2)
    local_text = "\n".join([h.page_content for h in local_hits])
    state.scraped_data.append(f"LOCAL DATA:\n{local_text}")
    return {"scraped_data": state.scraped_data}

def synthesizer_node(state: ResearchState):
    log("✍️  Writing Final Report...")
    text_llm = ChatOllama(model=Config.MODEL_NAME, base_url=Config.BASE_URL, temperature=0.7)
    context = "\n\n".join(state.scraped_data)
    report = text_llm.invoke(f"Data:\n{context}\n\nTask: {state.task}. Write report:")
    return {"final_report": report.content}

# --- 4. BUILD THE GRAPH ---
builder = StateGraph(ResearchState)
builder.add_node("planner", planner_node)
builder.add_node("researcher", research_node)
builder.add_node("rag", rag_node)
builder.add_node("synthesizer", synthesizer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "rag")
builder.add_edge("rag", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()

async def main():
    try:
        goal = input("\n[Deep Research Mission]: ")
        log("Execution Started.")
        
        # We use astream to catch every state change
        async for event in graph.astream({"task": goal}, stream_mode="values"):
            if "final_report" in event and event["final_report"]:
                print(f"\n\n{'='*30}\n✅ FINAL REPORT\n{'='*30}")
                print(event["final_report"])
                
    except Exception as e:
        log(f"CRITICAL SYSTEM FAILURE: {str(e)}")
        import traceback
        traceback.print_exc()
    
    input("\n[FINISHED] Press Enter to exit...")

if __name__ == "__main__":
    asyncio.run(main())