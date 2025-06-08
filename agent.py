import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

from langgraph.graph import StateGraph
from langchain.tools import Tool as LangTool
from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)

# ----------- Define custom tool: ExcelToTextTool -----------
def excel_to_markdown(excel_path: str, sheet_name: Optional[str] = None) -> str:
    file_path = Path(excel_path).expanduser().resolve()
    if not file_path.is_file():
        return f"Error: Excel file not found at {file_path}"

    try:
        sheet: Union[str, int] = (
            int(sheet_name) if sheet_name and sheet_name.isdigit() else sheet_name or 0
        )
        df = pd.read_excel(file_path, sheet_name=sheet)
        if hasattr(df, "to_markdown"):
            return df.to_markdown(index=False)
        return tabulate(df, headers="keys", tablefmt="github", showindex=False)
    except Exception as e:
        return f"Error reading Excel file: {e}"

excel_tool = LangTool.from_function(
    name="excel_to_text",
    func=excel_to_markdown,
    description="Reads an Excel file and returns a Markdown table. Inputs: 'excel_path' (str), 'sheet_name' (str, optional).",
)

# ----------- Initialize Other Tools -----------
duckduckgo_tool = DuckDuckGoSearchResults()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ----------- Define LangGraph -----------
def call_tool_or_llm(state: dict) -> dict:
    question = state["question"]

    # Simple heuristic (or you can use langchain agents)
    if "excel" in question.lower():
        result = excel_tool.run({"excel_path": "./data.xlsx", "sheet_name": "0"})
    elif "wikipedia" in question.lower():
        result = wiki_tool.run(question)
    else:
        result = llm.invoke(question).content

    return {"question": question, "result": result}


# ----------- Define LangGraph State Machine -----------
workflow = StateGraph(dict)
workflow.add_node("processor", RunnableLambda(call_tool_or_llm))
workflow.set_entry_point("processor")
workflow.set_finish_point("processor")

graph = workflow.compile()

# ----------- Define Agent class wrapper -----------
class GaiaAgent:
    def __init__(self):
        print("GaiaAgent (LangGraph) initialized with LLM and tools.")

    def __call__(self, task_id: str, question: str) -> str:
        print(f"[{task_id}] Received: {question}")
        result = graph.invoke({"question": question})
        answer = result["result"]
        print(f"[{task_id}] Answer: {answer}")
        return answer