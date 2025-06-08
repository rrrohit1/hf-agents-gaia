import os
import re
from pathlib import Path
from typing import Optional, Union, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import requests
import tempfile

import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

from langgraph.graph import StateGraph, END
from langchain.tools import Tool as LangTool
from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage

import whisper
from pathlib import Path
from PIL import Image

# Load environment variables
load_dotenv()

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
QUESTIONS_URL = f"{DEFAULT_API_URL}/questions"
SUBMIT_URL = f"{DEFAULT_API_URL}/submit"
FILE_PATH = f"{DEFAULT_API_URL}/files/"

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-pro"),
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ----------- Enhanced State Management -----------
from typing import TypedDict

class AgentState(TypedDict):
    """Enhanced state tracking for the agent - using TypedDict for LangGraph compatibility"""
    question: str
    original_question: str
    conversation_history: List[Dict[str, str]]
    selected_tools: List[str]
    tool_results: Dict[str, Any]
    final_answer: str
    current_step: str
    error_count: int
    max_errors: int

class AgentStep(Enum):
    ANALYZE_QUESTION = "analyze_question"
    SELECT_TOOLS = "select_tools"
    EXECUTE_TOOLS = "execute_tools"
    SYNTHESIZE_ANSWER = "synthesize_answer"
    ERROR_RECOVERY = "error_recovery"
    COMPLETE = "complete"

# ----------- Helper Functions -----------
def initialize_state(question: str) -> AgentState:
    """Initialize agent state with default values"""
    return {
        "question": question,
        "original_question": question,
        "conversation_history": [],
        "selected_tools": [],
        "tool_results": {},
        "final_answer": "",
        "current_step": "start",
        "error_count": 0,
        "max_errors": 3
    }

# ----------- Enhanced Tools -----------
def excel_to_markdown(excel_path: str, sheet_name: Optional[str] = None) -> str:
    """Enhanced Excel tool with better error handling"""
    try:
        file_path = Path(excel_path).expanduser().resolve()
        if not file_path.is_file():
            return f"Error: Excel file not found at {file_path}"

        sheet: Union[str, int] = (
            int(sheet_name) if sheet_name and sheet_name.isdigit() else sheet_name or 0
        )
        df = pd.read_excel(file_path, sheet_name=sheet)
        
        # Add metadata about the dataframe
        metadata = f"Dataset Info: {len(df)} rows, {len(df.columns)} columns\n"
        metadata += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        
        if hasattr(df, "to_markdown"):
            return metadata + df.head(10).to_markdown(index=False) + f"\n\n(Showing first 10 rows of {len(df)})"
        return metadata + tabulate(df.head(10), headers="keys", tablefmt="github", showindex=False)
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"

def image_file_info(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        return f"Image file info: format={img.format}, size={img.size}, mode={img.mode}"
    except Exception as e:
        return f"Error reading image: {e}"

def audio_file_info(audio_path: str) -> str:

    try:
        model = whisper.load_model("turbo")  # or "small", "medium", "large"
        result = model.transcribe(audio_path)
        return f"Audio file transcription: {result}"
    except Exception as e:
        return f"Error transcribing MP3: {str(e)}"
    # Placeholder - you could add actual audio metadata reading


def code_file_read(code_path: str) -> str:
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading code file: {e}"


def extract_file_path(question: str) -> Optional[str]:
    """Extract file path from question using regex"""
    patterns = [
        r"file\s+(?:at\s+)?['\"]?([^'\">\s]+\.xlsx?)['\"]?",
        r"excel\s+(?:file\s+)?['\"]?([^'\">\s]+\.xlsx?)['\"]?",
        r"['\"]?([^'\">\s]*\.xlsx?)['\"]?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

# Initialize tools
excel_tool = LangTool.from_function(
    name="excel_to_text",
    func=excel_to_markdown,
    description="Reads an Excel file and returns a Markdown table with metadata. Inputs: 'excel_path' (str), 'sheet_name' (str, optional).",
)

image_tool = LangTool.from_function(
    name="image_file_info",
    func=image_file_info,
    description="Reads an image file and returns its metadata."
)

audio_tool = LangTool.from_function(
    name="audio_file_info",
    func=audio_file_info,
    description="Processes an audio file and returns transcription."
)

code_tool = LangTool.from_function(
    name="code_file_read",
    func=code_file_read,
    description="Reads a code file and returns the code as a string file."
)


duckduckgo_tool = DuckDuckGoSearchResults(num_results=5)
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Tool registry
AVAILABLE_TOOLS = {
    "excel": excel_tool,
    "search": duckduckgo_tool,
    "wikipedia": wiki_tool,
    "image": image_tool,
    "audio": audio_tool,
    "code": code_tool,
}

# ----------- Intelligent Tool Selection -----------
def analyze_question(state: AgentState) -> AgentState:
    """Analyze the question and determine context"""
    question = state["question"].lower()
    
    # Use LLM to analyze the question
    analysis_prompt = f"""
    Analyze this question and determine what tools might be needed:
    Question: {state["question"]}
    
    Available tools:
    1. excel - for reading Excel/CSV files
    2. search - for web search queries
    3. wikipedia - for encyclopedic information
    4. image - for generating or analyzing images
    5. audio - for audio transcription or processing (e.g., transcribing mp3, extracting audio from video)
    6. code - for reading, summarizing, or executing code

    Respond with:
    1. Primary intent (data_analysis, information_search, general_knowledge, audio_transcription, video_analysis)
    2. Suggested tools (comma-separated list)
    3. Key entities or file paths mentioned
    
    Format: INTENT: <intent> | TOOLS: <tools> | ENTITIES: <entities>
    """
    
    try:
        response = llm.invoke(analysis_prompt).content
        state["conversation_history"].append({"role": "analysis", "content": response})
        state["current_step"] = AgentStep.SELECT_TOOLS.value
    except Exception as e:
        state["error_count"] += 1
        state["conversation_history"].append({"role": "error", "content": f"Analysis failed: {e}"})
        state["current_step"] = AgentStep.ERROR_RECOVERY.value
    
    return state

def select_tools(state: AgentState) -> AgentState:
    question = state["question"].lower()
    selected_tools = []

    if any(keyword in question for keyword in ["excel", "csv", "spreadsheet", ".xlsx"]):
        selected_tools.append("excel")
    if any(keyword in question for keyword in ["search", "find", "look up", "current", "recent", "news"]):
        selected_tools.append("search")
    if any(keyword in question for keyword in ["wikipedia", "history", "definition", "who is"]):
        selected_tools.append("wikipedia")
    if any(keyword in question for keyword in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]):
        selected_tools.append("image")
    if any(keyword in question for keyword in [".mp3", ".wav", ".ogg"]):
        selected_tools.append("audio")
    if any(keyword in question for keyword in [".py", ".ipynb"]):
        selected_tools.append("code")

    if not selected_tools:
        selected_tools.append("search")

    state["selected_tools"] = selected_tools
    state["current_step"] = AgentStep.EXECUTE_TOOLS.value
    return state


def execute_tools(state: AgentState) -> AgentState:
    results = {}

    # Check if downloaded file info is appended in question text
    file_path = None
    downloaded_file_marker = "A file was downloaded for this task and saved locally at:"
    if downloaded_file_marker in state["question"]:
        # Extract file path from question text
        lines = state["question"].splitlines()
        for i, line in enumerate(lines):
            if downloaded_file_marker in line:
                # The file path should be on the next line
                if i + 1 < len(lines):
                    file_path_candidate = lines[i + 1].strip()
                    if Path(file_path_candidate).exists():
                        file_path = file_path_candidate
                break

    for tool_name in state["selected_tools"]:
        try:
            if tool_name == "excel" and file_path:
                # Run excel tool on detected file path
                result = AVAILABLE_TOOLS["excel"].run({"excel_path": file_path, "sheet_name": None})
            elif tool_name == "image" and file_path:
                result = AVAILABLE_TOOLS["image_file_info"].run(file_path)
            elif tool_name == "audio" and file_path:
                result = AVAILABLE_TOOLS["audio_file_info"].run(file_path)
            elif tool_name == "code" and file_path:
                result = AVAILABLE_TOOLS["code_file_preview"].run(file_path)
            else:
                result = AVAILABLE_TOOLS[tool_name].run(state["question"])

            results[tool_name] = result
        except Exception as e:
            results[tool_name] = f"Error using {tool_name}: {str(e)}"
            state["error_count"] += 1

    state["tool_results"] = results
    state["current_step"] = AgentStep.SYNTHESIZE_ANSWER.value
    return state


def synthesize_answer(state: AgentState) -> AgentState:
    """Synthesize final answer from tool results"""
    if not state["tool_results"]:
        state["final_answer"] = "I couldn't find any relevant information to answer your question."
        state["current_step"] = AgentStep.COMPLETE.value
        return state
    
    # Create synthesis prompt
    synthesis_prompt = f"""You are given a factual question and some tool results to help answer it.

        Question: {state["question"]}

        Use ONLY the information in the tool results below to generate the answer.

        Output Requirements:
        - Your answer MUST be complete, precise, and follow the output format required by the question.
        - DO NOT include explanations, introductions, or reasoning.
        - Respond with only the answer.

        Tool Results:
        {chr(10).join([f"{tool}: {result}" for tool, result in state["tool_results"].items()])}
        """
        
    try:
        response = llm.invoke(synthesis_prompt).content
        state["final_answer"] = response
        state["current_step"] = AgentStep.COMPLETE.value
    except Exception as e:
        state["error_count"] += 1
        state["final_answer"] = f"Error synthesizing answer: {e}"
        state["current_step"] = AgentStep.ERROR_RECOVERY.value
    
    return state

def error_recovery(state: AgentState) -> AgentState:
    """Handle errors and attempt recovery"""
    if state["error_count"] >= state["max_errors"]:
        state["final_answer"] = "I encountered too many errors and cannot complete this task."
        state["current_step"] = AgentStep.COMPLETE.value
    else:
        # Try with a simpler approach - direct LLM response
        try:
            response = llm.invoke(state["question"]).content
            state["final_answer"] = f"Using direct reasoning: {response}"
            state["current_step"] = AgentStep.COMPLETE.value
        except Exception as e:
            state["final_answer"] = f"All approaches failed: {e}"
            state["current_step"] = AgentStep.COMPLETE.value
    
    return state

# ----------- Enhanced LangGraph Workflow -----------
def route_next_step(state: AgentState) -> str:
    """Route to next step based on current state"""
    step_routing = {
        "start": AgentStep.ANALYZE_QUESTION.value,
        AgentStep.ANALYZE_QUESTION.value: AgentStep.SELECT_TOOLS.value,
        AgentStep.SELECT_TOOLS.value: AgentStep.EXECUTE_TOOLS.value,
        AgentStep.EXECUTE_TOOLS.value: AgentStep.SYNTHESIZE_ANSWER.value,
        AgentStep.SYNTHESIZE_ANSWER.value: AgentStep.COMPLETE.value,
        AgentStep.ERROR_RECOVERY.value: AgentStep.COMPLETE.value,
        AgentStep.COMPLETE.value: END,
    }
    
    return step_routing.get(state["current_step"], END)

# Create workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze_question", RunnableLambda(analyze_question))
workflow.add_node("select_tools", RunnableLambda(select_tools))
workflow.add_node("execute_tools", RunnableLambda(execute_tools))
workflow.add_node("synthesize_answer", RunnableLambda(synthesize_answer))
workflow.add_node("error_recovery", RunnableLambda(error_recovery))

# Set entry point
workflow.set_entry_point("analyze_question")

# Add conditional edges
workflow.add_conditional_edges(
    "analyze_question",
    lambda state: "select_tools" if state["current_step"] == AgentStep.SELECT_TOOLS.value else "error_recovery"
)
workflow.add_edge("select_tools", "execute_tools")
workflow.add_conditional_edges(
    "execute_tools",
    lambda state: "synthesize_answer" if state["current_step"] == AgentStep.SYNTHESIZE_ANSWER.value else "error_recovery"
)
workflow.add_conditional_edges(
    "synthesize_answer",
    lambda state: END if state["current_step"] == AgentStep.COMPLETE.value else "error_recovery"
)
workflow.add_edge("error_recovery", END)

# Compile the graph
graph = workflow.compile()

# ----------- Enhanced Agent Class -----------
class GaiaAgent:
    def __init__(self):
        self.graph = graph
        print("Enhanced GaiaAgent initialized with multi-step reasoning and error recovery.")

    def __call__(self, task_id: str, question: str) -> str:
        print(f"[{task_id}] Processing: {question}")
        
        # Initialize state using our helper function
        question = process_file(task_id, question)
        initial_state = initialize_state(question)
        
        try:
            # Execute the workflow
            result = self.graph.invoke(initial_state)
            
            # Extract information from the result dictionary
            answer = result.get("final_answer", "No answer generated")
            selected_tools = result.get("selected_tools", [])
            conversation_history = result.get("conversation_history", [])
            
            print(f"[{task_id}] Selected tools: {selected_tools}")
            print(f"[{task_id}] Steps taken: {len(conversation_history)}")
            print(f"[{task_id}] Final answer: {answer}")
            
            return answer
            
        except Exception as e:
            error_msg = f"Critical error in agent execution: {str(e)}"
            print(f"[{task_id}] {error_msg}")
            return error_msg
        
def detect_file_type(file_path: str) -> Optional[str]:
    """Detect file type by file extension."""
    ext = Path(file_path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return "excel"
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
        return "image"
    elif ext in [".mp3", ".wav", ".ogg"]:
        return "audio"
    elif ext in [".py", ".ipynb"]:
        return "code"
    else:
        return None


def process_file(task_id: str, question_text: str) -> str:
    """
    Attempt to download a file associated with a task from the API.
    - If the file exists (HTTP 200), it is saved to a temp directory and the local file path is returned.
    - If no file is found (HTTP 404), returns None.
    - For all other HTTP errors, the exception is propagated to the caller.
    """
    file_url = f"{FILE_PATH}{task_id}"

    try:
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Exception in download_file>> {str(exc)}")
        return question_text # Unable to get the file

    # Determine filename from 'Content-Disposition' header, fallback to task_id
    content_disposition = response.headers.get("content-disposition", "")
    filename = task_id
    match = re.search(r'filename="([^"]+)"', content_disposition)
    if match:
        filename = match.group(1)

    # Save file in a temp directory
    temp_storage_dir = Path(tempfile.gettempdir()) / "gaia_cached_files"
    temp_storage_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_storage_dir / filename
    file_path.write_bytes(response.content)
    return (
                f"{question_text}\n\n"
                f"---\n"
                f"A file was downloaded for this task and saved locally at:\n"
                f"{str(file_path)}\n"
                f"---\n\n"
            )

# Usage example
if __name__ == "__main__":
    agent = GaiaAgent()
    
    # Test questions
    test_questions = [
        "What information is in the Excel file data.xlsx?",
        "Search for recent news about artificial intelligence",
        "Tell me about the history of machine learning",
    ]
    
    for i, question in enumerate(test_questions):
        print(f"\n{'='*50}")
        result = agent(f"test_{i}", question)
        print(f"Result: {result}")