import os
import re
from pathlib import Path
from typing import Optional, Union, Dict, List, Any
from enum import Enum
import requests
import tempfile

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain.tools import Tool as LangTool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path


from tools import (
    EnhancedSearchTool,
    EnhancedWikipediaTool,
    excel_to_markdown,
    image_file_info,
    audio_file_info,
    code_file_read,
    extract_youtube_info)

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



# Initialize enhanced tools
enhanced_search_tool = LangTool.from_function(
    name="enhanced_web_search",
    func=EnhancedSearchTool().run,
    description="Enhanced web search with intelligent query processing, multiple search strategies, and result filtering. Provides comprehensive and relevant search results."
)

enhanced_wiki_tool = LangTool.from_function(
    name="enhanced_wikipedia",
    func=EnhancedWikipediaTool().run,
    description="Enhanced Wikipedia search with entity extraction, multi-term search, and relevant content filtering. Provides detailed encyclopedic information."
)

excel_tool = LangTool.from_function(
    name="excel_to_text",
    func=excel_to_markdown,
    description="Enhanced Excel analysis with metadata, statistics, and structured data preview. Inputs: 'excel_path' (str), 'sheet_name' (str, optional).",
)

image_tool = LangTool.from_function(
    name="image_file_info",
    func=image_file_info,
    description="Enhanced image file analysis with detailed metadata and properties."
)

audio_tool = LangTool.from_function(
    name="audio_file_info",
    func=audio_file_info,
    description="Enhanced audio processing with transcription, language detection, and timestamped segments."
)

code_tool = LangTool.from_function(
    name="code_file_read",
    func=code_file_read,
    description="Enhanced code file analysis with language-specific insights and structure analysis."
)

youtube_tool = LangTool.from_function(
    name="extract_youtube_info",
    func=extract_youtube_info,
    description="Extracts transcription from the youtube link"
)

# Enhanced tool registry
AVAILABLE_TOOLS = {
    "excel": excel_tool,
    "search": enhanced_search_tool,
    "wikipedia": enhanced_wiki_tool,
    "image": image_tool,
    "audio": audio_tool,
    "code": code_tool,
    "youtube": youtube_tool
}

# ----------- Intelligent Tool Selection -----------
def analyze_question(state: AgentState) -> AgentState:
    """Enhanced question analysis with better tool recommendation"""
    analysis_prompt = f"""
    Analyze this question and determine the best tools and approach:
    Question: {state["question"]}
    
    Available enhanced tools:
    1. excel - Enhanced Excel/CSV analysis with statistics and metadata
    2. search - Enhanced web search with intelligent query processing and result filtering
    3. wikipedia - Enhanced Wikipedia search with entity extraction and content filtering
    4. image - Enhanced image analysis with what the image contains
    5. audio - Enhanced audio processing with transcription
    6. code - Enhanced code analysis with language-specific insights
    7. youtube - Extracts transcription from the youtube link
    
    Consider:
    - Question type (factual, analytical, current events, technical)
    - Required information sources (files, web, encyclopedic)
    - Time sensitivity (current vs historical information)
    - Complexity level
    
    Respond with:
    1. Question type: <type>
    2. Primary tools needed: <tools>
    3. Search strategy: <strategy>
    4. Expected answer format: <format>
    
    Format: TYPE: <type> | TOOLS: <tools> | STRATEGY: <strategy> | FORMAT: <format>
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
    """Enhanced tool selection with smarter logic"""
    question = state["question"].lower()
    selected_tools = []

    # File-based tool selection
    if any(keyword in question for keyword in ["excel", "csv", "spreadsheet", ".xlsx", ".xls"]):
        selected_tools.append("excel")
    if any(keyword in question for keyword in [".png", ".jpg", ".jpeg", ".bmp", ".gif", "image"]):
        selected_tools.append("image")
    if any(keyword in question for keyword in [".mp3", ".wav", ".ogg", "audio", "transcribe"]):
        selected_tools.append("audio")
    if any(keyword in question for keyword in [".py", ".ipynb", "code", "script", "function"]):
        selected_tools.append("code")
    if any(keyword in question for keyword in ["youtube"]):
        selected_tools.append("youtube")

    # Information-based tool selection
    current_indicators = ["recent", "current", "news", "today", "2025", "now"]
    encyclopedia_indicators = ["wiki", "wikipedia"]
    
    if any(indicator in question for indicator in current_indicators):
        selected_tools.append("search")
    elif any(indicator in question for indicator in encyclopedia_indicators):
        selected_tools.append("wikipedia")
    elif any(keyword in question for keyword in ["search", "find", "look up", "information about"]):
        # Use both for comprehensive coverage
        selected_tools.extend(["search", "wikipedia"])

    # Default fallback
    if not selected_tools:
        if any(word in question for word in ["who", "what", "when", "where"]):
            selected_tools.append("wikipedia")
        selected_tools.append("search")

    # Remove duplicates while preserving order
    selected_tools = list(dict.fromkeys(selected_tools))
    
    state["selected_tools"] = selected_tools
    state["current_step"] = AgentStep.EXECUTE_TOOLS.value
    return state

def execute_tools(state: AgentState) -> AgentState:
    """Enhanced tool execution with better error handling"""
    results = {}

    # Enhanced file detection
    file_path = None
    downloaded_file_marker = "A file was downloaded for this task and saved locally at:"
    if downloaded_file_marker in state["question"]:
        lines = state["question"].splitlines()
        for i, line in enumerate(lines):
            if downloaded_file_marker in line:
                if i + 1 < len(lines):
                    file_path_candidate = lines[i + 1].strip()
                    if Path(file_path_candidate).exists():
                        file_path = file_path_candidate
                        print(f"Detected file path: {file_path}")
                break

    for tool_name in state["selected_tools"]:
        try:
            print(f"Executing tool: {tool_name}")
            
            # File-based tools
            if tool_name in ["excel", "image", "audio", "code"] and file_path:
                if tool_name == "excel":
                    result = AVAILABLE_TOOLS["excel"].run({"excel_path": file_path, "sheet_name": None})
                elif tool_name == "image":
                    result = AVAILABLE_TOOLS["image"].run({"image_path": file_path, "question": state["question"], "llm": llm})
                elif tool_name == "youtube":
                    # print(f"Running YouTube tool with file path: {file_path}")
                    # result = AVAILABLE_TOOLS["youtube"].run(state["question"])
                    result = "AUDIO/VIDEO TOOL NOT IMPLEMENTED"
                else:
                    result = AVAILABLE_TOOLS[tool_name].run(file_path)
            # Information-based tools
            else:
                # Extract clean query for search tools
                clean_query = state["question"]
                if downloaded_file_marker in clean_query:
                    clean_query = clean_query.split(downloaded_file_marker)[0].strip()
                
                result = AVAILABLE_TOOLS[tool_name].run(clean_query)

            results[tool_name] = result

            print(f"Tool {tool_name} completed successfully.")
            print(f"Output for {tool_name}: {result}")
            
        except Exception as e:
            error_msg = f"Error using {tool_name}: {str(e)}"
            results[tool_name] = error_msg
            state["error_count"] += 1
            print(error_msg)

    state["tool_results"] = results
    state["current_step"] = AgentStep.SYNTHESIZE_ANSWER.value
    return state

def synthesize_answer(state: AgentState) -> AgentState:
    """Enhanced answer synthesis with better formatting"""
    if not state["tool_results"]:
        state["final_answer"] = "I couldn't find any relevant information to answer your question."
        state["current_step"] = AgentStep.COMPLETE.value
        return state
    
    # Enhanced synthesis prompt
    tool_results_str = "\n".join([f"=== {tool.upper()} RESULTS ===\n{result}\n" for tool, result in state["tool_results"].items()])

    cot_prompt = f"""You are a precise assistant tasked with analyzing the user's question using the available tool outputs.

    Question:
    {state["question"]}

    Available tool outputs (from text, tables, audio, video, or images):
    {tool_results_str}

    Instructions:
    - Carefully analyze the question and each tool output to determine a strategy to find the answer.
    - Break down your reasoning step-by-step using only the facts available in the outputs.
    - Consider any decoding, logical reasoning, counting, or interpretation required.
    - For audio, image, or video content, rely only on the transcriptions or structured descriptions provided.
    - Do not guess. If information is insufficient, note that clearly.
    - Conclude your response with a clearly marked line: `---END OF ANALYSIS---`

    Your step-by-step analysis:"""

    cot_response = llm.invoke(cot_prompt).content

    final_answer_prompt = f"""You are a precise assistant tasked with deriving the final answer from the step-by-step analysis below.

    Question:
    {state["question"]}

    Step-by-step analysis:
    {cot_response}

    Instructions:
    - Carefully read the question and the full analysis above carefully.
    - Output **only** the final answer — no reasoning, explanation, or formatting.
    - If the analysis concludes that a definitive answer cannot be determined, respond with "NA".
    - The answer should be concise and directly address the question.

    Final answer:"""

        
    try:
        response = llm.invoke(final_answer_prompt).content
        state["final_answer"] = response
        state["current_step"] = AgentStep.COMPLETE.value
    except Exception as e:
        state["error_count"] += 1
        state["final_answer"] = f"Error synthesizing answer: {e}"
        state["current_step"] = AgentStep.ERROR_RECOVERY.value
    
    return state

def error_recovery(state: AgentState) -> AgentState:
    """Enhanced error recovery with multiple fallback strategies"""
    if state["error_count"] >= state["max_errors"]:
        state["final_answer"] = "I encountered multiple errors and cannot complete this task reliably."
        state["current_step"] = AgentStep.COMPLETE.value
    else:
        # Enhanced fallback: try with simplified approach
        try:
            fallback_prompt = f"""
            Answer this question directly using your knowledge:
            {state["original_question"]}
            
            Provide a helpful response even if you cannot access external tools.
            Be clear about any limitations in your answer.
            """
            response = llm.invoke(fallback_prompt).content
            state["final_answer"] = f"Using available knowledge (some tools unavailable): {response}"
            state["current_step"] = AgentStep.COMPLETE.value
        except Exception as e:
            state["final_answer"] = f"All approaches failed. Error: {e}"
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

# Create enhanced workflow
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

# Compile the enhanced graph
graph = workflow.compile()

# ----------- Agent Class -----------
class GaiaAgent:
    """GAIA Agent with tools and intelligent processing"""
    
    def __init__(self):
        self.graph = graph
        self.tool_usage_stats = {}
        print("Enhanced GAIA Agent initialized with:")
        print("✓ Intelligent multi-query web search")
        print("✓ Entity-aware Wikipedia search")
        print("✓ Enhanced file processing tools")
        print("✓ Advanced error recovery")
        print("✓ Comprehensive result synthesis")

    def get_tool_stats(self) -> Dict[str, int]:
        """Get usage statistics for tools"""
        return self.tool_usage_stats.copy()

    def __call__(self, task_id: str, question: str) -> str:
        print(f"\n{'='*60}")
        print(f"[{task_id}] ENHANCED PROCESSING: {question}")
        
        # Initialize state
        processed_question = process_file(task_id, question)
        initial_state = initialize_state(processed_question)
        
        try:
            # Execute the enhanced workflow
            result = self.graph.invoke(initial_state)
            
            # Extract results
            answer = result.get("final_answer", "No answer generated")
            selected_tools = result.get("selected_tools", [])
            conversation_history = result.get("conversation_history", [])
            tool_results = result.get("tool_results", {})
            error_count = result.get("error_count", 0)
            
            # Update tool usage statistics
            for tool in selected_tools:
                self.tool_usage_stats[tool] = self.tool_usage_stats.get(tool, 0) + 1
            
            # Enhanced logging
            print(f"[{task_id}] Selected tools: {selected_tools}")
            print(f"[{task_id}] Tools executed: {list(tool_results.keys())}")
            print(f"[{task_id}] Processing steps: {len(conversation_history)}")
            print(f"[{task_id}] Errors encountered: {error_count}")
            
            # Log tool result sizes for debugging
            for tool, result in tool_results.items():
                result_size = len(str(result)) if result else 0
                print(f"[{task_id}] {tool} result size: {result_size} chars")
            
            print(f"[{task_id}] FINAL ANSWER: {answer}")
            print(f"{'='*60}")
            
            return answer
            
        except Exception as e:
            error_msg = f"Critical error in enhanced agent execution: {str(e)}"
            print(f"[{task_id}] {error_msg}")
            
            # Try fallback direct LLM response
            try:
                fallback_response = llm.invoke(f"Please answer this question: {question}").content
                return f"Fallback response: {fallback_response}"
            except:
                return error_msg

# ----------- Enhanced File Processing -----------
def detect_file_type(file_path: str) -> Optional[str]:
    """Enhanced file type detection with more formats"""
    ext = Path(file_path).suffix.lower()
    
    file_type_mapping = {
        # Spreadsheets
        '.xlsx': 'excel', '.xls': 'excel', '.csv': 'excel',
        # Images
        '.png': 'image', '.jpg': 'image', '.jpeg': 'image', 
        '.bmp': 'image', '.gif': 'image', '.tiff': 'image', '.webp': 'image',
        # Audio
        '.mp3': 'audio', '.wav': 'audio', '.ogg': 'audio', 
        '.flac': 'audio', '.m4a': 'audio', '.aac': 'audio',
        # Code
        '.py': 'code', '.ipynb': 'code', '.js': 'code', '.html': 'code',
        '.css': 'code', '.java': 'code', '.cpp': 'code', '.c': 'code',
        '.sql': 'code', '.r': 'code', '.json': 'code', '.xml': 'code',
        # Documents
        '.txt': 'text', '.md': 'text', '.pdf': 'document',
        '.doc': 'document', '.docx': 'document'
    }
    
    return file_type_mapping.get(ext)

def process_file(task_id: str, question_text: str) -> str:
    """Enhanced file processing with better error handling and metadata"""
    file_url = f"{FILE_PATH}{task_id}"
    
    try:
        print(f"[{task_id}] Attempting to download file from: {file_url}")
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        print(f"[{task_id}] File download successful. Status: {response.status_code}")
        
    except requests.exceptions.RequestException as exc:
        print(f"[{task_id}] File download failed: {str(exc)}")
        return question_text  # Return original question if no file
    
    # Enhanced filename extraction
    content_disposition = response.headers.get("content-disposition", "")
    filename = task_id  # Default fallback
    
    # Try to extract filename from Content-Disposition header
    filename_match = re.search(r'filename[*]?=(?:"([^"]+)"|([^;]+))', content_disposition)
    if filename_match:
        filename = filename_match.group(1) or filename_match.group(2)
        filename = filename.strip()
    
    # Create enhanced temp directory structure
    temp_storage_dir = Path(tempfile.gettempdir()) / "gaia_enhanced_files" / task_id
    temp_storage_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = temp_storage_dir / filename
    file_path.write_bytes(response.content)
    
    # Get file metadata
    file_size = len(response.content)
    file_type = detect_file_type(filename)
    
    print(f"[{task_id}] File saved: {filename} ({file_size:,} bytes, type: {file_type})")
    
    # Enhanced question augmentation
    enhanced_question = f"{question_text}\n\n"
    enhanced_question += f"{'='*50}\n"
    enhanced_question += f"FILE INFORMATION:\n"
    enhanced_question += f"A file was downloaded for this task and saved locally at:\n"
    enhanced_question += f"{str(file_path)}\n"
    enhanced_question += f"File details:\n"
    enhanced_question += f"- Name: {filename}\n"
    enhanced_question += f"- Size: {file_size:,} bytes ({file_size/1024:.1f} KB)\n"
    enhanced_question += f"- Type: {file_type or 'unknown'}\n"
    enhanced_question += f"{'='*50}\n\n"
    
    return enhanced_question

# ----------- Usage Examples and Testing -----------
def run_enhanced_tests():
    """Run comprehensive tests of the enhanced agent"""
    agent = GaiaAgent()
    
    test_cases = [
        {
            "id": "test_search_1",
            "question": "What are the latest developments in artificial intelligence in 2024?",
            "expected_tools": ["search"]
        },
        {
            "id": "test_wiki_1", 
            "question": "Tell me about Albert Einstein's contributions to physics",
            "expected_tools": ["wikipedia"]
        },
        {
            "id": "test_combined_1",
            "question": "What is machine learning and what are recent breakthroughs?",
            "expected_tools": ["wikipedia", "search"]
        },
        {
            "id": "test_excel_1",
            "question": "Analyze the data in the Excel file sales_data.xlsx",
            "expected_tools": ["excel"]
        }
    ]
    
    print("\n" + "="*80)
    print("RUNNING ENHANCED AGENT TESTS")
    print("="*80)
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['id']}")
        print(f"Question: {test_case['question']}")
        print(f"Expected tools: {test_case['expected_tools']}")
        
        try:
            result = agent(test_case['id'], test_case['question'])
            print(f"Result length: {len(result)} characters")
            print(f"Result preview: {result[:200]}...")
        except Exception as e:
            print(f"Test failed: {e}")
        
        print("-" * 60)
    
    # Print tool usage statistics
    print(f"\nTool Usage Statistics:")
    for tool, count in agent.get_tool_stats().items():
        print(f"  {tool}: {count} times")

# Usage example
if __name__ == "__main__":
    # Create enhanced agent
    agent = GaiaAgent()
    
    # Example usage
    sample_questions = [
        "What is the current population of Tokyo and how has it changed recently?",
        "Explain quantum computing and its recent developments",
        "Tell me about the history of machine learning and current AI trends",
    ]
    
    print("\n" + "="*80)
    print("ENHANCED GAIA AGENT DEMONSTRATION")
    print("="*80)
    
    for i, question in enumerate(sample_questions):
        print(f"\nExample {i+1}: {question}")
        result = agent(f"demo_{i}", question)
        print(f"Answer: {result[:300]}...")
        print("-" * 60)
    
    # Uncomment to run comprehensive tests
    # run_enhanced_tests()