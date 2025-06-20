from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from PIL import Image
import re
import time
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from tabulate import tabulate
import whisper

import numpy as np
import os

# ----------- Enhanced Search Functionality -----------
class EnhancedSearchTool:
    """Enhanced web search with intelligent query processing and result filtering"""
    
    def __init__(self, max_results: int = 10):
        self.base_tool = DuckDuckGoSearchResults(num_results=max_results)
        self.max_results = max_results
        
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key search terms from the question using LLM"""
        try:
            extract_prompt = f"""
            Extract the most important search terms from this question for web search:
            Question: {question}
            
            Return ONLY a comma-separated list of key terms, no explanations.
            Focus on: proper nouns, specific concepts, technical terms, dates, numbers.
            Avoid: common words like 'what', 'how', 'when', 'the', 'is', 'are'.
            
            Example: "What is the population of Tokyo in 2023?" -> "Tokyo population 2023"
            """
            
            response = llm.invoke(extract_prompt).content.strip()
            return [term.strip() for term in response.split(',')]
        except Exception:
            # Fallback to simple keyword extraction
            return self._simple_keyword_extraction(question)
    
    def _simple_keyword_extraction(self, question: str) -> List[str]:
        """Fallback keyword extraction using regex"""
        # Remove common question words
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'the', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'should', 'would'}
        words = re.findall(r'\b[A-Za-z]+\b', question.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _generate_search_queries(self, question: str) -> List[str]:
        """Generate multiple search queries for comprehensive results"""
        key_terms = self._extract_key_terms(question)
        
        queries = []
        
        # Original question (cleaned)
        cleaned_question = re.sub(r'[^\w\s]', ' ', question).strip()
        queries.append(cleaned_question)
        
        # Key terms combined
        if key_terms:
            queries.append(' '.join(key_terms[:5]))  # Top 5 terms
        
        # Specific query patterns based on question type
        if any(word in question.lower() for word in ['latest', 'recent', 'current', 'new']):
            queries.append(f"{' '.join(key_terms[:3])} 2024 2025")
        
        if any(word in question.lower() for word in ['statistics', 'data', 'number', 'count']):
            queries.append(f"{' '.join(key_terms[:3])} statistics data")
        
        if any(word in question.lower() for word in ['definition', 'what is', 'meaning']):
            queries.append(f"{' '.join(key_terms[:2])} definition meaning")
        
        return list(dict.fromkeys(queries))  # Remove duplicates while preserving order
    
    def _filter_and_rank_results(self, results: List[Dict], question: str) -> List[Dict]:
        """Filter and rank search results based on relevance"""
        if not results:
            return results
        
        key_terms = self._extract_key_terms(question)
        key_terms_lower = [term.lower() for term in key_terms]
        
        scored_results = []
        for result in results:
            score = 0
            text_content = (result.get('snippet', '') + ' ' + result.get('title', '')).lower()
            
            # Score based on key term matches
            for term in key_terms_lower:
                if term in text_content:
                    score += text_content.count(term)
            
            # Bonus for recent dates
            if any(year in text_content for year in ['2024', '2025', '2023']):
                score += 2
            
            # Penalty for very short snippets
            if len(result.get('snippet', '')) < 50:
                score -= 1
            
            scored_results.append((score, result))
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results[:self.max_results]]
    
    def run(self, question: str) -> str:
        """Enhanced search execution with multiple queries and result filtering"""
        try:
            search_queries = self._generate_search_queries(question)
            all_results = []
            
            for query in search_queries[:3]:  # Limit to 3 queries to avoid rate limits
                try:
                    results = self.base_tool.run(query)
                    if isinstance(results, str):
                        # Parse string results if needed
                        try:
                            results = json.loads(results) if results.startswith('[') else [{'snippet': results, 'title': 'Search Result'}]
                        except:
                            results = [{'snippet': results, 'title': 'Search Result'}]
                    
                    if isinstance(results, list):
                        all_results.extend(results)
                    
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"Search query failed: {query} - {e}")
                    continue
            
            if not all_results:
                return "No search results found."
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(all_results, question)
            
            # Format results
            formatted_results = []
            for i, result in enumerate(filtered_results[:5], 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No description')
                link = result.get('link', '')
                
                formatted_results.append(f"{i}. {title}\n   {snippet}\n   Source: {link}\n")
            
            return "ENHANCED SEARCH RESULTS:\n" + "\n".join(formatted_results)
            
        except Exception as e:
            return f"Enhanced search error: {str(e)}"

# ----------- Enhanced Wikipedia Tool -----------
class EnhancedWikipediaTool:
    """Enhanced Wikipedia search with intelligent query processing and content extraction"""
    
    def __init__(self):
        self.base_wrapper = WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=3000,
            load_all_available_meta=True
        )
        self.base_tool = WikipediaQueryRun(api_wrapper=self.base_wrapper)
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract named entities for Wikipedia search"""
        try:
            entity_prompt = f"""
            Extract named entities (people, places, organizations, concepts) from this question for Wikipedia search:
            Question: {question}
            
            Return ONLY a comma-separated list of the most important entities.
            Focus on: proper nouns, specific names, places, organizations, historical events, scientific concepts.
            
            Example: "Tell me about Einstein's theory of relativity" -> "Albert Einstein, theory of relativity, relativity"
            """
            
            response = llm.invoke(entity_prompt).content.strip()
            entities = [entity.strip() for entity in response.split(',')]
            return [e for e in entities if len(e) > 2]
        except Exception:
            # Fallback: extract capitalized words and phrases
            return self._extract_capitalized_terms(question)
    
    def _extract_capitalized_terms(self, question: str) -> List[str]:
        """Fallback: extract capitalized terms as potential entities"""
        # Find capitalized words and phrases
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        # Also look for quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', question)
        quoted_terms.extend(re.findall(r"'([^']+)'", question))
        
        return capitalized_words + quoted_terms
    
    def _search_multiple_terms(self, entities: List[str]) -> Dict[str, str]:
        """Search Wikipedia for multiple entities and return best results"""
        results = {}
        
        for entity in entities[:3]:  # Limit to avoid too many API calls
            try:
                result = self.base_tool.run(entity)
                if result and "Page:" in result and len(result) > 100:
                    results[entity] = result
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Wikipedia search failed for '{entity}': {e}")
                continue
        
        return results
    
    def _extract_relevant_sections(self, content: str, question: str) -> str:
        """Extract the most relevant sections from Wikipedia content"""
        if not content or len(content) < 200:
            return content
        
        # Split content into sections (usually separated by double newlines)
        sections = re.split(r'\n\s*\n', content)
        
        # Score sections based on relevance to question
        key_terms = self._extract_entities(question)
        key_terms_lower = [term.lower() for term in key_terms]
        
        scored_sections = []
        for section in sections:
            if len(section.strip()) < 50:
                continue
                
            score = 0
            section_lower = section.lower()
            
            # Score based on key term matches
            for term in key_terms_lower:
                score += section_lower.count(term)
            
            # Bonus for sections with dates, numbers, or specific facts
            if re.search(r'\b(19|20)\d{2}\b', section):  # Years
                score += 1
            if re.search(r'\b\d+([.,]\d+)?\s*(million|billion|thousand|percent|%)\b', section):
                score += 1
            
            scored_sections.append((score, section))
        
        # Sort by relevance and take top sections
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        top_sections = [section for score, section in scored_sections[:3] if score > 0]
        
        if not top_sections:
            # If no highly relevant sections, take first few sections
            top_sections = sections[:2]
        
        return '\n\n'.join(top_sections)
    
    def run(self, question: str) -> str:
        """Enhanced Wikipedia search with entity extraction and content filtering"""
        try:
            entities = self._extract_entities(question)
            
            if not entities:
                # Fallback to direct search with cleaned question
                cleaned_question = re.sub(r'[^\w\s]', ' ', question).strip()
                try:
                    result = self.base_tool.run(cleaned_question)
                    return self._extract_relevant_sections(result, question) if result else "No Wikipedia results found."
                except Exception as e:
                    return f"Wikipedia search error: {str(e)}"
            
            # Search for multiple entities
            search_results = self._search_multiple_terms(entities)
            
            if not search_results:
                return "No relevant Wikipedia articles found."
            
            # Combine and format results
            formatted_results = []
            for entity, content in search_results.items():
                relevant_content = self._extract_relevant_sections(content, question)
                if relevant_content:
                    formatted_results.append(f"=== {entity} ===\n{relevant_content}")
            
            if not formatted_results:
                return "No relevant information found in Wikipedia articles."
            
            return "ENHANCED WIKIPEDIA RESULTS:\n\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Enhanced Wikipedia error: {str(e)}"

# ----------- Enhanced File Processing Tools -----------
def excel_to_markdown(inputs: dict) -> str:
    """Enhanced Excel tool with better error handling and data analysis"""
    try:
        excel_path = inputs["excel_path"]
        sheet_name = inputs.get("sheet_name", None)
        file_path = Path(excel_path).expanduser().resolve()
        if not file_path.is_file():
            return f"Error: Excel file not found at {file_path}"

        sheet: Union[str, int] = (
            int(sheet_name) if sheet_name and sheet_name.isdigit() else sheet_name or 0
        )
        df = pd.read_excel(file_path, sheet_name=sheet)
        
        # Enhanced metadata
        metadata = f"EXCEL FILE ANALYSIS:\n"
        metadata += f"File: {file_path.name}\n"
        metadata += f"Dimensions: {len(df)} rows × {len(df.columns)} columns\n"
        metadata += f"Columns: {', '.join(df.columns.tolist())}\n"
        
        # Data type information
        metadata += f"Data types: {dict(df.dtypes)}\n"
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            metadata += f"Numeric columns: {list(numeric_cols)}\n"
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                metadata += f"  {col}: mean={df[col].mean():.2f}, min={df[col].min()}, max={df[col].max()}\n"
        
        metadata += "\nSAMPLE DATA (first 10 rows):\n"
        
        if hasattr(df, "to_markdown"):
            sample_data = df.head(10).to_markdown(index=False)
        else:
            sample_data = tabulate(df.head(10), headers="keys", tablefmt="github", showindex=False)
        
        return metadata + sample_data + f"\n\n(Showing first 10 rows of {len(df)} total rows)"
        
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"

def image_file_info(image_path: str, question: str) -> str:
    """Enhanced image file analysis using Gemini API"""
    try:
        from google import genai
        from google.genai.types import  Part

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Read content from a local file
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=[
                question,
                Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            ],
        )
        return response.text
    
    except Exception as e:
        return f"Error during image analysis: {e}"

def audio_file_info(audio_path: str) -> str:
    """Returns only the transcription of an audio file."""
    try:
        model = whisper.load_model("tiny")  # Fast + accurate balance
        result = model.transcribe(audio_path, fp16=False)
        return result['text']
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def code_file_read(code_path: str) -> str:
    """Enhanced code file analysis"""
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        file_path = Path(code_path)
        
        info = f"CODE FILE ANALYSIS:\n"
        info += f"File: {file_path.name}\n"
        info += f"Extension: {file_path.suffix}\n"
        info += f"Size: {len(content)} characters, {len(content.splitlines())} lines\n"
        
        # Language-specific analysis
        if file_path.suffix == '.py':
            # Python-specific analysis
            import_lines = [line for line in content.splitlines() if line.strip().startswith(('import ', 'from '))]
            if import_lines:
                info += f"Imports ({len(import_lines)}): {', '.join(import_lines[:5])}\n"
            
            # Count functions and classes
            func_count = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
            class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
            info += f"Functions: {func_count}, Classes: {class_count}\n"
        
        info += f"\nCODE CONTENT:\n{content}"
        return info
        
    except Exception as e:
        return f"Error reading code file: {e}"
    
    
import yt_dlp
from pathlib import Path

def extract_youtube_info(question: str) -> str:
    """
    Download a YouTube video or audio using yt-dlp without merging.

    Parameters:
    - url: str — YouTube URL
    - audio_only: bool — if True, downloads audio only; else best single video+audio stream

    Returns:
    - str: path to downloaded file or error message
    """
    pattern = r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w\-]+|youtu\.be/[\w\-]+))"
    match = re.search(pattern, question)
    youtube_url =  match.group(1) if match else None
    print(f"Extracting YouTube URL: {youtube_url}")
    
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", youtube_url)
    video_id = match.group(1) if match else "dummy_id"
    file_path = Path(video_id)

    output_dir = Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # best mp4 combined stream or fallback to best available
        'outtmpl': str(file_path),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return audio_file_info(str(file_path))
    except Exception as e:
        return f"Error: {e}"