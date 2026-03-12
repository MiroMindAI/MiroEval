"""
Base Evaluator for Deep Research Arena

This module provides the base class for all evaluators (pointwise, pairwise, etc.)
with common functionality including data loading, caching, LLM client management,
and utility functions.
"""

import json
import os
import logging
import re
import time
import glob
import base64
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from deepresearcharena.utils.llm_call import APIClient
from deepresearcharena.cache import CacheManager

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    def __init__(self, 
                 data_dir: str, # data_dir: Directory containing evaluation data
                 model_name: str = "google/gemini-2.5-pro-preview",  # LLM model name for evaluation
                 api_type: str = "openrouter", #  API type (openai or openrouter)
                 cache_dir: str = None): # cache_dir: Directory for cache files (defaults to outputs/cache)
        self.data_dir = data_dir
        self.model_name = model_name
        self.api_type = api_type
        
        # Initialize data containers
        self.queries: Dict[int, Dict[str, Any]] = {}
        self.model_results: Dict[str, Dict[int, str]] = {}
        
        # Initialize cache system
        if cache_dir is None:
            cache_dir = "outputs/cache"
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        
        # Initialize LLM client
        self.client = APIClient(model_name=model_name, API_Type=api_type)
        logger.info(f"Initialized evaluator with model: {model_name}")
        
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
    
    def load_data(self):
        """Load queries and model results data"""
        logger.info("Loading evaluation data...")
        
        # Load queries
        query_file = os.path.join(self.data_dir, "input_queries", "query.jsonl")
        self.queries = self._load_queries(query_file)
        logger.info(f"Loaded {len(self.queries)} queries")
        
        # Load model results
        results_dir = os.path.join(self.data_dir, "method_results")
        if os.path.exists(results_dir):
            model_folders = [f for f in os.listdir(results_dir) 
                           if os.path.isdir(os.path.join(results_dir, f))]
            
            for model_name in model_folders:
                model_results = self._load_model_results(os.path.join(results_dir, model_name))
                self.model_results[model_name] = model_results
                logger.info(f"Loaded {len(model_results)} results for model: {model_name}")
        else:
            logger.warning(f"Model results directory not found: {results_dir}")
    
    def _load_queries(self, query_file: str) -> Dict[int, Dict[str, Any]]:
        """
        Load queries from JSONL file.

        Each query is represented as Q = (I, A) where:
          - I (instruction): the 'prompt' field
          - A (attachments): optional 'attachment' field, which can be:
            - inline text content
            - a file path or list of file paths to attachment files
        """
        queries = {}
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        query_data = json.loads(line.strip())
                        if 'id' in query_data:
                            # Load attachment content if it references file paths
                            if 'attachment' in query_data:
                                raw_attachment = query_data['attachment']
                                # If attachment is a list of file paths, store per-file contents
                                if isinstance(raw_attachment, list):
                                    parts = [self._resolve_attachment(item) for item in raw_attachment]
                                    query_data['attachment_parts'] = parts
                                    query_data['attachment'] = "\n\n---\n\n".join(parts)
                                else:
                                    query_data['attachment'] = self._resolve_attachment(raw_attachment)
                            queries[query_data['id']] = query_data
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse query line: {line.strip()[:100]}... Error: {e}")
        return queries

    def _resolve_attachment(self, attachment) -> str:
        """
        Resolve attachment content from various formats.

        Supports:
          - Inline text: returned as-is
          - Single file path (str): file content is read and returned
          - List of file paths: each file is read and concatenated

        Supported file types: .txt, .csv, .json, .md, .pdf (text extraction)
        """
        # If it's a list of file paths, resolve each and concatenate
        if isinstance(attachment, list):
            parts = []
            for item in attachment:
                parts.append(self._resolve_attachment(item))
            return "\n\n---\n\n".join(parts)

        # If it's a string, check if it's a file path
        if isinstance(attachment, str):
            # Check if it looks like a file path (relative to data_dir or absolute)
            potential_path = attachment
            if not os.path.isabs(potential_path):
                potential_path = os.path.join(self.data_dir, "attachments", attachment)

            if os.path.isfile(potential_path):
                return self._read_attachment_file(potential_path)

            # Otherwise treat as inline text content
            return attachment

        return str(attachment)

    # Image file extensions that need LLM-based reading
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    def _read_attachment_file(self, filepath: str) -> str:
        """Read attachment file content based on file type."""
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext in ('.txt', '.csv', '.md', '.json', '.jsonl', '.tsv'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext == '.pdf':
                return self._extract_pdf_text(filepath)
            elif ext in self.IMAGE_EXTENSIONS:
                return self._describe_image_with_llm(filepath)
            else:
                # Fallback: try reading as text
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Failed to read attachment file {filepath}: {e}")
            return f"[Error reading attachment: {filepath}]"

    def _describe_image_with_llm(self, filepath: str) -> str:
        """Use multimodal LLM to describe and extract key information from an image."""
        logger.info(f"Reading image with LLM: {filepath}")

        # Read and encode image as base64
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        ext = os.path.splitext(filepath)[1].lower()
        mime_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.bmp': 'image/bmp', '.webp': 'image/webp',
            '.tiff': 'image/tiff', '.tif': 'image/tiff',
        }
        mime_type = mime_map.get(ext, 'image/png')

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please carefully examine this image and extract all key information from it. "
                            "Include:\n"
                            "1. A detailed description of what the image shows\n"
                            "2. All text, numbers, labels, and annotations visible in the image\n"
                            "3. Any data, charts, tables, or structured information\n"
                            "4. Key observations and notable details\n\n"
                            "Be thorough and precise. This description will be used as a text representation "
                            "of the image for downstream analysis."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    }
                ]
            }
        ]

        response = self.generate_llm_response(messages, max_tokens=4096, temperature=0.1)

        if response == "$ERROR$":
            logger.error(f"Failed to describe image: {filepath}")
            return f"[Error: LLM failed to describe image: {os.path.basename(filepath)}]"

        filename = os.path.basename(filepath)
        return f"[Image: {filename}]\n{response}"

    @staticmethod
    def _extract_pdf_text(filepath: str) -> str:
        """Extract text from PDF file. Requires PyPDF2 or falls back to placeholder."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("PyPDF2 not installed. PDF text extraction unavailable. Install with: pip install PyPDF2")
            return f"[PDF attachment: {os.path.basename(filepath)} — PyPDF2 required for text extraction]"
        except Exception as e:
            logger.warning(f"Failed to extract PDF text from {filepath}: {e}")
            return f"[Error extracting PDF: {filepath}]"
    
    def _load_model_results(self, model_dir: str) -> Dict[int, str]:
        """ Load model results from directory """
        results = {} 
        files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
        
        for filename in files:
            filepath = os.path.join(model_dir, filename)
            query_id = self._extract_query_id(filename)
            
            if query_id is not None:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        response = self._extract_response(result_data)
                        if response:
                            results[query_id] = response
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load result file {filepath}: {e}")
        
        return results
    
    def _extract_query_id(self, filename: str) -> Optional[int]:
        """ Extract query ID from filename """
        # Try different filename patterns
        patterns = [
            r'deep_research_(\d+)_',  # Format: deep_research_N_timestamp.json
            r'^(\d+)\.json$'           # Format: N.json
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_response(self, result_data: Dict[str, Any]) -> Optional[str]:
        """ Extract response content from result data """
        # Try different response formats
        if 'entries' in result_data and result_data['entries']:
            entry = result_data['entries'][0]
            return entry.get('response', '')
        else:
            assert False, "No response found in result data"
        return None
    
    def find_report_file(self, model_name: str, query_id: int) -> Optional[str]:
        """ Find the report file for a given model and query """    
        model_dir = os.path.join(self.data_dir, "method_results", model_name)

        # Look for files matching the query_id
        patterns = [
            f"deep_research_{query_id}_*.json",
            f"{query_id}.json"
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(model_dir, pattern))
            if files:
                return files[0]
        
        # If no exact match, list all files for debugging
        all_files = os.listdir(model_dir)
        logger.warning(f"No file found for query {query_id} in {model_name}. Available files: {all_files[:10]}...")
        return None
    
    def extract_json_from_response(self, text: str) -> Optional[str]:
        """ Extract JSON from LLM response """
        # Try to find JSON in markdown code blocks and other patterns
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'<json_output>\s*(.*?)\s*</json_output>',
            r'\[(.*?)\]',
            r'\{(.*?)\}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # For list patterns, we need to add back the brackets
                    if pattern.startswith(r'\['):
                        test_json = f'[{match}]'
                    elif pattern.startswith(r'\{'):
                        test_json = f'{{{match}}}'
                    else:
                        test_json = match.strip()
                    
                    json.loads(test_json)
                    return test_json
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def extract_json_from_analysis_output(self, text: str) -> Optional[str]:
        """ Extract JSON from analysis output with specific tags """
        if not isinstance(text, str):
            return None
        
        # Look for <json_output> tags first
        json_output_match = re.search(r'<json_output>\s*(.*?)\s*</json_output>', text, re.DOTALL)
        if json_output_match:
            json_str = json_output_match.group(1).strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # Fall back to general JSON extraction
        return self.extract_json_from_response(text)
    
    def generate_llm_response(self, 
                            messages: List[Dict[str, str]], 
                            max_tokens: int = 8192, 
                            temperature: float = 0.1,
                            max_retries: int = 3) -> str:
        """ Generate response from LLM with error handling """
        for attempt in range(max_retries):
            try:
                response = self.client.generate_response(
                    messages, 
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
                if response != "$ERROR$":
                    return response
                else:
                    logger.warning(f"LLM returned error response (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("All LLM call attempts failed")
        return "$ERROR$"
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """ Save results to JSON file """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    def get_cache_statistics(self) -> Dict[str, int]:
        """ Get statistics about cache usage """
        return self.cache_manager.get_cache_sizes()
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.cache_manager.clear_all_caches()
        logger.info("All caches cleared")
    
    @abstractmethod
    def evaluate_query(self, query_id: int, *args, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a single query (to be implemented by subclasses)
        
        Args:
            query_id: ID of the query to evaluate
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Evaluation results for the query
        """
        pass
    
    @abstractmethod
    def evaluate_all_queries(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Evaluate all queries (to be implemented by subclasses)
        
        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Evaluation results for all queries
        """
        pass
    
    def print_cache_info(self):
        """Print information about cache usage"""
        cache_sizes = self.get_cache_statistics()
        print("\n" + "="*50)
        print("Cache Information")
        print("="*50)
        
        if not cache_sizes:
            print("No caches in use")
        else:
            for cache_name, size in cache_sizes.items():
                print(f"{cache_name}: {size} items")
        
        print("="*50)
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        # Cache manager will automatically save any pending changes
        pass
