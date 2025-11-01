"""
PDF field extraction using Docling with LangGraph workflow and Phoenix observability.
Extracts custom fields from PDF documents using Docling + Google Gemini with stateful graph-based processing and LLM tracing.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
from datetime import datetime

from docling.document_converter import DocumentConverter
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Arize Phoenix imports (OpenInference instrumentation)
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
import phoenix as px


class ExtractionState(TypedDict):
    """State for the PDF extraction workflow."""
    pdf_path: str
    pdf_name: str
    fields_to_extract: Dict[str, str]  # field_name -> field_type
    field_names: List[str]  # Just the field names
    text_content: str
    raw_output: Dict
    attempt: int
    extracted_fields: Dict[str, Optional[str]]
    missing_fields: List[str]
    raw_output_files: List[str]
    success: bool
    error: Optional[str]
    logs: List[Dict[str, str]]
    log_callback: Optional[callable]


class DoclingExtractor:
    """Extract specific fields from PDF documents using Docling + Google Gemini."""
    
    REQUIRED_FIELDS = [
        "Shipping Charges",
        "Insurance",
        "BILL TO address",
        "INVOICE NUMBER",
        "Date"
    ]
    
    MAX_ATTEMPTS = 5
    
    # --- Configuration ---
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    MODEL_ID = "gemini-2.5-flash"
    
    # --- Phoenix Configuration ---
    PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "pdf-extraction")
    PHOENIX_COLLECTOR_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    _phoenix_initialized = False
    
    def __init__(self, output_dir: str = "raw_outputs", max_attempts: int = None):
        """Initialize extractor with output directory and LangGraph workflow."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.converter = DocumentConverter()
        
        # Set max attempts (use provided value or default)
        self.max_attempts = max_attempts or self.MAX_ATTEMPTS
        
        # Initialize LLM for extraction (Google Gemini)
        self.llm = ChatVertexAI(
            model=self.MODEL_ID,
            project=self.PROJECT_ID,
            location=self.LOCATION,
            temperature=0
        )
        
        # Initialize Arize Phoenix tracing
        self._setup_phoenix_tracing()
        
        # Build the LangGraph workflow
        self.workflow = self._build_extraction_workflow()
    
    @classmethod
    def _setup_phoenix_tracing(cls):
        """Initialize Arize Phoenix tracing for LLM observability (local mode)."""
        if cls._phoenix_initialized:
            return

        # Check if Phoenix is disabled via environment variable
        phoenix_enabled = os.environ.get('PHOENIX_ENABLED', 'true').lower()
        if phoenix_enabled == 'false':
            print("⚠️  Phoenix tracing disabled via PHOENIX_ENABLED=false")
            print("LLM calls will not be traced")
            cls._phoenix_initialized = True  # Mark as initialized to avoid repeated checks
            return

        try:
            # Ensure endpoint is set for local Phoenix collector
            os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", cls.PHOENIX_COLLECTOR_ENDPOINT)

            # Register Phoenix OpenTelemetry exporter and instrument LangChain
            tracer_provider = register(project_name=cls.PHOENIX_PROJECT_NAME, auto_instrument=False)
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

            # Launch Phoenix locally for development
            px.launch_app()
            print("✅ Arize Phoenix tracing enabled (local mode)")
            print("📊 Access Phoenix UI at: http://localhost:6006")
            print(f"✅ Phoenix dataset: {cls.PHOENIX_PROJECT_NAME}")

            cls._phoenix_initialized = True

        except Exception as e:
            print(f"⚠️  Failed to initialize Phoenix tracing: {e}")
            print("LLM calls will not be traced")

    def extract_fields(self, text: str, fields, log_callback=None) -> Dict[str, Optional[str]]:
        """Extract required fields using LLM with Phoenix tracing.
        
        Args:
            text: Document text to extract from
            fields: Either List[str] or Dict[str, str] (field_name: field_type)
            log_callback: Optional callback for streaming logs
        """
        # Run LLM extraction directly
        return self._extract_with_llm(text, fields)

    def _extract_with_llm(self, text: str, fields) -> Dict[str, Optional[str]]:
        """Extract fields using LangChain LLM directly with Phoenix tracing."""
        # Get field names for result dict
        if isinstance(fields, dict):
            field_names = list(fields.keys())
        else:
            field_names = fields
        
        prompt = self.build_extraction_prompt(fields)
        chain = prompt | self.llm
        
        # Truncate text if too long (keep first 10000 chars)
        truncated_text = text[:10000] if len(text) > 10000 else text
        response = chain.invoke({"text": truncated_text})
        
        try:
            # Parse LLM response as JSON
            content = response.content.strip()
            
            # Log the raw response for debugging
            print(f"\n=== LLM RAW RESPONSE ===")
            print(content[:1000])
            print(f"=== END RESPONSE ===\n")
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                # Remove first line (```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                # Remove 'json' if it's the first line
                if lines and lines[0].strip().lower() == 'json':
                    lines = lines[1:]
                content = '\n'.join(lines).strip()
            
            print(f"\n=== CLEANED CONTENT ===")
            print(content[:500])
            print(f"=== END CLEANED ===\n")
            
            extracted = json.loads(content)
            
            # Log extracted values
            print(f"Extracted fields: {extracted}")
            
            return extracted
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"LLM response: {response.content[:1000]}")
            # If LLM didn't return valid JSON, return empty dict
            return {field: None for field in field_names}
    
    def _build_extraction_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for PDF extraction."""
        workflow = StateGraph(ExtractionState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_extraction)
        workflow.add_node("parse_pdf", self._parse_pdf_node)
        workflow.add_node("build_prompt", self._build_prompt_node)
        workflow.add_node("extract_fields", self._extract_fields_node)
        workflow.add_node("validate_extraction", self._validate_extraction_node)
        workflow.add_node("handle_max_attempts", self._handle_max_attempts_node)
        
        # Define the flow
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "parse_pdf")
        workflow.add_edge("parse_pdf", "build_prompt")
        workflow.add_edge("build_prompt", "extract_fields")
        workflow.add_edge("extract_fields", "validate_extraction")
        
        # Conditional edges from validation
        workflow.add_conditional_edges(
            "validate_extraction",
            self._should_retry,
            {
                "retry": "extract_fields",
                "complete": "handle_max_attempts"
            }
        )
        
        workflow.add_edge("handle_max_attempts", END)
        
        return workflow.compile()
    
    def _initialize_extraction(self, state: ExtractionState) -> ExtractionState:
        """Initialize the extraction state."""
        pdf_name = Path(state["pdf_path"]).stem
        
        # Convert fields to dict format if needed
        fields_to_extract = state["fields_to_extract"]
        if isinstance(fields_to_extract, list):
            fields_to_extract = {field: "string" for field in fields_to_extract}
        
        field_names = list(fields_to_extract.keys())
        
        # Log initialization
        logs = state.get("logs", [])
        logs.append({"level": "info", "message": f"Starting extraction for {pdf_name}"})
        logs.append({"level": "info", "message": f"Fields to extract: {', '.join(field_names)}"})
        
        return {
            **state,
            "pdf_name": pdf_name,
            "fields_to_extract": fields_to_extract,
            "field_names": field_names,
            "attempt": 1,
            "extracted_fields": {},
            "missing_fields": field_names.copy(),
            "raw_output_files": [],
            "success": False,
            "error": None,
            "logs": logs
        }
    
    def _parse_pdf_node(self, state: ExtractionState) -> ExtractionState:
        """Parse PDF using Docling."""
        logs = state["logs"]
        logs.append({"level": "info", "message": "Parsing PDF with Docling..."})
        
        try:
            text_content, raw_output = self.parse_pdf(state["pdf_path"])
            logs.append({"level": "success", "message": f"PDF parsed successfully. Markdown length: {len(text_content)} chars"})
            
            return {
                **state,
                "text_content": text_content,
                "raw_output": raw_output,
                "logs": logs
            }
        except Exception as e:
            logs.append({"level": "error", "message": f"PDF parsing failed: {str(e)}"})
            return {
                **state,
                "error": str(e),
                "logs": logs
            }
    
    def _build_prompt_node(self, state: ExtractionState) -> ExtractionState:
        """Build the extraction prompt (only done once)."""
        # This node just passes through - prompt building happens in extract_fields_node
        return state
    
    def _extract_fields_node(self, state: ExtractionState) -> ExtractionState:
        """Extract fields using LLM."""
        logs = state["logs"]
        attempt = state["attempt"]
        
        logs.append({"level": "info", "message": f"Attempt {attempt}/{self.max_attempts}"})
        
        try:
            # Save raw output for this attempt
            raw_file = self.save_raw_output(state["raw_output"], attempt, state["pdf_name"])
            raw_output_files = state["raw_output_files"] + [raw_file]
            
            logs.append({"level": "info", "message": f"Saved raw output to: {raw_file}"})
            logs.append({"level": "info", "message": f"Extracting fields with Google Gemini {self.MODEL_ID}..."})
            
            # Stream LLM start if callback provided
            if state.get("log_callback"):
                state["log_callback"]({
                    'type': 'llm_start',
                    'model': self.MODEL_ID,
                    'fields': state["field_names"]
                })
            
            # Extract fields using LLM directly (avoid circular dependency)
            extracted = self._extract_with_llm(state["text_content"], state["fields_to_extract"])
            
            # Stream LLM complete if callback provided
            if state.get("log_callback"):
                state["log_callback"]({
                    'type': 'llm_complete',
                    'extracted': extracted
                })
            
            logs.append({"level": "success", "message": f"Extraction complete. Found {len([v for v in extracted.values() if v])} fields"})
            
            return {
                **state,
                "extracted_fields": extracted,
                "raw_output_files": raw_output_files,
                "logs": logs
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logs.append({"level": "error", "message": f"Attempt {attempt} failed: {str(e)}"})
            logs.append({"level": "error", "message": f"Error details: {error_trace}"})
            
            return {
                **state,
                "error": str(e),
                "logs": logs
            }
    
    def _validate_extraction_node(self, state: ExtractionState) -> ExtractionState:
        """Validate extraction results and check for missing fields."""
        logs = state["logs"]
        extracted = state["extracted_fields"]
        
        # Check for missing fields
        missing = self.check_missing_fields(extracted, state["fields_to_extract"])
        
        if not missing:
            logs.append({"level": "success", "message": "All fields extracted successfully!"})
            return {
                **state,
                "missing_fields": missing,
                "success": True,
                "logs": logs
            }
        else:
            # Increment attempt for retry
            new_attempt = state["attempt"] + 1
            if new_attempt <= self.max_attempts:
                logs.append({"level": "warning", "message": f"Still missing {len(missing)} fields: {', '.join(missing)}. Retrying..."})
            else:
                logs.append({"level": "warning", "message": f"Still missing {len(missing)} fields: {', '.join(missing)}. Max attempts reached."})
            
            return {
                **state,
                "missing_fields": missing,
                "attempt": new_attempt,
                "logs": logs
            }
    
    def _should_retry(self, state: ExtractionState) -> str:
        """Decide whether to retry or complete."""
        attempt = state["attempt"]
        missing = state["missing_fields"]
        
        if not missing:
            return "complete"
        elif attempt > self.max_attempts:
            return "complete"
        else:
            return "retry"
    
    def _handle_max_attempts_node(self, state: ExtractionState) -> ExtractionState:
        """Handle completion or max attempts reached."""
        logs = state["logs"]
        missing = state["missing_fields"]
        
        if missing:
            error_msg = f"Failed to extract all fields after {self.max_attempts} attempts"
            logs.append({"level": "error", "message": error_msg})
            
            # Save comprehensive audit log
            result_for_audit = {
                "success": state["success"],
                "attempts": state["attempt"],
                "extracted_fields": state["extracted_fields"],
                "missing_fields": state["missing_fields"],
                "raw_output_files": state["raw_output_files"],
                "error": state.get("error"),
                "logs": state["logs"]
            }
            audit_file = self.save_audit_log(state["pdf_name"], result_for_audit, state["fields_to_extract"])
            
            logs.append({"level": "info", "message": f"Saved audit log to: {audit_file}"})
            
            return {
                **state,
                "error": error_msg,
                "logs": logs,
                "audit_file": audit_file
            }
        
        return state
    
    def _extract_with_llm(self, text: str, fields) -> Dict[str, Optional[str]]:
        """Extract fields using LangChain LLM directly."""
        # Get field names for result dict
        if isinstance(fields, dict):
            field_names = list(fields.keys())
        else:
            field_names = fields
        
        prompt = self.build_extraction_prompt(fields)
        chain = prompt | self.llm
        
        # Truncate text if too long (keep first 10000 chars)
        truncated_text = text[:10000] if len(text) > 10000 else text
        response = chain.invoke({"text": truncated_text})
        
        try:
            # Parse LLM response as JSON
            content = response.content.strip()
            
            # Log the raw response for debugging
            print(f"\n=== LLM RAW RESPONSE ===")
            print(content[:1000])
            print(f"=== END RESPONSE ===\n")
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                # Remove first line (```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                # Remove 'json' if it's the first line
                if lines and lines[0].strip().lower() == 'json':
                    lines = lines[1:]
                content = '\n'.join(lines).strip()
            
            print(f"\n=== CLEANED CONTENT ===")
            print(content[:500])
            print(f"=== END CLEANED ===\n")
            
            extracted = json.loads(content)
            
            # Log extracted values
            print(f"Extracted fields: {extracted}")
            
            return extracted
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"LLM response: {response.content[:1000]}")
            # If LLM didn't return valid JSON, return empty dict
            return {field: None for field in field_names}

    def parse_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Parse PDF with Docling and return text + raw output."""
        # Convert PDF using Docling
        result = self.converter.convert(pdf_path)
        
        # Export to markdown for text extraction
        markdown_text = result.document.export_to_markdown()
        
        # Store raw output as dict
        raw_output = {
            "markdown": markdown_text,
            "metadata": {
                "num_pages": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                "source": pdf_path,
                "parser": "docling"
            }
        }
        
        return markdown_text, raw_output
    
    def build_extraction_prompt(self, fields) -> ChatPromptTemplate:
        """Build extraction prompt dynamically based on fields.
        
        Args:
            fields: Either List[str] or Dict[str, str] (field_name: field_type)
        """
        # Handle both list and dict formats
        if isinstance(fields, dict):
            fields_dict = fields
        else:
            # Convert list to dict with default 'string' type
            fields_dict = {field: "string" for field in fields}
        
        # Build field descriptions
        fields_list = "\n".join([f"- {name} ({ftype})" for name, ftype in fields_dict.items()])
        fields_json = ",\n".join([f'  "{name}": "<{ftype} value or null>"' for name, ftype in fields_dict.items()])
        
        system_message = f"""You are a precise document extraction assistant specialized in extracting structured data from invoices and documents.

Extract the following fields from the document text:
{fields_list}

CRITICAL INSTRUCTIONS:
1. Look CAREFULLY through ALL the text, including tables and lists
2. For invoice numbers: Look for "Invoice Number", "INV-", "Invoice #", etc.
3. For dates: Look for "Invoice Date", "Date", "Due Date", etc.
4. For amounts: Look for "Total", "Sub Total", "Tax", dollar amounts ($)
5. For addresses: Look in "To:", "From:", "Bill To", "Ship To" sections
6. Extract EXACT values as they appear (including $ signs for money)
7. For addresses, include the complete address with all lines
8. If a field is truly not found anywhere, use null
9. Return ONLY valid JSON, no explanations or markdown

SEARCH STRATEGY:
- Check table rows and columns
- Check labeled sections (Invoice Number:, Date:, etc.)
- Check headers and footers
- Look for similar field names (e.g., "Invoice #" for "INVOICE NUMBER")

Return a JSON object with these exact keys:
{{{{
{fields_json}
}}}}

Example:
{{{{
  "INVOICE NUMBER": "INV-3337",
  "Date": "January 25, 2016",
  "Total": "$93.50"
}}}}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Document text:\n\n{text}")
        ])
    
    def extract_fields(self, text: str, fields, log_callback=None) -> Dict[str, Optional[str]]:
        """Extract required fields using LLM.
        
        Args:
            text: Document text to extract from
            fields: Either List[str] or Dict[str, str] (field_name: field_type)
            log_callback: Optional callback for streaming logs
        """
        # Run LLM extraction directly
        return self._extract_with_llm(text, fields)
    
    def save_raw_output(self, raw_output: Dict, attempt: int, pdf_name: str) -> str:
        """Save raw Docling output to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON metadata
        json_filename = f"{pdf_name}_attempt_{attempt}_{timestamp}.json"
        json_filepath = self.output_dir / json_filename
        
        with open(json_filepath, 'w') as f:
            json.dump(raw_output, f, indent=2)
        
        # Save markdown separately for easy viewing
        md_filename = f"{pdf_name}_attempt_{attempt}_{timestamp}.md"
        md_filepath = self.output_dir / md_filename
        
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# PDF Extraction - Attempt {attempt}\n\n")
            f.write(f"**Timestamp:** {timestamp}\n\n")
            f.write(f"**Source:** {raw_output['metadata']['source']}\n\n")
            f.write(f"**Parser:** {raw_output['metadata']['parser']}\n\n")
            f.write("---\n\n")
            f.write(raw_output['markdown'])
        
        return str(json_filepath)
    
    def save_audit_log(self, pdf_name: str, result: Dict, fields) -> str:
        """Save comprehensive audit log for the extraction process."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_filename = f"{pdf_name}_audit_{timestamp}.json"
        audit_filepath = self.output_dir / audit_filename
        
        # Get field names
        if isinstance(fields, dict):
            field_names = list(fields.keys())
            field_types = fields
        else:
            field_names = fields
            field_types = {f: "string" for f in fields}
        
        audit_data = {
            "extraction_summary": {
                "pdf_name": pdf_name,
                "timestamp": timestamp,
                "success": result["success"],
                "total_attempts": result["attempts"],
                "fields_requested": field_names,
                "field_types": field_types,
                "fields_extracted": len([v for v in result["extracted_fields"].values() if v]),
                "fields_missing": len(result["missing_fields"])
            },
            "extraction_result": {
                "extracted_fields": result["extracted_fields"],
                "missing_fields": result["missing_fields"],
                "error": result.get("error")
            },
            "processing_details": {
                "raw_output_files": result["raw_output_files"],
                "markdown_files": [f.replace('.json', '.md') for f in result["raw_output_files"]],
                "logs": result.get("logs", [])
            },
            "field_analysis": {}
        }
        
        # Analyze each field
        for field_name in field_names:
            value = result["extracted_fields"].get(field_name)
            audit_data["field_analysis"][field_name] = {
                "type": field_types.get(field_name, "string"),
                "extracted": value is not None and value != "",
                "value": value,
                "attempts_to_extract": result["attempts"] if value else None
            }
        
        with open(audit_filepath, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        # Also create a human-readable audit report
        report_filename = f"{pdf_name}_audit_{timestamp}.txt"
        report_filepath = self.output_dir / report_filename
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PDF EXTRACTION AUDIT REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Document: {pdf_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            f.write(f"Total Attempts: {result['attempts']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("EXTRACTION RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            for field_name in field_names:
                value = result["extracted_fields"].get(field_name)
                field_type = field_types.get(field_name, "string")
                status = "✓ FOUND" if value else "✗ MISSING"
                f.write(f"{status} | {field_name} ({field_type})\n")
                if value:
                    f.write(f"       Value: {value}\n")
                f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("PROCESSING LOGS\n")
            f.write("-" * 80 + "\n\n")
            
            for log_entry in result.get("logs", []):
                level = log_entry.get("level", "info").upper()
                message = log_entry.get("message", "")
                f.write(f"[{level}] {message}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        return str(audit_filepath)
    
    def check_missing_fields(self, extracted: Dict[str, Optional[str]], fields) -> List[str]:
        """Return list of fields that are still missing (None or empty).
        
        Args:
            extracted: Dict of extracted values
            fields: Either List[str] or Dict[str, str] (field_name: field_type)
        """
        # Get field names
        if isinstance(fields, dict):
            field_names = list(fields.keys())
        else:
            field_names = fields
        
        missing = []
        for field in field_names:
            value = extracted.get(field)
            if value is None or value == "" or value == "null":
                missing.append(field)
        return missing
    
    def process_pdf(self, pdf_path: str, custom_fields: Optional[List[str]] = None) -> Dict:
        """
        Process PDF with up to MAX_ATTEMPTS retries.
        Returns extraction result with status and missing fields.
        
        Args:
            pdf_path: Path to PDF file
            custom_fields: Optional list of custom fields to extract. If None, uses REQUIRED_FIELDS.
        """
        # Use process_pdf_with_logging for the implementation
        result = self.process_pdf_with_logging(pdf_path, custom_fields)
        
        # Remove logs from the result to match the original interface
        result_copy = result.copy()
        result_copy.pop("logs", None)
        return result_copy
    
    def process_pdf_with_logging(self, pdf_path: str, custom_fields: Optional[List[str]] = None) -> Dict:
        """
        Process PDF with detailed logging for debugging.
        Returns extraction result with status, missing fields, and detailed logs.
        """
        # Use custom fields if provided, otherwise use default
        fields_to_extract = custom_fields if custom_fields else self.REQUIRED_FIELDS.copy()
        
        pdf_name = Path(pdf_path).stem
        
        # Initialize the extraction state
        initial_state = ExtractionState(
            pdf_path=pdf_path,
            pdf_name=pdf_name,
            fields_to_extract=fields_to_extract if isinstance(fields_to_extract, dict) else {field: "string" for field in fields_to_extract},
            field_names=[],
            text_content="",
            raw_output={},
            attempt=1,
            extracted_fields={},
            missing_fields=[],
            raw_output_files=[],
            success=False,
            error=None,
            logs=[],
            log_callback=None
        )
        
        # Run the LangGraph workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Convert back to the expected result format
        result = {
            "success": final_state["success"],
            "attempts": final_state["attempt"],
            "extracted_fields": final_state["extracted_fields"],
            "missing_fields": final_state["missing_fields"],
            "raw_output_files": final_state["raw_output_files"],
            "error": final_state.get("error"),
            "logs": final_state["logs"]
        }
        
        # Add audit file if it was saved
        if "audit_file" in final_state:
            result["audit_file"] = final_state["audit_file"]
        
        return result
    
    def process_pdf_streaming(self, pdf_path: str, custom_fields: Optional[List[str]] = None, log_callback=None) -> Dict:
        """
        Process PDF with real-time streaming logs via callback.
        
        Args:
            pdf_path: Path to PDF file
            custom_fields: Optional list of custom fields to extract
            log_callback: Function to call with log messages in real-time
        """
        # Use custom fields if provided, otherwise use default
        fields_to_extract = custom_fields if custom_fields else self.REQUIRED_FIELDS.copy()
        
        pdf_name = Path(pdf_path).stem
        
        # Initialize the extraction state
        initial_state = ExtractionState(
            pdf_path=pdf_path,
            pdf_name=pdf_name,
            fields_to_extract=fields_to_extract if isinstance(fields_to_extract, dict) else {field: "string" for field in fields_to_extract},
            field_names=[],
            text_content="",
            raw_output={},
            attempt=1,
            extracted_fields={},
            missing_fields=[],
            raw_output_files=[],
            success=False,
            error=None,
            logs=[],
            log_callback=log_callback
        )
        
        # Run the LangGraph workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Convert back to the expected result format
        result = {
            "success": final_state["success"],
            "attempts": final_state["attempt"],
            "extracted_fields": final_state["extracted_fields"],
            "missing_fields": final_state["missing_fields"],
            "raw_output_files": final_state["raw_output_files"],
            "error": final_state.get("error"),
            "logs": final_state["logs"]
        }
        
        # Add audit file if it was saved
        if "audit_file" in final_state:
            result["audit_file"] = final_state["audit_file"]
        
        return result


if __name__ == "__main__":
    # Test the extractor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    extractor = DoclingExtractor()
    result = extractor.process_pdf(pdf_path)
    
    print("\n" + "="*60)
    print("EXTRACTION RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))
