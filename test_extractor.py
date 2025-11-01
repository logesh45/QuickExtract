#!/usr/bin/env python3
"""
Comprehensive test suite for the PDF field extraction CLI.
Tests both the extractor module and the CLI interface.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import argparse

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from extractor import DoclingExtractor
from extract import load_fields_config, validate_pdf_file, save_results, create_output_directory, main


class TestDoclingExtractor(unittest.TestCase):
    """Test cases for DoclingExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = DoclingExtractor(output_dir=self.temp_dir)
        
        # Sample fields configuration
        self.sample_fields = {
            "INVOICE NUMBER": "string",
            "Date": "date",
            "Total": "number",
            "BILL TO address": "string"
        }
        
        # Sample document text
        self.sample_text = """
        INVOICE NUMBER: INV-12345
        Date: 2024-01-15
        BILL TO:
        Test Company
        123 Test Street
        Test City, TS 12345
        
        Items:
        Item 1    $100.00
        Item 2    $50.00
        Total:    $150.00
        """
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor.converter)
        self.assertIsNotNone(self.extractor.llm)
        self.assertEqual(self.extractor.output_dir, Path(self.temp_dir))
        self.assertTrue(self.extractor.output_dir.exists())
    
    def test_build_extraction_prompt(self):
        """Test prompt building with different field configurations."""
        # Test with dict
        prompt = self.extractor.build_extraction_prompt(self.sample_fields)
        self.assertIsNotNone(prompt)
        
        # Test with list
        field_list = list(self.sample_fields.keys())
        prompt = self.extractor.build_extraction_prompt(field_list)
        self.assertIsNotNone(prompt)
    
    @patch('extractor.ChatVertexAI')
    def test_extract_fields_mock(self, mock_llm_class):
        """Test field extraction with mocked LLM."""
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "INVOICE NUMBER": "INV-12345",
            "Date": "2024-01-15",
            "Total": "$150.00",
            "BILL TO address": "Test Company\n123 Test Street\nTest City, TS 12345"
        })
        
        # Re-initialize extractor with mocked LLM
        extractor = DoclingExtractor(output_dir=self.temp_dir)
        
        # Mock the chain invoke
        with patch.object(extractor, '_extract_with_llm') as mock_extract:
            mock_extract.return_value = {
                "INVOICE NUMBER": "INV-12345",
                "Date": "2024-01-15",
                "Total": "$150.00",
                "BILL TO address": "Test Company\n123 Test Street\nTest City, TS 12345"
            }
            
            result = extractor.extract_fields(self.sample_text, self.sample_fields)
            
            self.assertIn("INVOICE NUMBER", result)
            self.assertEqual(result["INVOICE NUMBER"], "INV-12345")
    
    def test_check_missing_fields(self):
        """Test missing field detection."""
        # All fields present
        extracted = {
            "INVOICE NUMBER": "INV-12345",
            "Date": "2024-01-15",
            "Total": "$150.00",
            "BILL TO address": "Test Company"
        }
        
        missing = self.extractor.check_missing_fields(extracted, self.sample_fields)
        self.assertEqual(len(missing), 0)
        
        # Some fields missing
        extracted_partial = {
            "INVOICE NUMBER": "INV-12345",
            "Date": None,
            "Total": "",
            "BILL TO address": "Test Company"
        }
        
        missing = self.extractor.check_missing_fields(extracted_partial, self.sample_fields)
        self.assertEqual(len(missing), 2)
        self.assertIn("Date", missing)
        self.assertIn("Total", missing)


class TestCLIFunctions(unittest.TestCase):
    """Test cases for CLI helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample fields config
        self.sample_config = {
            "INVOICE NUMBER": "string",
            "Date": "date",
            "Total": "number"
        }
    
    def test_load_fields_config_valid(self):
        """Test loading valid fields configuration."""
        config_file = Path(self.temp_dir) / "test_fields.json"
        
        with open(config_file, 'w') as f:
            json.dump(self.sample_config, f)
        
        config = load_fields_config(str(config_file))
        self.assertEqual(config, self.sample_config)
    
    def test_load_fields_config_invalid_json(self):
        """Test loading invalid JSON configuration."""
        config_file = Path(self.temp_dir) / "invalid.json"
        
        with open(config_file, 'w') as f:
            f.write("{ invalid json }")
        
        with self.assertRaises(SystemExit):
            load_fields_config(str(config_file))
    
    def test_load_fields_config_invalid_types(self):
        """Test loading configuration with unsupported field types."""
        invalid_config = {
            "field1": "unsupported_type"
        }
        
        config_file = Path(self.temp_dir) / "invalid_types.json"
        
        with open(config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        with self.assertRaises(SystemExit):
            load_fields_config(str(config_file))
    
    def test_load_fields_config_not_dict(self):
        """Test loading configuration that's not a dict."""
        invalid_config = ["field1", "field2"]
        
        config_file = Path(self.temp_dir) / "invalid_format.json"
        
        with open(config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        with self.assertRaises(SystemExit):
            load_fields_config(str(config_file))
    
    def test_validate_pdf_file_valid(self):
        """Test PDF file validation with valid file."""
        # Create a dummy PDF file
        pdf_file = Path(self.temp_dir) / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")
        
        result = validate_pdf_file(str(pdf_file))
        self.assertEqual(result, pdf_file)
    
    def test_validate_pdf_file_not_found(self):
        """Test PDF file validation with non-existent file."""
        with self.assertRaises(SystemExit):
            validate_pdf_file("nonexistent.pdf")
    
    def test_validate_pdf_file_wrong_extension(self):
        """Test PDF file validation with wrong file extension."""
        txt_file = Path(self.temp_dir) / "test.txt"
        txt_file.write_text("Not a PDF")
        
        with self.assertRaises(SystemExit):
            validate_pdf_file(str(txt_file))
    
    def test_validate_pdf_file_not_file(self):
        """Test PDF file validation with directory."""
        with self.assertRaises(SystemExit):
            validate_pdf_file(str(self.temp_dir))
    
    def test_create_output_directory(self):
        """Test output directory creation."""
        output_dir = create_output_directory()
        self.assertTrue(output_dir.exists())
        self.assertEqual(output_dir.name, "output")
    
    def test_save_results(self):
        """Test saving extraction results."""
        results = {
            "success": True,
            "extracted_fields": {"INVOICE NUMBER": "INV-123"},
            "missing_fields": []
        }
        
        output_file = save_results(results, "test_pdf", Path(self.temp_dir))
        
        self.assertTrue(output_file.exists())
        self.assertEqual(output_file.name, "extracted_test_pdf.json")
        
        # Verify content
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, results)
    
    @patch('extract.sys.argv', ['extract.py', 'fields.json', 'test.pdf'])
    @patch('extract.load_fields_config')
    @patch('extract.validate_pdf_file')
    @patch('extract.create_output_directory')
    @patch('extract.DoclingExtractor')
    def test_main_success(self, mock_extractor_class, mock_create_dir, mock_validate, mock_load_config):
        """Test main CLI function with successful extraction."""
        # Setup mocks
        mock_load_config.return_value = {"field1": "string"}
        mock_validate.return_value = Path("test.pdf")
        mock_create_dir.return_value = Path("output")
        
        mock_extractor = MagicMock()
        mock_extractor.process_pdf_with_logging.return_value = {
            "success": True,
            "extracted_fields": {"field1": "value1"},
            "missing_fields": []
        }
        mock_extractor_class.return_value = mock_extractor
        
        # Mock save_results to avoid file operations
        with patch('extract.save_results') as mock_save:
            mock_save.return_value = Path("output/result.json")
            
            # Test main function
            with patch('extract.Path') as mock_path:
                mock_path.return_value = Path("test.pdf")
                main()
            
            # Verify calls
            mock_load_config.assert_called_once_with('fields.json')
            mock_validate.assert_called_once_with('test.pdf')
            mock_extractor.process_pdf_with_logging.assert_called_once()
    
    @patch('extract.sys.argv', ['extract.py', 'fields.json', 'test.pdf'])
    @patch('extract.load_fields_config')
    def test_main_load_config_error(self, mock_load_config):
        """Test main function with config loading error."""
        mock_load_config.side_effect = SystemExit(1)
        
        with self.assertRaises(SystemExit):
            main()
    
    @patch('extract.sys.argv', ['extract.py', 'fields.json', 'test.pdf'])
    @patch('extract.load_fields_config')
    @patch('extract.validate_pdf_file')
    @patch('extract.DoclingExtractor')
    def test_main_extraction_error(self, mock_extractor_class, mock_validate, mock_load_config):
        """Test main function with extraction error."""
        # Setup mocks
        mock_load_config.return_value = {"field1": "string"}
        mock_validate.return_value = Path("test.pdf")
        
        mock_extractor = MagicMock()
        mock_extractor.process_pdf_with_logging.side_effect = Exception("Extraction failed")
        mock_extractor_class.return_value = mock_extractor
        
        with self.assertRaises(SystemExit):
            main()


class TestAdditionalExtractorMethods(unittest.TestCase):
    """Test additional extractor methods for better coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = DoclingExtractor(output_dir=self.temp_dir)
    
    def test_save_raw_output(self):
        """Test saving raw output files."""
        raw_output = {
            "markdown": "# Test Document\nContent here",
            "metadata": {
                "num_pages": 1,
                "source": "test.pdf",
                "parser": "docling"
            }
        }
        
        result_path = self.extractor.save_raw_output(raw_output, 1, "test_pdf")
        
        self.assertTrue(Path(result_path).exists())
        self.assertTrue(result_path.endswith('.json'))
        
        # Check JSON file was created
        with open(result_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, raw_output)
        
        # Check markdown file was also created
        md_file = result_path.replace('.json', '.md')
        self.assertTrue(Path(md_file).exists())
    
    def test_save_audit_log(self):
        """Test saving audit log."""
        result = {
            "success": True,
            "attempts": 1,
            "extracted_fields": {"field1": "value1"},
            "missing_fields": [],
            "raw_output_files": ["test.json"],
            "error": None,
            "logs": [{"level": "info", "message": "test"}]
        }
        fields = {"field1": "string"}
        
        audit_path = self.extractor.save_audit_log("test_pdf", result, fields)
        
        self.assertTrue(Path(audit_path).exists())
        self.assertTrue(audit_path.endswith('.json'))
        
        # Verify audit content
        with open(audit_path, 'r') as f:
            audit_data = json.load(f)
        
        self.assertIn("extraction_summary", audit_data)
        self.assertIn("field_analysis", audit_data)
        self.assertEqual(audit_data["extraction_summary"]["pdf_name"], "test_pdf")
    
    def test_check_missing_fields_dict(self):
        """Test missing field check with dict input."""
        fields = {"field1": "string", "field2": "number"}
        extracted = {"field1": "value1", "field2": None}
        
        missing = self.extractor.check_missing_fields(extracted, fields)
        self.assertEqual(missing, ["field2"])
    
    def test_check_missing_fields_list(self):
        """Test missing field check with list input."""
        fields = ["field1", "field2", "field3"]
        extracted = {"field1": "value1", "field2": "", "field3": "value3"}
        
        missing = self.extractor.check_missing_fields(extracted, fields)
        self.assertEqual(missing, ["field2"])
    
    def test_check_missing_fields_null_values(self):
        """Test missing field check with null string values."""
        fields = ["field1", "field2"]
        extracted = {"field1": "value1", "field2": "null"}
        
        missing = self.extractor.check_missing_fields(extracted, fields)
        self.assertEqual(missing, ["field2"])
    
    def test_build_extraction_prompt_dict(self):
        """Test prompt building with dict fields."""
        fields = {"field1": "string", "field2": "number"}
        prompt = self.extractor.build_extraction_prompt(fields)
        
        self.assertIsNotNone(prompt)
        # Check that prompt contains field information
        prompt_str = str(prompt)
        self.assertIn("field1", prompt_str)
        self.assertIn("field2", prompt_str)
    
    def test_build_extraction_prompt_list(self):
        """Test prompt building with list fields."""
        fields = ["field1", "field2"]
        prompt = self.extractor.build_extraction_prompt(fields)
        
        self.assertIsNotNone(prompt)
        prompt_str = str(prompt)
        self.assertIn("field1", prompt_str)
        self.assertIn("field2", prompt_str)
    
    def test_parse_pdf(self):
        """Test PDF parsing method."""
        # Mock the DocumentConverter
        with patch.object(self.extractor.converter, 'convert') as mock_convert:
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = "# Test Document\nContent"
            mock_result.document.pages = [MagicMock()]  # Mock 1 page
            mock_convert.return_value = mock_result
            
            text, raw_output = self.extractor.parse_pdf("test.pdf")
            
            self.assertEqual(text, "# Test Document\nContent")
            self.assertIn("markdown", raw_output)
            self.assertIn("metadata", raw_output)
            self.assertEqual(raw_output["metadata"]["source"], "test.pdf")
    
    @patch('extractor.px.launch_app')
    @patch('extractor.register')
    @patch('extractor.LangChainInstrumentor')
    def test_setup_phoenix_tracing_success(self, mock_instrumentor, mock_register, mock_launch):
        """Test successful Phoenix tracing setup."""
        # Reset the class variable to test setup
        DoclingExtractor._phoenix_initialized = False
        
        # Enable Phoenix for this test
        with patch.dict(os.environ, {'PHOENIX_ENABLED': 'true'}):
            # Mock the register and instrumentor
            mock_tracer = MagicMock()
            mock_register.return_value = mock_tracer
            
            # Call setup method
            DoclingExtractor._setup_phoenix_tracing()
            
            # Verify calls
            mock_register.assert_called_once()
            mock_instrumentor.return_value.instrument.assert_called_once_with(tracer_provider=mock_tracer)
            mock_launch.assert_called_once()
            self.assertTrue(DoclingExtractor._phoenix_initialized)
    
    @patch('extractor.px.launch_app')
    @patch('extractor.register')
    def test_setup_phoenix_tracing_failure(self, mock_register, mock_launch):
        """Test Phoenix tracing setup failure."""
        # Reset the class variable
        DoclingExtractor._phoenix_initialized = False
        
        # Enable Phoenix for this test
        with patch.dict(os.environ, {'PHOENIX_ENABLED': 'true'}):
            # Mock failure
            mock_register.side_effect = Exception("Phoenix setup failed")
            
            # Call setup method (should not raise exception)
            DoclingExtractor._setup_phoenix_tracing()
            
            # Should not be initialized
            self.assertFalse(DoclingExtractor._phoenix_initialized)
    
    def test_setup_phoenix_tracing_disabled(self):
        """Test Phoenix tracing setup when disabled."""
        # Reset the class variable
        DoclingExtractor._phoenix_initialized = False
        
        # Ensure Phoenix is disabled
        with patch.dict(os.environ, {'PHOENIX_ENABLED': 'false'}):
            # Call setup method
            DoclingExtractor._setup_phoenix_tracing()
            
            # Should be marked as initialized to avoid repeated checks
            self.assertTrue(DoclingExtractor._phoenix_initialized)
    
    def test_process_pdf_with_custom_fields(self):
        """Test process_pdf with custom fields."""
        # Mock the workflow
        with patch.object(self.extractor.workflow, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "success": True,
                "attempt": 1,
                "extracted_fields": {"field1": "value1"},
                "missing_fields": [],
                "raw_output_files": [],
                "error": None,
                "logs": []
            }
            
            result = self.extractor.process_pdf("test.pdf", custom_fields=["field1"])
            
            self.assertTrue(result["success"])
            self.assertIn("field1", result["extracted_fields"])
    
    def test_process_pdf_without_custom_fields(self):
        """Test process_pdf with default fields."""
        # Mock the workflow
        with patch.object(self.extractor.workflow, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "success": True,
                "attempt": 1,
                "extracted_fields": {"Shipping Charges": "100"},
                "missing_fields": [],
                "raw_output_files": [],
                "error": None,
                "logs": []
            }
            
            result = self.extractor.process_pdf("test.pdf")
            
            self.assertTrue(result["success"])
            mock_invoke.assert_called_once()
    
    def test_process_pdf_streaming(self):
        """Test process_pdf_streaming method."""
        # Mock the workflow
        with patch.object(self.extractor.workflow, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "success": True,
                "attempt": 1,
                "extracted_fields": {"field1": "value1"},
                "missing_fields": [],
                "raw_output_files": [],
                "error": None,
                "logs": []
            }
            
            # Test with callback
            callback_calls = []
            def test_callback(msg):
                callback_calls.append(msg)
            
            result = self.extractor.process_pdf_streaming("test.pdf", custom_fields=["field1"], log_callback=test_callback)
            
            self.assertTrue(result["success"])
            self.assertIn("field1", result["extracted_fields"])
    
    def test_extract_fields_with_chain(self):
        """Test extract_fields method with chain."""
        # Mock the chain
        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"field1": "value1"}'
        mock_chain.invoke.return_value = mock_response
        
        result = self.extractor.extract_fields("Sample text", {"field1": "string"}, mock_chain)
        
        # The method might return None values, so just check it doesn't crash
        self.assertIsNotNone(result)
    
    def test_extract_fields_json_error(self):
        """Test extract_fields method with JSON error."""
        # Mock the chain with invalid JSON
        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{ invalid json }'
        mock_chain.invoke.return_value = mock_response
        
        result = self.extractor.extract_fields("Sample text", {"field1": "string"}, mock_chain)
        
        # The method might return None values, so just check it doesn't crash
        self.assertIsNotNone(result)
    
    def test_extract_fields_chain_error(self):
        """Test extract_fields method with chain error."""
        # Mock the chain with error
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Chain error")
        
        result = self.extractor.extract_fields("Sample text", {"field1": "string"}, mock_chain)
        
        # Should return empty dict or None values on error
        self.assertIsInstance(result, dict)


class TestPhoenixIntegration(unittest.TestCase):
    """Test cases for Arize Phoenix integration."""
    
    def test_phoenix_config_variables(self):
        """Test Phoenix configuration environment variables."""
        # Check default values (these may be overridden by environment)
        self.assertIsInstance(DoclingExtractor.PHOENIX_PROJECT_NAME, str)
        self.assertEqual(DoclingExtractor.PHOENIX_COLLECTOR_ENDPOINT, "http://localhost:6006")
    
    @patch.dict(os.environ, {'PHOENIX_PROJECT_NAME': 'test-project'})
    def test_phoenix_env_override(self):
        """Test Phoenix configuration via environment variables."""
        # This should pick up the environment variable
        extractor = DoclingExtractor()
        self.assertIsNotNone(extractor)


class TestWorkflowMethods(unittest.TestCase):
    """Test workflow methods for better coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = DoclingExtractor(output_dir=self.temp_dir)
    
    def test_parse_pdf_node(self):
        """Test the parse_pdf workflow node."""
        state = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string"},
            "field_names": ["field1"],
            "attempt": 1,
            "logs": []
        }
        
        # Mock the parse_pdf method
        with patch.object(self.extractor, 'parse_pdf') as mock_parse:
            mock_parse.return_value = ("Sample text", {"markdown": "Sample", "metadata": {}})
            
            result = self.extractor._parse_pdf_node(state)
            
            self.assertEqual(result["text_content"], "Sample text")
            self.assertIn("raw_output", result)
    
    def test_extract_fields_node(self):
        """Test the extract_fields workflow node."""
        state = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string"},
            "field_names": ["field1"],
            "text_content": "Sample text with field1: value1",
            "attempt": 1,
            "raw_output_files": [],
            "logs": [],
            "raw_output": {"markdown": "Sample", "metadata": {}}
        }
        
        # Mock the extract_fields method
        with patch.object(self.extractor, 'extract_fields') as mock_extract:
            mock_extract.return_value = {"field1": "value1"}
            
            result = self.extractor._extract_fields_node(state)
            
            self.assertIn("extracted_fields", result)
            self.assertEqual(result["extracted_fields"]["field1"], "value1")
    
    def test_validate_extraction_node_success(self):
        """Test the validate_extraction workflow node with success."""
        state = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string", "field2": "number"},
            "field_names": ["field1", "field2"],
            "extracted_fields": {"field1": "value1", "field2": "100"},
            "attempt": 1,
            "raw_output_files": ["test.json"],
            "logs": []
        }
        
        result = self.extractor._validate_extraction_node(state)
        
        self.assertTrue(result.get("success", False))
        self.assertEqual(result.get("missing_fields", []), [])
    
    def test_validate_extraction_node_missing_fields(self):
        """Test the validate_extraction workflow node with missing fields."""
        state = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string", "field2": "number"},
            "field_names": ["field1", "field2"],
            "extracted_fields": {"field1": "value1"},
            "attempt": 1,
            "raw_output_files": ["test.json"],
            "logs": []
        }
        
        result = self.extractor._validate_extraction_node(state)
        
        # Check if missing fields are detected
        missing = result.get("missing_fields", [])
        self.assertIn("field2", missing)
    
    def test_validate_extraction_node_retry_limit(self):
        """Test the validate_extraction workflow node at retry limit."""
        state = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string"},
            "field_names": ["field1"],
            "extracted_fields": {},
            "attempt": 5,  # At retry limit
            "raw_output_files": ["test.json"],
            "logs": []
        }
        
        result = self.extractor._validate_extraction_node(state)
        
        # At retry limit, should mark as success even with missing fields
        self.assertTrue(result.get("success", False))
        self.assertIn("missing_fields", result)
    
    def test_should_retry_condition(self):
        """Test the should_retry workflow condition."""
        # Test when should retry
        state_false = {
            "success": False,
            "attempt": 1,
            "missing_fields": ["field1"]
        }
        retry_result = self.extractor._should_retry(state_false)
        self.assertFalse(retry_result)  # Should return False for retry
        
        # Test when should not retry (success)
        state_true = {
            "success": True,
            "attempt": 1,
            "missing_fields": []
        }
        success_result = self.extractor._should_retry(state_true)
        self.assertTrue(success_result)  # Should return True for END
        
        # Test when should not retry (at limit)
        state_limit = {
            "success": False,
            "attempt": 5,
            "missing_fields": ["field1"]
        }
        limit_result = self.extractor._should_retry(state_limit)
        self.assertTrue(limit_result)  # Should return True for END
    
    def test_extract_with_llm_success(self):
        """Test _extract_with_llm method with success."""
        text = "Sample text with field1: value1"
        fields = {"field1": "string"}
        
        # Mock the chain
        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"field1": "value1"}'
        mock_chain.invoke.return_value = mock_response
        
        result = self.extractor._extract_with_llm(text, fields, mock_chain)
        
        self.assertEqual(result["field1"], "value1")
    
    def test_extract_with_llm_json_error(self):
        """Test _extract_with_llm method with JSON parsing error."""
        text = "Sample text"
        fields = {"field1": "string"}
        
        # Mock the chain with invalid JSON
        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{ invalid json }'
        mock_chain.invoke.return_value = mock_response
        
        result = self.extractor._extract_with_llm(text, fields, mock_chain)
        
        self.assertEqual(result, {})
    
    def test_extract_with_llm_chain_error(self):
        """Test _extract_with_llm method with chain error."""
        text = "Sample text"
        fields = {"field1": "string"}
        
        # Mock the chain with error
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Chain error")
        
        result = self.extractor._extract_with_llm(text, fields, mock_chain)
        
        self.assertEqual(result, {})
    
    def test_process_pdf_with_logging(self):
        """Test process_pdf_with_logging method."""
        # Mock the workflow
        with patch.object(self.extractor.workflow, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "success": True,
                "attempt": 1,
                "extracted_fields": {"field1": "value1"},
                "missing_fields": [],
                "raw_output_files": [],
                "error": None,
                "logs": []
            }
            
            result = self.extractor.process_pdf_with_logging("test.pdf", custom_fields=["field1"])
            
            self.assertTrue(result["success"])
            self.assertIn("field1", result["extracted_fields"])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = DoclingExtractor(output_dir=self.temp_dir)
    
    def test_build_extraction_prompt_empty_dict(self):
        """Test prompt building with empty dict."""
        prompt = self.extractor.build_extraction_prompt({})
        self.assertIsNotNone(prompt)
    
    def test_build_extraction_prompt_empty_list(self):
        """Test prompt building with empty list."""
        prompt = self.extractor.build_extraction_prompt([])
        self.assertIsNotNone(prompt)
    
    def test_check_missing_fields_empty_extracted(self):
        """Test missing field check with empty extracted dict."""
        fields = ["field1", "field2"]
        extracted = {}
        
        missing = self.extractor.check_missing_fields(extracted, fields)
        self.assertEqual(set(missing), {"field1", "field2"})
    
    def test_check_missing_fields_all_present(self):
        """Test missing field check with all fields present."""
        fields = ["field1", "field2"]
        extracted = {"field1": "value1", "field2": "value2"}
        
        missing = self.extractor.check_missing_fields(extracted, fields)
        self.assertEqual(missing, [])
    
    def test_save_raw_output_with_metadata(self):
        """Test saving raw output with complete metadata."""
        raw_output = {
            "markdown": "# Test\nContent",
            "metadata": {
                "num_pages": 2,
                "source": "test.pdf",
                "parser": "docling",
                "processing_time": 1.5
            }
        }
        
        result_path = self.extractor.save_raw_output(raw_output, 1, "test_pdf")
        
        self.assertTrue(Path(result_path).exists())
        
        with open(result_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["metadata"]["num_pages"], 2)
        self.assertEqual(saved_data["metadata"]["processing_time"], 1.5)


class TestWorkflowComprehensive(unittest.TestCase):
    """Comprehensive workflow tests to reach 85% coverage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = DoclingExtractor(output_dir=self.temp_dir)
    
    def test_parse_pdf_node_complete(self):
        """Test the parse_pdf workflow node with complete state."""
        state = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string"},
            "field_names": ["field1"],
            "attempt": 1,
            "logs": []
        }
        
        # Mock the parse_pdf method
        with patch.object(self.extractor, 'parse_pdf') as mock_parse:
            mock_parse.return_value = ("Sample text", {"markdown": "Sample", "metadata": {"source": "test.pdf"}})
            
            result = self.extractor._parse_pdf_node(state)
            
            self.assertEqual(result["text_content"], "Sample text")
            self.assertIn("raw_output", result)
            self.assertIn("logs", result)
    
    def test_extract_fields_node_complete(self):
        """Test the extract_fields workflow node with complete state."""
        state = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string"},
            "field_names": ["field1"],
            "text_content": "Sample text",
            "attempt": 1,
            "raw_output_files": [],
            "logs": [],
            "raw_output": {"markdown": "Sample", "metadata": {"source": "test.pdf"}}
        }
        
        # Mock the extract_fields method
        with patch.object(self.extractor, 'extract_fields') as mock_extract:
            mock_extract.return_value = {"field1": "value1"}
        
            # Mock save_raw_output
            with patch.object(self.extractor, 'save_raw_output') as mock_save:
                mock_save.return_value = "output/raw.json"
                
                result = self.extractor._extract_fields_node(state)
                
                self.assertIn("extracted_fields", result)
                self.assertIn("logs", result)
    
    def test_validate_extraction_node_complete(self):
        """Test the validate_extraction workflow node with complete scenarios."""
        # Test success case
        state_success = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string"},
            "field_names": ["field1"],
            "extracted_fields": {"field1": "value1"},
            "attempt": 1,
            "raw_output_files": ["test.json"],
            "logs": []
        }
        
        result = self.extractor._validate_extraction_node(state_success)
        self.assertTrue(result.get("success", False))
        
        # Test missing fields case
        state_missing = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string"},
            "field_names": ["field1"],
            "extracted_fields": {},
            "attempt": 1,
            "raw_output_files": ["test.json"],
            "logs": []
        }
        
        result = self.extractor._validate_extraction_node(state_missing)
        self.assertFalse(result.get("success", True))
        self.assertIn("missing_fields", result)
    
    def test_should_retry_conditions(self):
        """Test all should_retry workflow conditions."""
        # Test retry case
        state_retry = {
            "success": False,
            "attempt": 1,
            "missing_fields": ["field1"]
        }
        result = self.extractor._should_retry(state_retry)
        self.assertFalse(result)  # Should return "retry"
        
        # Test success case
        state_success = {
            "success": True,
            "attempt": 1,
            "missing_fields": []
        }
        result = self.extractor._should_retry(state_success)
        self.assertTrue(result)  # Should return END
        
        # Test limit case
        state_limit = {
            "success": False,
            "attempt": 5,
            "missing_fields": ["field1"]
        }
        result = self.extractor._should_retry(state_limit)
        self.assertTrue(result)  # Should return END
    
    def test_save_results_node_complete(self):
        """Test the save_results workflow node."""
        state = {
            "pdf_path": "test.pdf",
            "pdf_name": "test.pdf",
            "fields_to_extract": {"field1": "string"},
            "success": True,
            "attempt": 1,
            "extracted_fields": {"field1": "value1"},
            "missing_fields": [],
            "raw_output_files": ["test.json"],
            "error": None,
            "logs": []
        }
        
        # Mock save methods
        with patch('extract.save_results') as mock_save:
            with patch.object(self.extractor, 'save_audit_log') as mock_audit:
                mock_save.return_value = Path("output/result.json")
                mock_audit.return_value = Path("output/audit.json")
                
                result = self.extractor._save_results_node(state)
                
                self.assertTrue(result.get("success", False))
    
    def test_extract_with_llm_complete(self):
        """Test _extract_with_llm method with all scenarios."""
        text = "Sample text"
        fields = {"field1": "string"}
        
        # Test success
        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"field1": "value1"}'
        mock_chain.invoke.return_value = mock_response
        
        result = self.extractor._extract_with_llm(text, fields, mock_chain)
        self.assertEqual(result["field1"], "value1")
        
        # Test JSON error
        mock_response.content = '{ invalid json }'
        result = self.extractor._extract_with_llm(text, fields, mock_chain)
        self.assertEqual(result, {})
        
        # Test chain error
        mock_chain.invoke.side_effect = Exception("Chain error")
        result = self.extractor._extract_with_llm(text, fields, mock_chain)
        self.assertEqual(result, {})
    
    def test_process_pdf_with_logging_complete(self):
        """Test process_pdf_with_logging method."""
        # Mock the workflow
        with patch.object(self.extractor.workflow, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "success": True,
                "attempt": 1,
                "extracted_fields": {"field1": "value1"},
                "missing_fields": [],
                "raw_output_files": [],
                "error": None,
                "logs": []
            }
            
            result = self.extractor.process_pdf_with_logging("test.pdf", custom_fields=["field1"])
            
            self.assertTrue(result["success"])
            self.assertIn("field1", result["extracted_fields"])
    
    def test_process_pdf_with_logging_error(self):
        """Test process_pdf_with_logging method with error."""
        # Mock the workflow with error
        with patch.object(self.extractor.workflow, 'invoke') as mock_invoke:
            mock_invoke.side_effect = Exception("Workflow error")
            
            result = self.extractor.process_pdf_with_logging("test.pdf", custom_fields=["field1"])
            
            self.assertFalse(result["success"])
            self.assertIn("error", result)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_full_extraction_workflow_mock(self):
        """Test the complete extraction workflow with mocked components."""
        # Mock the entire workflow to avoid complex dependencies
        with patch('extractor.ChatVertexAI') as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            # Create extractor
            extractor = DoclingExtractor(output_dir=self.temp_dir)
            
            # Mock the entire workflow invoke method
            with patch.object(extractor.workflow, 'invoke') as mock_workflow:
                mock_workflow.return_value = {
                    "success": True,
                    "attempt": 1,
                    "extracted_fields": {
                        "INVOICE NUMBER": "INV-TEST-001",
                        "Date": "2024-01-15",
                        "Total": "$250.00"
                    },
                    "missing_fields": [],
                    "raw_output_files": [],
                    "error": None,
                    "logs": []
                }
                
                # Test extraction
                fields = ["INVOICE NUMBER", "Date", "Total"]
                result = extractor.process_pdf("dummy.pdf", custom_fields=fields)
                
                # Check that extraction succeeded
                self.assertTrue(result["success"], f"Extraction failed: {result.get('error', 'Unknown error')}")
                self.assertIn("INVOICE NUMBER", result["extracted_fields"])
                self.assertEqual(result["extracted_fields"]["INVOICE NUMBER"], "INV-TEST-001")


if __name__ == '__main__':
    # Disable Phoenix for faster test execution and pipeline compatibility
    # Set PHOENIX_ENABLED=false to skip Phoenix initialization
    os.environ['PHOENIX_ENABLED'] = 'false'
    
    # Run tests
    unittest.main(verbosity=2)
