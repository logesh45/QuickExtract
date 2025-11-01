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
from unittest.mock import patch, MagicMock

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from extractor import DoclingExtractor
from extract import load_fields_config, validate_pdf_file, save_results, create_output_directory


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


class TestPhoenixIntegration(unittest.TestCase):
    """Test cases for Arize Phoenix integration."""
    
    def test_phoenix_config_variables(self):
        """Test Phoenix configuration environment variables."""
        # Check default values
        self.assertEqual(DoclingExtractor.PHOENIX_PROJECT_NAME, "pdf-extraction")
        self.assertEqual(DoclingExtractor.PHOENIX_COLLECTOR_ENDPOINT, "http://localhost:6006")
    
    @patch.dict(os.environ, {'PHOENIX_PROJECT_NAME': 'test-project'})
    def test_phoenix_env_override(self):
        """Test Phoenix configuration via environment variables."""
        # This should pick up the environment variable
        extractor = DoclingExtractor()
        self.assertIsNotNone(extractor)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_full_extraction_workflow_mock(self):
        """Test the complete extraction workflow with mocked components."""
        # Mock the LLM to avoid actual API calls
        with patch('extractor.ChatVertexAI') as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            # Mock response
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "INVOICE NUMBER": "INV-TEST-001",
                "Date": "2024-01-15",
                "Total": "$250.00"
            })
            
            # Create extractor
            extractor = DoclingExtractor(output_dir=self.temp_dir)
            
            # Mock the extraction method
            with patch.object(extractor, '_extract_with_llm') as mock_extract:
                mock_extract.return_value = {
                    "INVOICE NUMBER": "INV-TEST-001",
                    "Date": "2024-01-15",
                    "Total": "$250.00"
                }
                
                # Mock PDF parsing
                with patch.object(extractor, 'parse_pdf') as mock_parse:
                    mock_parse.return_value = (
                        "Sample invoice content",
                        {"markdown": "Sample", "metadata": {"source": "test"}}
                    )
                    
                    # Test extraction
                    fields = ["INVOICE NUMBER", "Date", "Total"]
                    result = extractor.process_pdf("dummy.pdf", custom_fields=fields)
                    
                    self.assertTrue(result["success"])
                    self.assertIn("INVOICE NUMBER", result["extracted_fields"])


if __name__ == '__main__':
    # Configure test environment
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'test-project'
    os.environ['PHOENIX_PROJECT_NAME'] = 'test-extraction'
    
    # Run tests
    unittest.main(verbosity=2)
