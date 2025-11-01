#!/usr/bin/env python3
"""
QuickExtract - Fast PDF field extraction CLI.
Usage: extract.py fields.json invoice-sample.pdf

The fields.json defines what fields to extract and their types:
{
  "shippingCharges": "number",
  "Insurance": "number",
  "BILL TO address": "string",
  "INVOICE NUMBER": "string",
  "Date": "date"
}

Output is saved to output/extracted_[pdf_name].json
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

from extractor import DoclingExtractor


def load_fields_config(fields_file: str) -> Dict[str, str]:
    """Load field configuration from JSON file."""
    try:
        with open(fields_file, 'r') as f:
            config = json.load(f)
        
        # Validate config format
        if not isinstance(config, dict):
            raise ValueError("Fields configuration must be a JSON object")
        
        for field_name, field_type in config.items():
            if not isinstance(field_name, str) or not isinstance(field_type, str):
                raise ValueError(f"Invalid field configuration: {field_name}: {field_type}")
            
            if field_type not in ['string', 'number', 'date']:
                raise ValueError(f"Unsupported field type '{field_type}' for field '{field_name}'. Supported types: string, number, date")
        
        return config
    
    except FileNotFoundError:
        print(f"Error: Fields configuration file '{fields_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in fields configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def validate_pdf_file(pdf_file: str) -> Path:
    """Validate PDF file exists and is readable."""
    pdf_path = Path(pdf_file)
    
    if not pdf_path.exists():
        print(f"Error: PDF file '{pdf_file}' not found")
        sys.exit(1)
    
    if not pdf_path.is_file():
        print(f"Error: '{pdf_file}' is not a file")
        sys.exit(1)
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"Error: '{pdf_file}' is not a PDF file")
        sys.exit(1)
    
    return pdf_path


def create_output_directory() -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_results(results: Dict[str, Any], pdf_name: str, output_dir: Path) -> Path:
    """Save extraction results to JSON file."""
    output_file = output_dir / f"extracted_{pdf_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return output_file


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QuickExtract: Extract specific fields from PDF documents using Docling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  extract.py fields.json invoice-sample.pdf
  extract.py config.json document.pdf
        """
    )
    
    parser.add_argument('fields_file', help='JSON file containing field definitions and types')
    parser.add_argument('pdf_file', help='PDF file to extract fields from')
    parser.add_argument(
        '--max-attempts', 
        type=int, 
        default=5,
        help='Maximum retry attempts for extraction (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Load field configuration
    print(f"Loading field configuration from: {args.fields_file}")
    fields_config = load_fields_config(args.fields_file)
    print(f"Found {len(fields_config)} fields to extract")
    
    # Validate PDF file
    pdf_path = validate_pdf_file(args.pdf_file)
    print(f"Processing PDF: {pdf_path}")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Initialize extractor
    print("Initializing Docling extractor...")
    extractor = DoclingExtractor(max_attempts=args.max_attempts)
    
    # Process PDF
    print("Extracting fields...")
    try:
        result = extractor.process_pdf_with_logging(
            str(pdf_path), 
            custom_fields=fields_config
        )
        
        # Save results
        output_file = save_results(result, pdf_path.stem, output_dir)
        
        # Display results
        print(f"\n✅ Extraction completed successfully!")
        print(f"📁 Results saved to: {output_file}")
        
        if result.get('success'):
            print(f"📊 Extracted {len(result.get('extracted_fields', {}))} fields")
            
            # Show extracted fields
            extracted = result.get('extracted_fields', {})
            if extracted:
                print("\n📋 Extracted fields:")
                for field, value in extracted.items():
                    if value:
                        print(f"  • {field}: {value}")
            
            # Show missing fields
            missing = result.get('missing_fields', [])
            if missing:
                print(f"\n⚠️  Missing fields: {', '.join(missing)}")
        else:
            print(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
