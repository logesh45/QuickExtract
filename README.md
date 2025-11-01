# PDF Field Extraction CLI

A powerful command-line tool for extracting specific fields from PDF documents using Docling and Google Gemini AI with Arize Phoenix observability.

## Features

- 🚀 **Fast PDF Processing**: Uses Docling for efficient PDF parsing
- 🤖 **AI-Powered Extraction**: Google Gemini LLM for intelligent field extraction
- 📊 **Observability**: Built-in Arize Phoenix integration for tracing and monitoring
- 🔧 **Configurable Fields**: Define custom fields and types via JSON configuration
- 📝 **Detailed Logging**: Comprehensive extraction logs and audit trails
- 🧪 **Well Tested**: 70% test coverage with comprehensive test suite
- ⚡ **Pipeline Ready**: Fast test execution without Phoenix dependencies
- 🔄 **Retry Logic**: Automatic retry mechanism for failed extractions

## Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud Project with Vertex AI API enabled
- Google Cloud credentials configured

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd docling-flow
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Google Cloud project ID
   ```

5. **Authenticate with Google Cloud**
   ```bash
   gcloud auth application-default login
   ```

### Usage

#### Basic Extraction

```bash
python extract.py fields.json invoice-sample.pdf
```

#### Custom Retry Attempts

```bash
# Use custom retry limit (default: 5)
python extract.py fields.json invoice-sample.pdf --max-attempts 3

# Fast processing with single attempt
python extract.py fields.json invoice-sample.pdf --max-attempts 1

# High reliability with more attempts
python extract.py fields.json invoice-sample.pdf --max-attempts 10
```

#### Custom Fields Configuration

Create a `fields.json` file with your desired fields:

```json
{
  "INVOICE NUMBER": "string",
  "Date": "date",
  "Total": "number",
  "BILL TO address": "string",
  "Shipping Charges": "number",
  "Insurance": "number"
}
```

#### Supported Field Types

- `string`: Text values
- `number`: Numeric values (integers and decimals)
- `date`: Date values

### Output

Results are saved to `output/extracted_[pdf_name].json`:

```json
{
  "success": true,
  "attempts": 1,
  "extracted_fields": {
    "INVOICE NUMBER": "F1000876/23",
    "Date": "14/08/2023",
    "Total": "$93.50",
    "BILL TO address": "IMPORTING COMPANY\n100 Mighty Bay\n125863 Rome, IT",
    "Shipping Charges": 100,
    "Insurance": 0
  },
  "missing_fields": [],
  "raw_output_files": ["raw_outputs/invoice-sample_attempt_1_20251027_210515.json"],
  "error": null
}
```

## Arize Phoenix Integration

The tool includes built-in observability with Arize Phoenix:

- **Automatic Tracing**: All LLM calls are automatically traced
- **Local Phoenix UI**: Launches locally at `http://localhost:6006`
- **Custom Project**: Uses project name "pdf-extraction" (configurable via `PHOENIX_PROJECT_NAME`)
- **Pipeline Compatible**: Can be disabled for CI/CD environments

### Phoenix Configuration

Optional environment variables for Phoenix:

```bash
PHOENIX_PROJECT_NAME=pdf-extraction
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_ENABLED=false  # Disable for faster execution/pipelines
```

## Testing

### Fast Test Execution (Recommended)

```bash
# Run tests without Phoenix (3x faster, pipeline compatible)
python run_tests.py

# Run tests with coverage report
python run_tests.py --coverage

# Verbose output
python run_tests.py --verbose
```

### Test Options

```bash
# Enable Phoenix for debugging tests
python run_tests.py --phoenix

# Run integration test only
python run_tests.py --integration

# Coverage with Phoenix enabled
python run_tests.py --phoenix --coverage
```

### Test Coverage

The test suite includes:
- **70% overall coverage** (92% for extract.py, 65% for extractor.py)
- Unit tests for all core components
- CLI functionality tests
- Phoenix integration tests
- End-to-end workflow tests
- Mock-based testing to avoid API costs
- Fast execution mode (3.4 seconds vs 11+ seconds with Phoenix)

### Running Tests Directly

```bash
# Run all tests with unittest
python -m unittest test_extractor.py -v

# Run specific test classes
python -m unittest test_extractor.TestDoclingExtractor -v
```

## Development

### Project Structure

```
.
├── extractor.py          # Core extraction logic with Docling + Gemini
├── extract.py            # CLI interface
├── test_extractor.py     # Comprehensive test suite (42 tests)
├── run_tests.py          # Enhanced test runner with Phoenix control
├── fields.json           # Sample field configuration
├── invoice-sample.pdf    # Sample invoice for testing
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── htmlcov/             # HTML coverage reports (generated)
├── output/              # Extraction results
├── raw_outputs/         # Raw extraction logs
└── README.md            # This file
```

### Key Components

- **DoclingExtractor**: Main extraction class with LangGraph workflow
- **CLI Interface**: Argument parsing and file handling
- **Phoenix Integration**: Automatic LLM tracing and observability
- **Test Suite**: Comprehensive testing with mocking and fast execution modes
- **Retry Logic**: Automatic retry with configurable limits
- **Audit Trails**: Detailed logging and raw output preservation

### Workflow Architecture

The extraction process uses a LangGraph workflow with the following nodes:

1. **Parse PDF**: Extract text and metadata using Docling
2. **Extract Fields**: Use Gemini LLM to extract specified fields
3. **Validate Extraction**: Check for missing fields and determine retry need
4. **Save Results**: Store extracted data and audit logs

## Configuration

### Environment Variables

Required:
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID

Optional:
- `GOOGLE_CLOUD_LOCATION`: GCP region (default: `us-central1`)
- `PHOENIX_PROJECT_NAME`: Phoenix project name (default: `pdf-extraction`)
- `PHOENIX_COLLECTOR_ENDPOINT`: Phoenix endpoint (default: `http://localhost:6006`)
- `PHOENIX_ENABLED`: Enable/disable Phoenix (default: `true`, set to `false` for pipelines)

### Field Configuration Tips

1. **Be Specific**: Use exact field names from your documents
2. **Include Variations**: Add multiple similar fields if needed
3. **Choose Right Types**: Use appropriate field types for better extraction
4. **Test Iteratively**: Start with a few fields, then expand

### Retry Configuration

The extractor automatically retries failed extractions with configurable limits:

```bash
# Default: 5 retry attempts
python extract.py fields.json invoice.pdf

# Custom retry limit
python extract.py fields.json invoice.pdf --max-attempts 3

# Single attempt (fastest)
python extract.py fields.json invoice.pdf --max-attempts 1
```

**Retry Behavior:**
- Progressive retry delays
- Missing field validation after each attempt
- Comprehensive error logging
- Early termination when all fields are successfully extracted
- Marks as success after max attempts to avoid infinite loops

## Troubleshooting

### Common Issues

1. **Google Cloud Authentication**
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Vertex AI API Not Enabled**
   - Enable Vertex AI API in your Google Cloud console
   - Ensure proper IAM permissions

3. **Phoenix UI Not Accessible**
   - Check if port 6006 is available
   - Verify Phoenix installation: `pip install arize-phoenix`
   - Use `PHOENIX_ENABLED=false` to disable if not needed

4. **Tests Running Slow**
   - Use `python run_tests.py` for fast execution (Phoenix disabled)
   - Avoid `--phoenix` flag unless debugging tracing issues

### Debug Mode

For detailed debugging, the extractor provides:
- Raw output files in `raw_outputs/`
- Comprehensive audit logs
- Step-by-step extraction logs
- Phoenix tracing (when enabled)

## Performance

- **Fast Test Mode**: 3.4 seconds execution time
- **Standard Mode**: 11+ seconds with Phoenix enabled
- **Pipeline Optimized**: Phoenix disabled by default in test runner
- **Memory Efficient**: Streaming PDF processing for large files

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `python run_tests.py`
5. Check coverage: `python run_tests.py --coverage`
6. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the test files for usage examples
- Open an issue on GitHub
