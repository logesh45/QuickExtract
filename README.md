# PDF Field Extraction CLI

A powerful command-line tool for extracting specific fields from PDF documents using Docling and Google Gemini AI with Arize Phoenix observability.

## Features

- 🚀 **Fast PDF Processing**: Uses Docling for efficient PDF parsing
- 🤖 **AI-Powered Extraction**: Google Gemini LLM for intelligent field extraction
- 📊 **Observability**: Built-in Arize Phoenix integration for tracing and monitoring
- 🔧 **Configurable Fields**: Define custom fields and types via JSON configuration
- 📝 **Detailed Logging**: Comprehensive extraction logs and audit trails
- 🧪 **Well Tested**: Comprehensive test suite included

## Quick Start

### Prerequisites

- Python 3.8+
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

### Phoenix Configuration

Optional environment variables for Phoenix:

```bash
PHOENIX_PROJECT_NAME=pdf-extraction
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

## Testing

### Run All Tests

```bash
python run_tests.py
```

### Run Unit Tests Only

```bash
python -m unittest test_extractor.py -v
```

### Test Coverage

The test suite includes:
- Unit tests for all core components
- CLI functionality tests
- Phoenix integration tests
- End-to-end workflow tests
- Mock-based testing to avoid API costs

## Development

### Project Structure

```
.
├── extractor.py          # Core extraction logic with Docling + Gemini
├── extract.py            # CLI interface
├── test_extractor.py     # Comprehensive test suite
├── run_tests.py          # Test runner with coverage
├── fields.json           # Sample field configuration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

### Key Components

- **DoclingExtractor**: Main extraction class with LangGraph workflow
- **CLI Interface**: Argument parsing and file handling
- **Phoenix Integration**: Automatic LLM tracing and observability
- **Test Suite**: Comprehensive testing with mocking

## Configuration

### Environment Variables

Required:
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID

Optional:
- `GOOGLE_CLOUD_LOCATION`: GCP region (default: `us-central1`)
- `PHOENIX_PROJECT_NAME`: Phoenix project name (default: `pdf-extraction`)
- `PHOENIX_COLLECTOR_ENDPOINT`: Phoenix endpoint (default: `http://localhost:6006`)

### Field Configuration Tips

1. **Be Specific**: Use exact field names from your documents
2. **Include Variations**: Add multiple similar fields if needed
3. **Choose Right Types**: Use appropriate field types for better extraction
4. **Test Iteratively**: Start with a few fields, then expand

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

### Debug Mode

For detailed debugging, the extractor provides:
- Raw output files in `raw_outputs/`
- Comprehensive audit logs
- Step-by-step extraction logs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the test files for usage examples
- Open an issue on GitHub
