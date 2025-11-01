#!/usr/bin/env python3
"""
Test runner for the PDF extractor CLI.
Run all tests and generate coverage report.
"""
import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the test suite with coverage."""
    print("🧪 Running PDF Extractor Tests...")
    print("=" * 50)
    
    # Install coverage if not available
    try:
        import coverage
    except ImportError:
        print("📦 Installing coverage...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
    
    # Run tests with coverage
    try:
        # Start coverage
        cov = coverage.Coverage(source=['extractor', 'extract'])
        cov.start()
        
        # Import and run tests
        import unittest
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.discover('.', pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Stop coverage
        cov.stop()
        cov.save()
        
        print("\n" + "=" * 50)
        print("📊 Coverage Report:")
        cov.report()
        
        # Generate HTML coverage report
        cov.html_report(directory='htmlcov')
        print("📁 HTML coverage report generated in 'htmlcov/' directory")
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_integration_test():
    """Run a quick integration test with the sample PDF."""
    print("\n🔧 Running Integration Test...")
    print("=" * 50)
    
    try:
        # Test CLI with sample files
        cmd = [sys.executable, "extract.py", "fields.json", "invoice-sample.pdf"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Integration test passed!")
            print("📄 Sample output:")
            print(result.stdout)
            return True
        else:
            print("❌ Integration test failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Integration test timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running integration test: {e}")
        return False

if __name__ == "__main__":
    print("🚀 PDF Extractor Test Suite")
    print("=" * 50)
    
    # Set environment for testing
    import os
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'test-project'
    os.environ['PHOENIX_PROJECT_NAME'] = 'test-extraction'
    
    # Run unit tests
    unit_tests_passed = run_tests()
    
    # Run integration test if unit tests pass
    if unit_tests_passed:
        integration_tests_passed = run_integration_test()
        
        if unit_tests_passed and integration_tests_passed:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed!")
            sys.exit(1)
    else:
        print("\n💥 Unit tests failed!")
        sys.exit(1)
