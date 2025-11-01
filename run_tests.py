#!/usr/bin/env python3
"""
Test runner script with Phoenix control options.

Usage:
    python run_tests.py [--phoenix] [--coverage] [--verbose] [--integration]

Options:
    --phoenix     Enable Phoenix tracing during tests (default: disabled for speed)
    --coverage    Run tests with coverage report
    --verbose     Run tests with verbose output
    --integration Run integration tests with sample PDF
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(phoenix_enabled=False, coverage=False, verbose=False):
    """Run tests with specified options."""
    
    # Set environment variables
    env = os.environ.copy()
    
    if phoenix_enabled:
        env['PHOENIX_ENABLED'] = 'true'
        print("🔥 Running tests WITH Phoenix tracing enabled...")
    else:
        env['PHOENIX_ENABLED'] = 'false'
        print("⚡ Running tests WITHOUT Phoenix for faster execution...")
    
    # Build command
    if coverage:
        print("📊 Running with coverage report...")
        # Install coverage if not available
        try:
            import coverage
        except ImportError:
            print("📦 Installing coverage...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
        
        # Run with coverage
        cmd = [
            sys.executable, '-m', 'coverage', 'run',
            '--source=extractor,extract', '-m', 'unittest'
        ]
        if verbose:
            cmd.append('-v')
        
        # Test modules to run
        test_modules = [
            'test_extractor.TestDoclingExtractor',
            'test_extractor.TestCLIFunctions', 
            'test_extractor.TestAdditionalExtractorMethods',
            'test_extractor.TestPhoenixIntegration',
            'test_extractor.TestEdgeCases',
            'test_extractor.TestEndToEnd'
        ]
        cmd.extend(test_modules)
        
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())
        
        if result.returncode == 0:
            # Generate coverage report
            print("\n" + "="*50)
            print("📊 COVERAGE REPORT")
            print("="*50)
            subprocess.run([
                sys.executable, '-m', 'coverage', 'report'
            ], env=env, cwd=os.getcwd())
            
            # Generate HTML coverage report
            subprocess.run([
                sys.executable, '-m', 'coverage', 'html'
            ], env=env, cwd=os.getcwd())
            print("📁 HTML coverage report generated in 'htmlcov/' directory")
        else:
            print(f"Tests failed with return code: {result.returncode}")
            return False
    else:
        # Run without coverage using unittest directly
        cmd = [sys.executable, '-m', 'unittest']
        if verbose:
            cmd.append('-v')
        
        # Test modules to run
        test_modules = [
            'test_extractor.TestDoclingExtractor',
            'test_extractor.TestCLIFunctions', 
            'test_extractor.TestAdditionalExtractorMethods',
            'test_extractor.TestPhoenixIntegration',
            'test_extractor.TestEdgeCases',
            'test_extractor.TestEndToEnd'
        ]
        cmd.extend(test_modules)
        
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"Tests failed with return code: {result.returncode}")
            return False
    
    return True


def run_integration_test(phoenix_enabled=False):
    """Run a quick integration test with the sample PDF."""
    print("\n🔧 Running Integration Test...")
    print("=" * 50)
    
    # Set environment variables
    env = os.environ.copy()
    
    if phoenix_enabled:
        env['PHOENIX_ENABLED'] = 'true'
        print("🔥 Integration test WITH Phoenix tracing...")
    else:
        env['PHOENIX_ENABLED'] = 'false'
        print("⚡ Integration test WITHOUT Phoenix for faster execution...")
    
    try:
        # Test CLI with sample files
        cmd = [sys.executable, "extract.py", "fields.json", "invoice-sample.pdf"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests with Phoenix control options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py                    # Fast tests without Phoenix
    python run_tests.py --phoenix          # Tests with Phoenix enabled
    python run_tests.py --coverage         # Tests with coverage (no Phoenix)
    python run_tests.py --phoenix --coverage  # Tests with coverage and Phoenix
    python run_tests.py --integration      # Run integration test only
        """
    )
    
    parser.add_argument(
        '--phoenix',
        action='store_true',
        help='Enable Phoenix tracing during tests (default: disabled for speed)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true', 
        help='Run tests with coverage report'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Run tests with verbose output'
    )
    
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run integration test with sample PDF'
    )
    
    args = parser.parse_args()
    
    print("🚀 PDF Extractor Test Suite")
    print("=" * 50)
    
    # Run integration test only if requested
    if args.integration:
        success = run_integration_test(phoenix_enabled=args.phoenix)
        sys.exit(0 if success else 1)
    
    # Run unit tests
    unit_tests_passed = run_tests(
        phoenix_enabled=args.phoenix,
        coverage=args.coverage,
        verbose=args.verbose
    )
    
    # Run integration test if unit tests pass
    if unit_tests_passed:
        integration_tests_passed = run_integration_test(phoenix_enabled=args.phoenix)
        
        if unit_tests_passed and integration_tests_passed:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed!")
            sys.exit(1)
    else:
        print("\n💥 Unit tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
