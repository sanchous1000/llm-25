#!/usr/bin/env python3
"""
Simple test script to verify Langfuse v2 connection.
Tests basic trace logging functionality.
"""
import os
import sys
from datetime import datetime

try:
    from langfuse import Langfuse
except ImportError:
    print("ERROR: langfuse package not installed")
    print("Installing langfuse...")
    os.system("pip install langfuse")
    from langfuse import Langfuse


def test_langfuse_connection():
    """Test basic Langfuse connection and trace logging."""
    
    # Get credentials from environment
    host = os.getenv('LANGFUSE_HOST', 'http://langfuse:3000')
    public_key = os.getenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-lab3-public-key')
    secret_key = os.getenv('LANGFUSE_SECRET_KEY', 'sk-lf-lab3-secret-key')
    
    print("=" * 60)
    print("Langfuse v2 Connection Test")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Public Key: {public_key[:20]}...")
    print(f"Secret Key: {secret_key[:20]}...")
    print()
    
    try:
        # Initialize Langfuse client
        print("Initializing Langfuse client...")
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        print("✓ Client initialized successfully")
        print()
        
        # Create a test trace
        print("Creating test trace...")
        trace = langfuse.trace(
            name="test-trace",
            user_id="test-user",
            metadata={
                "test": True,
                "timestamp": datetime.now().isoformat(),
                "source": "lab3-test-script"
            }
        )
        print(f"✓ Trace created with ID: {trace.id}")
        print()
        
        # Add a span to the trace
        print("Adding span to trace...")
        span = trace.span(
            name="test-span",
            metadata={"operation": "test"}
        )
        print(f"✓ Span created")
        print()
        
        # Log a generation
        print("Logging generation...")
        generation = trace.generation(
            name="test-generation",
            model="test-model",
            model_parameters={"temperature": 0.7},
            input="Test input",
            output="Test output"
        )
        print(f"✓ Generation logged")
        print()
        
        # Flush to ensure data is sent
        print("Flushing data to Langfuse...")
        langfuse.flush()
        print("✓ Data flushed successfully")
        print()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("You can view the trace in Langfuse UI at:")
        print(f"http://localhost:3000")
        print()
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Check that Langfuse is running: docker ps")
        print("2. Check Langfuse logs: docker logs langfuse-lab3")
        print("3. Verify network connectivity")
        print("4. Confirm environment variables are set correctly")
        return False


if __name__ == "__main__":
    success = test_langfuse_connection()
    sys.exit(0 if success else 1)
