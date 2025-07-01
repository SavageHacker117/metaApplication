
#!/bin/bash
# Run all tests for the RL-LLM development tool

# The script expects to be run from the v8.2 directory or its subdirectories
# Adjust TEST_DIR relative to the current script's location
TEST_DIR="../tests"

echo "Running unit tests..."
python3 -m unittest discover $TEST_DIR/unit_tests

echo "Running integration tests..."
python3 -m unittest discover $TEST_DIR/integration_tests

echo "Running performance tests..."
python3 -m unittest discover $TEST_DIR/performance_tests

echo "All tests completed."


