#!/bin/bash
# Test runner script for Rapido system

set -e

echo "🧪 Running Rapido Test Suite"
echo "============================"

# Check if we're in the right directory
if [ ! -f "src/rapido_refactored_main.py" ]; then
    echo "❌ Error: Please run this script from the rapido project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install -q pytest pytest-asyncio pytest-cov

# Set test environment variables
export JWT_SECRET="test_secret_key_12345"
export RAPIDO_TEST_MODE="true"
export RAPIDO_LOG_DIR="./test_logs"

# Create test directories
mkdir -p test_logs
mkdir -p test_data/presentation_frames/test_lesson_001

# Create test data files
echo "📝 Creating test data..."
cat > test_data/test_lesson.json << EOF
{
  "slide_data": {
    "slideId": "test_lesson_001",
    "narrationData": {
      "text": "This is a test narration for the comprehensive test suite.",
      "timing": [
        {"start": 0.0, "end": 3.0, "text": "This is a test narration"},
        {"start": 3.0, "end": 6.0, "text": "for the comprehensive test suite."}
      ]
    }
  }
}
EOF

# Create dummy slide frames
for i in {1..3}; do
    # Create minimal PNG header + data
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x64\x00\x00\x00\x64\x08\x02\x00\x00\x00\xff\x80\x02\x03' > test_data/presentation_frames/test_lesson_001/slide_$(printf "%03d" $i).png
    printf '\x00\x00\x00\x00IEND\xaeB`\x82' >> test_data/presentation_frames/test_lesson_001/slide_$(printf "%03d" $i).png
done

echo "🔬 Running Unit Tests..."
echo "----------------------"
pytest tests/unit/ -v --tb=short

echo ""
echo "🔗 Running Integration Tests..."
echo "------------------------------"
pytest tests/integration/ -v --tb=short

echo ""
echo "📊 Running Tests with Coverage..."
echo "--------------------------------"
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

echo ""
echo "🧹 Cleaning up test data..."
rm -rf test_data test_logs htmlcov

echo ""
echo "✅ All tests completed successfully!"
echo "📋 Test Summary:"
echo "   - Unit tests: Passed"
echo "   - Integration tests: Passed"
echo "   - Coverage report: Generated in htmlcov/"
echo ""
echo "🚀 System is ready for deployment!"
