"""
Pytest configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require network)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture
def sample_math_problem():
    """Sample math problem for testing."""
    return {
        "id": "test_math_1",
        "problem": "John has 5 apples. He buys 3 more apples. How many apples does John have now?",
        "answer": "8",
    }


@pytest.fixture
def sample_code_problem():
    """Sample code problem for testing."""
    return {
        "id": "test_code_1",
        "problem": """def add(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
""",
        "answer": "    return a + b",
        "entry_point": "add",
        "test": """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
    assert candidate(0, 0) == 0
""",
    }


@pytest.fixture
def sample_reasoning_path():
    """Sample reasoning path for testing."""
    return {
        "problem_id": "test_1",
        "problem": "What is 5 + 3?",
        "reasoning": "To find 5 + 3, we add the numbers together. 5 + 3 = 8. #### 8",
        "final_answer": "8",
        "expected_answer": "8",
        "is_correct": True,
    }


@pytest.fixture
def sample_dpo_pair():
    """Sample DPO training pair for testing."""
    return {
        "prompt": "What is 5 + 3?",
        "chosen": "To solve this, I add 5 and 3. 5 + 3 = 8. The answer is 8.",
        "rejected": "5 + 3 = 9. The answer is 9.",
    }
