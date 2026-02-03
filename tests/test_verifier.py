"""
Tests for the verifier module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verifier import MathVerifier, GSM8KVerifier, VerificationStatus


class TestMathVerifier:
    """Tests for MathVerifier."""
    
    def setup_method(self):
        self.verifier = MathVerifier()
    
    def test_exact_match(self):
        """Test exact string match."""
        result = self.verifier.verify("42", "42")
        assert result.status == VerificationStatus.CORRECT
    
    def test_numeric_comparison(self):
        """Test numeric comparison."""
        result = self.verifier.verify("42.0", "42")
        assert result.status == VerificationStatus.CORRECT
    
    def test_fraction_equivalence(self):
        """Test fraction equivalence."""
        result = self.verifier.verify("0.5", "1/2")
        assert result.status == VerificationStatus.CORRECT
    
    def test_incorrect_answer(self):
        """Test incorrect answer detection."""
        result = self.verifier.verify("41", "42")
        assert result.status == VerificationStatus.INCORRECT
    
    def test_extract_boxed_answer(self):
        """Test extraction of boxed LaTeX answers."""
        text = "The answer is \\boxed{42}."
        answer = self.verifier.extract_final_answer(text)
        assert answer == "42"
    
    def test_extract_gsm8k_answer(self):
        """Test extraction of GSM8K format answers."""
        text = "So the total is 42.\n#### 42"
        answer = self.verifier.extract_final_answer(text)
        assert answer == "42"
    
    def test_normalize_currency(self):
        """Test currency normalization."""
        normalized = self.verifier.normalize_answer("$100")
        assert normalized == "100"


class TestGSM8KVerifier:
    """Tests for GSM8KVerifier."""
    
    def setup_method(self):
        self.verifier = GSM8KVerifier()
    
    def test_gsm8k_format(self):
        """Test GSM8K answer format."""
        reasoning = """
        Step 1: Calculate 5 + 5 = 10
        Step 2: Calculate 10 * 2 = 20
        #### 20
        """
        result = self.verifier.verify_reasoning_path(reasoning, "20")
        assert result.status == VerificationStatus.CORRECT
    
    def test_gsm8k_with_comma(self):
        """Test GSM8K with comma-formatted numbers."""
        result = self.verifier.verify("1,234", "1234")
        assert result.status == VerificationStatus.CORRECT


class TestReasoningPathExtraction:
    """Tests for reasoning path answer extraction."""
    
    def setup_method(self):
        self.verifier = MathVerifier()
    
    def test_the_answer_is_pattern(self):
        """Test 'the answer is X' pattern."""
        text = "Therefore, the answer is 42."
        answer = self.verifier.extract_final_answer(text)
        assert answer == "42"
    
    def test_therefore_pattern(self):
        """Test 'therefore X' pattern."""
        text = "We calculated everything. Therefore, 100."
        answer = self.verifier.extract_final_answer(text)
        assert answer is not None
    
    def test_equals_at_end(self):
        """Test 'X = Y' pattern at end."""
        text = "Adding them all up: total = 55"
        answer = self.verifier.extract_final_answer(text)
        assert answer == "55"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
