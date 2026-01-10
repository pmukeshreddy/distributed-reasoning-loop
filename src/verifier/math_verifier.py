"""
Mathematical Expression Verifier using SymPy.
Validates reasoning paths by comparing final answers.
"""

import re
import sympy
from sympy import simplify, sympify, Eq, solve, N
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from typing import Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARSE_ERROR = "parse_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class VerificationResult:
    status: VerificationStatus
    expected: Optional[str] = None
    predicted: Optional[str] = None
    confidence: float = 1.0
    error_message: Optional[str] = None
    intermediate_steps_valid: bool = True


class MathVerifier:
    """
    Verifies mathematical expressions and reasoning paths.
    Supports:
    - Numeric answer comparison
    - Symbolic expression equivalence
    - Unit handling
    - Fraction/decimal equivalence
    """
    
    def __init__(self, tolerance: float = 1e-6, timeout: int = 10):
        self.tolerance = tolerance
        self.timeout = timeout
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        
    def extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extract the final answer from a reasoning path.
        Looks for patterns like:
        - "The answer is X"
        - "#### X"
        - "= X" at the end
        - Boxed answers: \\boxed{X}
        """
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'####\s*(.+?)(?:\n|$)',
            r'[Tt]he (?:final )?answer is[:\s]*([^\n.]+)',
            r'[Tt]herefore[,:]?\s*(?:the answer is\s*)?([^\n.]+)',
            r'[Ss]o[,:]?\s*(?:the answer is\s*)?([^\n.]+?)(?:\.|$)',
            r'=\s*([^\n=]+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                answer = matches[-1].strip()
                # Clean up common artifacts
                answer = re.sub(r'^\$|\$$', '', answer)
                answer = re.sub(r'^\\text\{|\}$', '', answer)
                return answer
        
        return None
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize an answer string for comparison."""
        if answer is None:
            return ""
        
        answer = str(answer).strip()
        
        # Remove currency symbols and units
        answer = re.sub(r'[\$£€]', '', answer)
        answer = re.sub(r'\s*(dollars?|cents?|percent|%|miles?|hours?|minutes?|seconds?|kg|km|m|cm|mm)\s*$', '', answer, flags=re.IGNORECASE)
        
        # Handle fractions written as "X/Y"
        answer = answer.replace('\\frac', '')
        
        # Remove commas in numbers
        answer = re.sub(r'(\d),(\d)', r'\1\2', answer)
        
        # Remove LaTeX formatting
        answer = re.sub(r'\\[a-zA-Z]+', '', answer)
        answer = answer.replace('{', '').replace('}', '')
        
        return answer.strip()
    
    def parse_numeric(self, value: str) -> Optional[float]:
        """Try to parse a string as a numeric value."""
        try:
            normalized = self.normalize_answer(value)
            
            # Handle fractions
            if '/' in normalized:
                parts = normalized.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            
            # Handle mixed numbers like "3 1/2"
            mixed_match = re.match(r'(-?\d+)\s+(\d+)/(\d+)', normalized)
            if mixed_match:
                whole, num, denom = mixed_match.groups()
                return float(whole) + float(num) / float(denom)
            
            # Direct numeric parse
            return float(normalized)
        except (ValueError, ZeroDivisionError):
            return None
    
    def parse_symbolic(self, expr: str) -> Optional[Any]:
        """Parse a string as a symbolic expression."""
        try:
            normalized = self.normalize_answer(expr)
            return parse_expr(normalized, transformations=self.transformations)
        except Exception:
            return None
    
    def compare_numeric(self, pred: str, expected: str) -> Tuple[bool, float]:
        """Compare two values numerically."""
        pred_num = self.parse_numeric(pred)
        exp_num = self.parse_numeric(expected)
        
        if pred_num is None or exp_num is None:
            return False, 0.0
        
        if exp_num == 0:
            is_equal = abs(pred_num) < self.tolerance
        else:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
            is_equal = relative_error < self.tolerance
        
        confidence = 1.0 if is_equal else max(0, 1 - abs(pred_num - exp_num) / max(abs(exp_num), 1))
        return is_equal, confidence
    
    def compare_symbolic(self, pred: str, expected: str) -> Tuple[bool, float]:
        """Compare two expressions symbolically."""
        pred_sym = self.parse_symbolic(pred)
        exp_sym = self.parse_symbolic(expected)
        
        if pred_sym is None or exp_sym is None:
            return False, 0.0
        
        try:
            # Try symbolic simplification
            diff = simplify(pred_sym - exp_sym)
            is_equal = diff == 0
            
            # If not symbolically equal, try numeric evaluation
            if not is_equal:
                pred_val = complex(N(pred_sym))
                exp_val = complex(N(exp_sym))
                if abs(exp_val) > 0:
                    is_equal = abs(pred_val - exp_val) / abs(exp_val) < self.tolerance
                else:
                    is_equal = abs(pred_val) < self.tolerance
            
            return is_equal, 1.0 if is_equal else 0.0
        except Exception:
            return False, 0.0
    
    def verify(self, predicted: str, expected: str) -> VerificationResult:
        """
        Verify if predicted answer matches expected answer.
        Tries multiple comparison strategies.
        """
        # Extract final answers if full reasoning paths provided
        pred_answer = self.extract_final_answer(predicted) or predicted
        exp_answer = self.extract_final_answer(expected) or expected
        
        # Normalize both answers
        pred_norm = self.normalize_answer(pred_answer)
        exp_norm = self.normalize_answer(exp_answer)
        
        # Exact string match
        if pred_norm.lower() == exp_norm.lower():
            return VerificationResult(
                status=VerificationStatus.CORRECT,
                expected=exp_norm,
                predicted=pred_norm,
                confidence=1.0
            )
        
        # Numeric comparison
        is_equal, confidence = self.compare_numeric(pred_norm, exp_norm)
        if is_equal:
            return VerificationResult(
                status=VerificationStatus.CORRECT,
                expected=exp_norm,
                predicted=pred_norm,
                confidence=confidence
            )
        
        # Symbolic comparison
        is_equal, confidence = self.compare_symbolic(pred_norm, exp_norm)
        if is_equal:
            return VerificationResult(
                status=VerificationStatus.CORRECT,
                expected=exp_norm,
                predicted=pred_norm,
                confidence=confidence
            )
        
        return VerificationResult(
            status=VerificationStatus.INCORRECT,
            expected=exp_norm,
            predicted=pred_norm,
            confidence=0.0
        )
    
    def verify_reasoning_path(self, reasoning: str, expected_answer: str) -> VerificationResult:
        """
        Verify a complete reasoning path.
        Extracts the final answer and compares it.
        """
        final_answer = self.extract_final_answer(reasoning)
        
        if final_answer is None:
            return VerificationResult(
                status=VerificationStatus.PARSE_ERROR,
                expected=expected_answer,
                predicted=None,
                error_message="Could not extract final answer from reasoning"
            )
        
        return self.verify(final_answer, expected_answer)
    
    def verify_intermediate_steps(self, steps: list[str]) -> list[VerificationResult]:
        """
        Verify intermediate calculation steps.
        Each step should be in format "expression = result"
        """
        results = []
        
        for step in steps:
            if '=' not in step:
                results.append(VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    error_message="Step does not contain equation"
                ))
                continue
            
            parts = step.split('=')
            if len(parts) != 2:
                results.append(VerificationResult(
                    status=VerificationStatus.PARSE_ERROR,
                    error_message="Invalid equation format"
                ))
                continue
            
            lhs, rhs = parts[0].strip(), parts[1].strip()
            
            try:
                lhs_val = parse_expr(lhs, transformations=self.transformations)
                rhs_val = parse_expr(rhs, transformations=self.transformations)
                
                if simplify(lhs_val - rhs_val) == 0:
                    results.append(VerificationResult(
                        status=VerificationStatus.CORRECT,
                        expected=rhs,
                        predicted=str(simplify(lhs_val))
                    ))
                else:
                    results.append(VerificationResult(
                        status=VerificationStatus.INCORRECT,
                        expected=rhs,
                        predicted=str(simplify(lhs_val))
                    ))
            except Exception as e:
                results.append(VerificationResult(
                    status=VerificationStatus.PARSE_ERROR,
                    error_message=str(e)
                ))
        
        return results


# GSM8K specific verifier
class GSM8KVerifier(MathVerifier):
    """
    Specialized verifier for GSM8K dataset.
    GSM8K answers are always numeric and use #### delimiter.
    """
    
    def extract_final_answer(self, text: str) -> Optional[str]:
        """Extract answer using GSM8K format."""
        # GSM8K uses #### to mark the final answer
        match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
        
        # Fallback to parent implementation
        return super().extract_final_answer(text)
    
    def verify(self, predicted: str, expected: str) -> VerificationResult:
        """Verify GSM8K style numeric answers."""
        pred_answer = self.extract_final_answer(predicted) or predicted
        exp_answer = self.extract_final_answer(expected) or expected
        
        try:
            pred_num = float(pred_answer.replace(',', ''))
            exp_num = float(exp_answer.replace(',', ''))
            
            # GSM8K requires exact numeric match (integers)
            if abs(pred_num - exp_num) < 0.001:
                return VerificationResult(
                    status=VerificationStatus.CORRECT,
                    expected=str(exp_num),
                    predicted=str(pred_num),
                    confidence=1.0
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.INCORRECT,
                    expected=str(exp_num),
                    predicted=str(pred_num),
                    confidence=0.0
                )
        except ValueError as e:
            return VerificationResult(
                status=VerificationStatus.PARSE_ERROR,
                expected=exp_answer,
                predicted=pred_answer,
                error_message=str(e)
            )
