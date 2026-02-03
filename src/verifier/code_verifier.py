"""
Code Execution Verifier using Docker sandboxes.
Safely executes generated code and validates output.
"""

import tempfile
import os
import json
import hashlib
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import re

# Docker is optional - only needed for code execution
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    docker = None
    DOCKER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    SUCCESS = "success"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    COMPILATION_ERROR = "compilation_error"
    WRONG_ANSWER = "wrong_answer"
    SANDBOX_ERROR = "sandbox_error"


@dataclass
class TestCase:
    input: str
    expected_output: str
    is_hidden: bool = False


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time: float = 0.0
    memory_used: int = 0
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


class DockerSandbox:
    """
    Secure Docker-based sandbox for code execution.
    Provides isolation, resource limits, and timeout handling.
    """
    
    DEFAULT_MEMORY_LIMIT = "256m"
    DEFAULT_CPU_LIMIT = 1.0
    DEFAULT_TIMEOUT = 30
    
    LANGUAGE_CONFIGS = {
        "python": {
            "image": "python:3.11-slim",
            "file_ext": ".py",
            "compile_cmd": None,
            "run_cmd": "python {file}",
        },
        "javascript": {
            "image": "node:20-slim",
            "file_ext": ".js",
            "compile_cmd": None,
            "run_cmd": "node {file}",
        },
        "typescript": {
            "image": "node:20-slim",
            "file_ext": ".ts",
            "compile_cmd": "npx tsc {file} --outDir /tmp",
            "run_cmd": "node /tmp/{basename}.js",
        },
        "cpp": {
            "image": "gcc:13",
            "file_ext": ".cpp",
            "compile_cmd": "g++ -o /tmp/a.out {file} -std=c++17",
            "run_cmd": "/tmp/a.out",
        },
        "java": {
            "image": "openjdk:17-slim",
            "file_ext": ".java",
            "compile_cmd": "javac -d /tmp {file}",
            "run_cmd": "java -cp /tmp Main",
        },
        "rust": {
            "image": "rust:1.75-slim",
            "file_ext": ".rs",
            "compile_cmd": "rustc -o /tmp/a.out {file}",
            "run_cmd": "/tmp/a.out",
        },
    }
    
    def __init__(
        self,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        cpu_limit: float = DEFAULT_CPU_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
        network_disabled: bool = True,
    ):
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        self.network_disabled = network_disabled
        self.client = None
        
    def _get_client(self):
        """Lazy initialization of Docker client."""
        if not DOCKER_AVAILABLE:
            raise ImportError(
                "Docker SDK not installed. Install with: pip install docker\n"
                "For math-only tasks, Docker is not required."
            )
        if self.client is None:
            self.client = docker.from_env()
        return self.client
    
    def _create_container(
        self,
        image: str,
        code_path: str,
        workdir: str = "/code"
    ):
        """Create a sandboxed container with security restrictions."""
        client = self._get_client()
        
        return client.containers.create(
            image=image,
            command="sleep infinity",
            volumes={code_path: {"bind": workdir, "mode": "rw"}},
            working_dir=workdir,
            mem_limit=self.memory_limit,
            nano_cpus=int(self.cpu_limit * 1e9),
            network_disabled=self.network_disabled,
            read_only=False,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            detach=True,
        )
    
    def execute(
        self,
        code: str,
        language: str,
        stdin: str = "",
        test_cases: Optional[List[TestCase]] = None,
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed Docker container.
        
        Args:
            code: The source code to execute
            language: Programming language (python, javascript, etc.)
            stdin: Standard input to provide
            test_cases: Optional list of test cases to run
            
        Returns:
            ExecutionResult with status, output, and test results
        """
        if language not in self.LANGUAGE_CONFIGS:
            return ExecutionResult(
                status=ExecutionStatus.SANDBOX_ERROR,
                error_message=f"Unsupported language: {language}"
            )
        
        config = self.LANGUAGE_CONFIGS[language]
        
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            filename = f"main{config['file_ext']}"
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write(code)
            
            container = None
            try:
                # Create container
                container = self._create_container(config["image"], tmpdir)
                container.start()
                
                start_time = time.time()
                
                # Compile if needed
                if config["compile_cmd"]:
                    compile_cmd = config["compile_cmd"].format(
                        file=f"/code/{filename}",
                        basename=os.path.splitext(filename)[0]
                    )
                    exit_code, output = container.exec_run(
                        compile_cmd,
                        demux=True
                    )
                    if exit_code != 0:
                        stdout, stderr = output if output else ("", "")
                        return ExecutionResult(
                            status=ExecutionStatus.COMPILATION_ERROR,
                            stdout=stdout.decode() if stdout else "",
                            stderr=stderr.decode() if stderr else "",
                            exit_code=exit_code,
                            error_message="Compilation failed"
                        )
                
                # Run code
                run_cmd = config["run_cmd"].format(
                    file=f"/code/{filename}",
                    basename=os.path.splitext(filename)[0]
                )
                
                if test_cases:
                    return self._run_test_cases(
                        container, run_cmd, test_cases, start_time
                    )
                else:
                    return self._run_single(
                        container, run_cmd, stdin, start_time
                    )
                    
            except docker.errors.ImageNotFound:
                return ExecutionResult(
                    status=ExecutionStatus.SANDBOX_ERROR,
                    error_message=f"Docker image not found: {config['image']}"
                )
            except docker.errors.APIError as e:
                return ExecutionResult(
                    status=ExecutionStatus.SANDBOX_ERROR,
                    error_message=f"Docker API error: {str(e)}"
                )
            finally:
                if container:
                    try:
                        container.stop(timeout=1)
                        container.remove(force=True)
                    except Exception:
                        pass
    
    def _run_single(
        self,
        container,
        cmd: str,
        stdin: str,
        start_time: float
    ) -> ExecutionResult:
        """Run a single execution."""
        try:
            # Create stdin file if provided
            if stdin:
                container.exec_run(f"sh -c 'echo \"{stdin}\" > /tmp/stdin.txt'")
                cmd = f"sh -c '{cmd} < /tmp/stdin.txt'"
            
            # Execute with timeout
            exec_result = container.exec_run(
                cmd,
                demux=True,
            )
            
            execution_time = time.time() - start_time
            
            if execution_time > self.timeout:
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    execution_time=execution_time,
                    error_message=f"Execution exceeded {self.timeout}s timeout"
                )
            
            stdout, stderr = exec_result.output if exec_result.output else (b"", b"")
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS if exec_result.exit_code == 0 else ExecutionStatus.RUNTIME_ERROR,
                stdout=stdout.decode() if stdout else "",
                stderr=stderr.decode() if stderr else "",
                exit_code=exec_result.exit_code,
                execution_time=execution_time,
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.SANDBOX_ERROR,
                error_message=str(e)
            )
    
    def _run_test_cases(
        self,
        container,
        cmd: str,
        test_cases: List[TestCase],
        start_time: float
    ) -> ExecutionResult:
        """Run multiple test cases."""
        test_results = []
        all_passed = True
        
        for i, test in enumerate(test_cases):
            # Write input to file
            if test.input:
                container.exec_run(
                    f"sh -c 'cat > /tmp/input_{i}.txt << EOF\n{test.input}\nEOF'"
                )
                test_cmd = f"sh -c '{cmd} < /tmp/input_{i}.txt'"
            else:
                test_cmd = cmd
            
            exec_result = container.exec_run(test_cmd, demux=True)
            
            stdout, stderr = exec_result.output if exec_result.output else (b"", b"")
            actual_output = stdout.decode().strip() if stdout else ""
            expected_output = test.expected_output.strip()
            
            passed = actual_output == expected_output
            if not passed:
                all_passed = False
            
            test_results.append({
                "test_id": i,
                "passed": passed,
                "expected": expected_output if not test.is_hidden else "[hidden]",
                "actual": actual_output if not test.is_hidden else "[hidden]",
                "stderr": stderr.decode() if stderr else "",
            })
        
        execution_time = time.time() - start_time
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if all_passed else ExecutionStatus.WRONG_ANSWER,
            execution_time=execution_time,
            test_results=test_results,
        )


class CodeVerifier:
    """
    High-level code verification interface.
    Manages sandbox execution and result validation.
    """
    
    def __init__(self, timeout: int = 30, max_workers: int = 4):
        self.sandbox = DockerSandbox(timeout=timeout)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def extract_code(self, text: str, language: str = "python") -> Optional[str]:
        """Extract code block from text."""
        # Try to find code blocks with language tags
        patterns = [
            rf'```{language}\n(.*?)```',
            rf'```\n(.*?)```',
            r'```(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code blocks, return the whole text (might be just code)
        return text.strip()
    
    def verify_humaneval(
        self,
        code: str,
        entry_point: str,
        test_code: str,
    ) -> ExecutionResult:
        """
        Verify code against HumanEval style test cases.
        
        Args:
            code: The generated solution code
            entry_point: Function name to test
            test_code: The test code to run
        """
        # Combine solution with tests
        full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"
        
        return self.sandbox.execute(full_code, "python")
    
    def verify_with_tests(
        self,
        code: str,
        language: str,
        test_cases: List[TestCase],
    ) -> ExecutionResult:
        """Verify code with input/output test cases."""
        return self.sandbox.execute(code, language, test_cases=test_cases)
    
    def verify_function_output(
        self,
        code: str,
        language: str,
        function_call: str,
        expected_output: str,
    ) -> ExecutionResult:
        """Verify that a function call produces expected output."""
        if language == "python":
            test_code = f"""
{code}

result = {function_call}
print(result)
"""
        elif language == "javascript":
            test_code = f"""
{code}

console.log({function_call});
"""
        else:
            return ExecutionResult(
                status=ExecutionStatus.SANDBOX_ERROR,
                error_message=f"Function verification not implemented for {language}"
            )
        
        result = self.sandbox.execute(test_code, language)
        
        if result.status == ExecutionStatus.SUCCESS:
            actual = result.stdout.strip()
            expected = expected_output.strip()
            
            if actual == expected:
                result.status = ExecutionStatus.SUCCESS
            else:
                result.status = ExecutionStatus.WRONG_ANSWER
                result.test_results = [{
                    "expected": expected,
                    "actual": actual,
                    "passed": False
                }]
        
        return result
    
    def batch_verify(
        self,
        submissions: List[Dict[str, Any]],
    ) -> List[ExecutionResult]:
        """
        Verify multiple code submissions in parallel.
        
        Args:
            submissions: List of dicts with 'code', 'language', 'test_cases'
        """
        futures = []
        for sub in submissions:
            future = self.executor.submit(
                self.sandbox.execute,
                sub["code"],
                sub["language"],
                test_cases=sub.get("test_cases", [])
            )
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.sandbox.timeout * 2)
                results.append(result)
            except FuturesTimeoutError:
                results.append(ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error_message="Batch execution timeout"
                ))
            except Exception as e:
                results.append(ExecutionResult(
                    status=ExecutionStatus.SANDBOX_ERROR,
                    error_message=str(e)
                ))
        
        return results
    
    def __del__(self):
        self.executor.shutdown(wait=False)


# HumanEval specific verifier
class HumanEvalVerifier(CodeVerifier):
    """
    Specialized verifier for HumanEval dataset.
    """
    
    def verify_problem(
        self,
        task_id: str,
        prompt: str,
        completion: str,
        test: str,
        entry_point: str,
    ) -> ExecutionResult:
        """
        Verify a HumanEval problem.
        
        Args:
            task_id: HumanEval task ID (e.g., "HumanEval/0")
            prompt: The function signature and docstring
            completion: The generated completion
            test: The test code
            entry_point: Function name
        """
        # Combine prompt and completion
        full_code = prompt + completion
        
        # Clean up the code
        full_code = self._clean_code(full_code)
        
        # Run verification
        result = self.verify_humaneval(full_code, entry_point, test)
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Clean up generated code."""
        # Remove markdown code blocks if present
        code = re.sub(r'^```python\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
        
        # Remove any extra function definitions after the main one
        lines = code.split('\n')
        cleaned_lines = []
        in_function = False
        function_indent = 0
        
        for line in lines:
            if line.strip().startswith('def ') and not in_function:
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                cleaned_lines.append(line)
            elif in_function:
                current_indent = len(line) - len(line.lstrip()) if line.strip() else function_indent + 1
                if line.strip() and current_indent <= function_indent:
                    if line.strip().startswith('def '):
                        break
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
