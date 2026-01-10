from setuptools import setup, find_packages

setup(
    name="distributed-reasoning-loop",
    version="0.1.0",
    description="Distributed infrastructure for training reasoning models with synthetic data generation and RL",
    author="Mukesh Reddy",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core ML/Inference
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        
        # RL Training
        "trl>=0.7.0",
        "peft>=0.7.0",
        
        # Data Processing
        "datasets>=2.16.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        
        # Verification
        "sympy>=1.12",
        "docker>=7.0.0",
        
        # Utils
        "tqdm>=4.65.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "inference": [
            "vllm>=0.4.0",
            "sglang>=0.1.0",
        ],
        "distributed": [
            "ray[all]>=2.9.0",
            "kafka-python>=2.0.2",
            "confluent-kafka>=2.3.0",
        ],
        "training": [
            "deepspeed>=0.12.0",
            "wandb>=0.16.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "all": [
            "vllm>=0.4.0",
            "sglang>=0.1.0",
            "ray[all]>=2.9.0",
            "kafka-python>=2.0.2",
            "confluent-kafka>=2.3.0",
            "deepspeed>=0.12.0",
            "wandb>=0.16.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "reasoning-loop=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
