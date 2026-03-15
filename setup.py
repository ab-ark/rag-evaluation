from setuptools import setup, find_packages

setup(
    name="abark-rag-eval",
    version="0.1.0",
    author="AbArk",
    description="Production-grade RAG evaluation framework",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.27.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "pytest-asyncio", "httpx"]
    },
)
