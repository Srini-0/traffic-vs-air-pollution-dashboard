#!/usr/bin/env python3
"""
Deployment script for Traffic Pollution Dashboard.

This script helps deploy the dashboard to various platforms.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    
    return result


def check_requirements():
    """Check if all requirements are installed."""
    print("Checking requirements...")
    
    try:
        import streamlit
        import pandas
        import plotly
        import requests
        import numpy
        import scipy
        print("✓ All required packages are installed")
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    result = run_command("python3 -m pytest tests/ -v", check=False)
    
    if result.returncode != 0:
        print("✗ Tests failed")
        return False
    else:
        print("✓ All tests passed")
        return True


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from .env.example...")
        env_file.write_text(env_example.read_text())
        print("✓ Created .env file")
        print("Please edit .env file with your API keys")
    elif env_file.exists():
        print("✓ .env file already exists")
    else:
        print("✗ .env.example file not found")


def deploy_local():
    """Deploy locally using Streamlit."""
    print("Deploying locally...")
    print("Starting Streamlit server...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    run_command("streamlit run app.py", check=False)


def deploy_docker():
    """Deploy using Docker."""
    print("Deploying with Docker...")
    
    # Create Dockerfile if it doesn't exist
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    dockerfile = Path("Dockerfile")
    if not dockerfile.exists():
        dockerfile.write_text(dockerfile_content)
        print("✓ Created Dockerfile")
    
    # Build Docker image
    run_command("docker build -t traffic-pollution-dashboard .")
    
    # Run Docker container
    print("Starting Docker container...")
    print("Dashboard will be available at: http://localhost:8501")
    run_command("docker run -p 8501:8501 traffic-pollution-dashboard")


def deploy_cloud():
    """Deploy to cloud platforms."""
    print("Cloud deployment options:")
    print("1. Streamlit Cloud: Push to GitHub and connect at https://share.streamlit.io/")
    print("2. Heroku: Use the provided Procfile")
    print("3. AWS/GCP/Azure: Use Docker deployment")
    
    # Create Procfile for Heroku
    procfile = Path("Procfile")
    if not procfile.exists():
        procfile.write_text("web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0")
        print("✓ Created Procfile for Heroku deployment")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Traffic Pollution Dashboard")
    parser.add_argument(
        "target",
        choices=["local", "docker", "cloud", "test"],
        help="Deployment target"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests before deployment"
    )
    
    args = parser.parse_args()
    
    print("🚦 Traffic Pollution Dashboard Deployment")
    print("=" * 50)
    
    # Check requirements
    check_requirements()
    
    # Create .env file
    create_env_file()
    
    # Run tests unless skipped
    if not args.skip_tests and args.target != "test":
        if not run_tests():
            print("Deployment aborted due to test failures")
            sys.exit(1)
    
    # Deploy based on target
    if args.target == "local":
        deploy_local()
    elif args.target == "docker":
        deploy_docker()
    elif args.target == "cloud":
        deploy_cloud()
    elif args.target == "test":
        run_tests()
    
    print("🎉 Deployment completed!")


if __name__ == "__main__":
    main()