"""
Property-based tests for project structure validation.

**Feature: traffic-pollution-dashboard, Property 1: Project structure consistency**
"""

import os
from pathlib import Path
from hypothesis import given, strategies as st
import pytest


class TestProjectStructure:
    """Test project structure consistency and organization."""
    
    def test_required_directories_exist(self):
        """Test that all required project directories exist."""
        # **Feature: traffic-pollution-dashboard, Property 1: Project structure consistency**
        # **Validates: Requirements 6.4**
        
        required_dirs = [
            "traffic_pollution_dashboard",
            "traffic_pollution_dashboard/data",
            "traffic_pollution_dashboard/services", 
            "traffic_pollution_dashboard/visualization",
            "traffic_pollution_dashboard/config",
            "tests",
            ".kiro/specs/traffic-pollution-dashboard"
        ]
        
        project_root = Path.cwd()
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Required directory {dir_path} does not exist"
            assert full_path.is_dir(), f"Path {dir_path} exists but is not a directory"
    
    def test_required_files_exist(self):
        """Test that all required project files exist."""
        # **Feature: traffic-pollution-dashboard, Property 1: Project structure consistency**
        # **Validates: Requirements 6.4**
        
        required_files = [
            "requirements.txt",
            "README.md",
            "app.py",
            "pytest.ini",
            ".gitignore",
            ".env.example",
            "traffic_pollution_dashboard/__init__.py",
            "traffic_pollution_dashboard/config/settings.py",
            "tests/__init__.py"
        ]
        
        project_root = Path.cwd()
        
        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Required file {file_path} does not exist"
            assert full_path.is_file(), f"Path {file_path} exists but is not a file"
    
    def test_python_packages_have_init_files(self):
        """Test that all Python packages have __init__.py files."""
        # **Feature: traffic-pollution-dashboard, Property 1: Project structure consistency**
        # **Validates: Requirements 6.4**
        
        python_packages = [
            "traffic_pollution_dashboard",
            "traffic_pollution_dashboard/data",
            "traffic_pollution_dashboard/services",
            "traffic_pollution_dashboard/visualization", 
            "traffic_pollution_dashboard/config",
            "tests"
        ]
        
        project_root = Path.cwd()
        
        for package_path in python_packages:
            init_file = project_root / package_path / "__init__.py"
            assert init_file.exists(), f"Package {package_path} missing __init__.py file"
            assert init_file.is_file(), f"__init__.py in {package_path} is not a file"
    
    @given(st.sampled_from([
        "traffic_pollution_dashboard",
        "traffic_pollution_dashboard/data", 
        "traffic_pollution_dashboard/services",
        "traffic_pollution_dashboard/visualization",
        "traffic_pollution_dashboard/config"
    ]))
    def test_package_structure_consistency(self, package_path: str):
        """Property test: For any package directory, it should have proper Python package structure."""
        # **Feature: traffic-pollution-dashboard, Property 1: Project structure consistency**
        # **Validates: Requirements 6.4**
        
        project_root = Path.cwd()
        package_dir = project_root / package_path
        
        # Package directory should exist
        assert package_dir.exists(), f"Package directory {package_path} does not exist"
        assert package_dir.is_dir(), f"Package path {package_path} is not a directory"
        
        # Package should have __init__.py
        init_file = package_dir / "__init__.py"
        assert init_file.exists(), f"Package {package_path} missing __init__.py"
        
        # __init__.py should be readable
        assert init_file.is_file(), f"__init__.py in {package_path} is not a file"
        
        # Should be able to read the init file content
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Content should be valid (not empty or just whitespace for main packages)
                if package_path == "traffic_pollution_dashboard":
                    assert len(content.strip()) > 0, "Main package __init__.py should not be empty"
        except Exception as e:
            pytest.fail(f"Could not read __init__.py in {package_path}: {e}")
    
    def test_configuration_files_are_valid(self):
        """Test that configuration files have valid structure."""
        # **Feature: traffic-pollution-dashboard, Property 1: Project structure consistency**
        # **Validates: Requirements 6.4**
        
        project_root = Path.cwd()
        
        # Test requirements.txt
        requirements_file = project_root / "requirements.txt"
        with open(requirements_file, 'r') as f:
            requirements_content = f.read()
            # Should contain essential dependencies
            essential_deps = ["streamlit", "pandas", "plotly", "requests", "pytest", "hypothesis"]
            for dep in essential_deps:
                assert dep in requirements_content, f"Missing essential dependency: {dep}"
        
        # Test .env.example
        env_example = project_root / ".env.example"
        with open(env_example, 'r') as f:
            env_content = f.read()
            # Should contain essential environment variables
            essential_vars = ["TRAFFIC_API_KEY", "POLLUTION_API_KEY", "SUPPORTED_CITIES"]
            for var in essential_vars:
                assert var in env_content, f"Missing essential environment variable: {var}"
        
        # Test pytest.ini
        pytest_ini = project_root / "pytest.ini"
        with open(pytest_ini, 'r') as f:
            pytest_content = f.read()
            # Should contain test configuration
            assert "testpaths" in pytest_content, "pytest.ini missing testpaths configuration"
            assert "traffic_pollution_dashboard" in pytest_content, "pytest.ini missing coverage configuration"