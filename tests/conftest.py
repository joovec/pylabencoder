"""Pytest configuration and fixtures."""
import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "name": "test_user",
        "id": 1,
        "email": "test@example.com"
    }