"""Tests for core.base module."""
import pytest
from pytemplate.core.base import BaseModel, add


class TestBaseModel:
    """Test BaseModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model = BaseModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        model = BaseModel(name="test", value=42)
        result = model.to_dict()
        assert result == {"name": "test", "value": 42}
    
    def test_repr(self):
        """Test string representation."""
        model = BaseModel(name="test")
        assert repr(model) == "BaseModel(name='test')"


def test_add():
    """Test add function."""
    # 기본적인 덧셈
    assert add(2, 3) == 5
    # 음수 테스트
    assert add(-1, 1) == 0
    # 0과의 덧셈
    assert add(5, 0) == 5
    # 소수 테스트
    assert add(1.5, 2.5) == 4.0