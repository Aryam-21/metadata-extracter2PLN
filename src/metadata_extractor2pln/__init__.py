"""Schema-adaptive metadata extraction for PeTTaChainer."""

from .compiler import CONTRACT_VERSION, compile_fact
from .models import ExtractionPlan, PropertySpec

__all__ = ["CONTRACT_VERSION", "ExtractionPlan", "PropertySpec", "compile_fact"]
"""Schema-adaptive metadata extraction and PeTTa fact compilation."""

__version__ = "0.1.0"
