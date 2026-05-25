from pydantic import BaseModel, Field
from typing import List, Any
from pydantic import field_validator

class DiagnosticResult(BaseModel):
    """Strictly enforced schema for the LLM's diagnostic generation."""
    summary: str = Field(default="N/A", description="Concise summary of the issue.")
    diagnostics: str = Field(default="Could not parse structured response. See summary.", description="Step-by-step repair instructions.")
    references: List[str] = Field(default_factory=list, description="Page citations from the manual.")
    warnings: List[str] = Field(default_factory=list, description="Safety warnings.")

    @field_validator('references', 'warnings', mode='before')
    @classmethod
    def ensure_list(cls, v: Any) -> List[str]:
        """Coerce unexpected LLM outputs (like 'N/A' strings) into empty lists."""
        if isinstance(v, list):
            return [str(item) for item in v]
        if isinstance(v, str):
            if v.strip().upper() in ["N/A", "NONE", "", "NULL"]:
                return []
            return [v]
        return []

class AnalyzeResponse(BaseModel):
    """Final API response payload for the /analyze endpoint."""
    summary: str
    diagnostics: str
    references: List[str]
    warnings: List[str]
    query_used: str
    chunks_used: int
