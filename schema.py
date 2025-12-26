from pydantic import BaseModel, Field
from typing import List

class ResearchPlan(BaseModel):
    """The master plan for the deep research."""
    queries: List[str] = Field(description="Search queries for web/local data.")
    reasoning: str = Field(description="Why these specific queries matter.")

class ResearchState(BaseModel):
    """The state shared between nodes of the graph."""
    task: str
    queries: List[str] = []
    scraped_data: List[str] = []
    final_report: str = ""