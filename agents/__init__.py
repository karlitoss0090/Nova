from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """
    Base class for all NOVA agents.

    To add a new agent:
    1. Create agents/my_agent.py
    2. Subclass BaseAgent
    3. Implement can_handle() and handle()
    4. Register in main.py's AGENTS list

    Example: agents/study_agent.py
    """

    name: str = "base"
    description: str = "Base agent"
    icon: str = "🤖"

    def __init__(self, llm_client: Any, memory_manager: Any):
        self.llm = llm_client
        self.memory = memory_manager

    @abstractmethod
    async def can_handle(self, message: str, context: Dict) -> bool:
        """Return True if this agent should handle the given message."""

    @abstractmethod
    async def handle(self, message: str, context: Dict) -> str:
        """Process the message and return a response string."""

    def get_system_prompt(self) -> Optional[str]:
        """Override to inject a custom system prompt for this agent."""
        return None

    def __repr__(self) -> str:
        return f"<Agent:{self.name}>"
