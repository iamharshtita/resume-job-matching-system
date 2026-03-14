"""
Base Agent class for all agents in the system
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from loguru import logger


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the resume-job matching system.
    """

    def __init__(self, agent_name: str, config: Dict[str, Any] = None):
        """
        Initialize the base agent.

        Args:
            agent_name: Name of the agent
            config: Configuration dictionary
        """
        self.agent_name = agent_name
        self.config = config or {}
        logger.info(f"Initialized {self.agent_name}")

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.

        Args:
            input_data: Input data dictionary

        Returns:
            Processing results dictionary
        """
        pass

    def validate_input(self, input_data: Dict[str, Any], required_fields: list) -> bool:
        """
        Validate that input data contains required fields.

        Args:
            input_data: Input data to validate
            required_fields: List of required field names

        Returns:
            True if valid, False otherwise
        """
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            logger.error(f"{self.agent_name}: Missing required fields: {missing_fields}")
            return False
        return True

    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log agent performance metrics.

        Args:
            metrics: Dictionary of metric names and values
        """
        logger.info(f"{self.agent_name} Metrics: {metrics}")
