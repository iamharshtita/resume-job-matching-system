from abc import ABC, abstractmethod
from typing import Dict, Any
from loguru import logger

class BaseAgent(ABC):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def validate_input(self, input_data: Dict[str, Any], required: list) -> bool:
        missing = [f for f in required if f not in input_data]
        if missing:
            logger.error(f"{self.name}: missing fields {missing}")
            return False
        return True

    def log_metrics(self, metrics: Dict[str, Any]):
        logger.info(f"{self.name}: {metrics}")
