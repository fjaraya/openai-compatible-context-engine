class ContextEngineError(Exception):
    """Base exception for the package."""


class BudgetConfigurationError(ContextEngineError):
    """Raised when a token budget is invalid."""


class PinnedItemOverflowError(ContextEngineError):
    """Raised when mandatory items cannot fit into the configured budget."""
