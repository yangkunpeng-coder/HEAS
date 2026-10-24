"""Explicit failures at fixed control-plane boundaries."""


class HEASError(ValueError):
    """Identify a handled HEAS contract or execution failure."""

    pass


class MissingEvidence(HEASError):
    """Indicate an evidence reference that cannot be resolved from the store."""

    pass


class IntegrityError(HEASError):
    """Indicate conflicting immutable content or a failed digest check."""

    pass


class BudgetExceeded(HEASError):
    """Stop execution when a configured resource boundary cannot be satisfied."""

    pass


class SecurityError(HEASError):
    """Reject an unauthorized capability, private-data flow, or filesystem access."""

    pass


class ModelProtocolError(HEASError):
    """Reject malformed or contract-incompatible model responses."""

    def __init__(self, message: str, data: object = None):
        super().__init__(message)
        self.data = data


class RecoveryRequired(HEASError):
    """Require host inspection rather than replaying uncertain side effects."""

    pass
