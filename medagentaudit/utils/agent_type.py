from enum import Enum
class AgentType(Enum):
    """Agent type enumeration."""
    DOMAIN = "DomainAgent"
    SYNTHESIZER = "Synthesizer"
    DECISION_MAKER = "DecisionMaker"
    META = "Synthesizer and DecisionMaker"
    AUDITOR = "Auditor"