from enum import Enum, unique

@unique
class MedicalSpecialty(Enum):
    """Medical specialty enumeration."""
    INTERNAL_MEDICINE = "Internal Medicine"
    SURGERY = "Surgery"
    RADIOLOGY = "Radiology"
    GENERAL_MEDICINE = "General Medicine"
    SAFETY_SUPERVISOR = "Safety Supervisor"
    FACTUAL_ACCURACY = "Factual Accuracy"
    PEDIATRICS = "Pediatrics"
    CARDIOLOGY = "Cardiology"
    PULMONOLOGY = "Pulmonology"
    NEONATOLOGY = "Neonatology"
    GENETICS = "Genetics"
    RECONCILE = "Reconcile"

