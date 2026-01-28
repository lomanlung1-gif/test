

from typing import Any, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field

# ==========================================
# Data Models
# ==========================================

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Location(StrictBaseModel):
    """
    Represents a physical or geographical location associated with a specific time period.
    """
    location: str = Field( ..., description="The address, city, country, or region name.")
    start_date: Optional[str] = Field(default=None, alias="from", description="The date when the entity began association with this location (ISO 8601 format: YYYY-MM-DD).")
    end_date: Optional[str] = Field(default=None, alias="to", description="The date when the entity ended association with this location (ISO 8601 format: YYYY-MM-DD).")

class Occupation(StrictBaseModel):
    """
    Details regarding a professional role or job position held by an entity.
    """
    title: str = Field(..., description="The job title or role name (e.g., 'CEO', 'Director', 'Software Engineer').")
    institution: str = Field(...,  description="The name of the company, organization, or institution where this role was held.")
    start_date: Optional[str] = Field(default=None, alias="from", description="The start date of the employment (ISO 8601 format).")
    end_date: Optional[str] = Field(default=None, alias="to", description="The end date of the employment (ISO 8601 format).")

class IDNumber(StrictBaseModel):
    """
    Government or organizational identification numbers.
    """
    type: str = Field(..., description="The type of ID (e.g., 'National ID', 'Passport', 'SSN', 'Tax ID', 'Driver License').")
    number: str = Field(..., description="The alphanumeric string representing the identification number.")

class Relation(StrictBaseModel):
    """
    Represents a relationship between the profile entity and another entity or person.
    """
    name: str = Field(..., description="The name of ALL the related person or organization.")
    relation: str = Field(...,description="The nature of the relationship (e.g., 'Spouse', 'Subsidiary', 'Parent Company', 'Associate').")
    risk_analysis: Optional[str] = Field(default=None, description="Any relevant risk information about this relationship.")
    risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = Field(default=None, description="The assessed risk level associated with this relationship.")

class EntityProfile(StrictBaseModel):
    """
    A comprehensive profile of a legal entity or natural person, containing biographical and professional details.
    """
    type: List[str] = Field(
        default=[], 
        description="The classification of the entity (e.g., 'Person', 'Organization').",
        json_schema_extra={"examples": [["Person"], ["Organization"]]}
    )
    primary_name: List[str] = Field(
        default=[], 
        description="The official or legal name(s) of the Person/Organization.",
        json_schema_extra={"examples": [["John Michael Smith"], ["Acme Corporation Ltd"]]}
    )
    aliases: List[str] = Field(
        default=[], 
        description="Alternative names, nicknames, trading names, or 'Doing Business As' (DBA) names.",
        json_schema_extra={"examples": [["Johnny", "J.M. Smith"], ["Acme Corp", "Acme Inc"]]}
    )
    gender: List[str] = Field(
        default=[], 
        description="Gender of the person.",
        json_schema_extra={"examples": [["Male"], ["Female"], ["Unknown"]]}
    )
    date_of_birth: List[str] = Field(
        default=[], 
        description="Date of birth of the Person in ISO 8601 format (YYYY-MM-DD).",
        json_schema_extra={"examples": [["1985-03-15"], ["1990-12-01"]]}
    )
    citizenship: List[str] = Field(
        default=[], 
        description="Countries where the Person holds citizenship or nationality.",
        json_schema_extra={"examples": [["United States", "Canada"], ["United Kingdom"]]}
    )
    place_of_birth: List[str] = Field(
        default=[], 
        description="City and country where the person was born.",
        json_schema_extra={"examples": [["New York, United States"], ["London, United Kingdom"]]}
    )
    deceased: List[str] = Field(
        default=[], 
        description="Indication if the person is deceased, or date of death.",
        json_schema_extra={"examples": [["Yes"], ["No"]]}
    )
    id_numbers: List[IDNumber] = Field(
        default=[], 
        description="A list of identification documents that are owned by the subject.",
        json_schema_extra={"examples": [[{"type": "Passport", "number": "ABC123"}]]}
    )
    domicile: List[Location] = Field(
        default=[], 
        description="The legal home or permanent residence of the Person.",
        json_schema_extra={"examples": [[{"location": "London, UK", "from": "2020-01-01", "to": "2024-12-31"}]]}
    )
    addresses: List[Location] = Field(
        default=[], 
        description="Known physical addresses associated with the Person/Organization.",
        json_schema_extra={"examples": [[{"location": "123 Main St, New York, NY", "from": "2018-06-01", "to": "2022-08-15"}]]}
    )
    roles_primary_occupation: List[Occupation] = Field(
        default=[], 
        description="The current or most significant professional roles/employment held by the Person.",
        json_schema_extra={"examples": [[{"title": "CEO", "institution": "Tech Corp", "from": "2020-01-01", "to": "2024-12-31"}]]}
    )
    roles_history_occupation: List[Occupation] = Field(
        default=[], 
        description="Past employment history of the Person.",
        json_schema_extra={"examples": [[{"title": "Manager", "institution": "Old Company", "from": "2015-03-01", "to": "2019-12-31"}]]}
    )
    associated_entities: List[Relation] = Field(
        default=[], 
        description="Companies or organizations linked to the subject.",
        json_schema_extra={"examples": [[{"name": "Subsidiary Inc", "relation": "Parent Company"}]]}
    )
    associated_persons: List[Relation] = Field(
        default=[], 
        description="Natural Persons (family, business partners) linked to the subject. The relation field describes the nature of the relationship.",
        json_schema_extra={"examples": [[{"name": "Jane Doe", "relation": "Spouse"}, {"name": "Bob Smith", "relation": "Business Partner"}]]}
    )
    date_of_incorporation: List[str] = Field(
        default=[], 
        description="If a company/organization, the date it was legally formed.",
        json_schema_extra={"examples": [["2010-06-15"], ["1995-01-20"]]}
    )
    country_of_incorporation: List[str] = Field(
        default=[], 
        description="If a company/organization, the jurisdiction under whose laws it was formed.",
        json_schema_extra={"examples": [["Delaware, United States"], ["Cayman Islands"]]}
    )
    country_of_affiliation: List[str] = Field(
        default=[], 
        description="If a company/organization, countries where the entity operates or has significant ties.",
        json_schema_extra={"examples": [["United States", "United Kingdom", "Singapore"]]}
    )

class NewsEntityProfile(EntityProfile):
    news_analysis: str = Field(..., description="A brief summary of the context in which this entity is mentioned in the news article.")



class FieldMatchResult(StrictBaseModel):
    field_name: str
    decision: Literal["MATCH", "MISMATCH", "UNCERTAIN"]
    confidence: Literal["HIGH", "MODERATE", "LOW"]
    reasoning: str

class FieldQualityCheck(StrictBaseModel):
    """QA validation result for field comparison."""
    is_valid: bool
    feedback: str
    quality: Literal["HIGH", "MODERATE", "LOW"]

# Changed_20260120
class MatchResult(StrictBaseModel):
    field_results: List[Any] = []
    scores: dict = {}

# ==========================================
# Scoring Models
# ==========================================
# Changed_20260120

# Field Groups for comparison
FIELD_GROUPS = {
    "type": ["type"],
    "names": [
        "primary_name", 
        "aliases", 
    ],
    "id_numbers":["id_numbers"],
    "locations": [
        "citizenship", 
        "place_of_birth", 
        "addresses", 
        "domicile", 
        "country_of_incorporation", 
        "country_of_affiliation"
    ],
    "gender": ["gender"], 
    "date_of_birth": ["date_of_birth"], 
    "deceased": ["deceased"],
    "date_of_incorporation": ["date_of_incorporation"],
    "roles": [
        "roles_primary_occupation", 
        "roles_history_occupation"
    ],
    "relations": [
        "associated_entities", 
        "associated_persons"
    ]
}

# HIGH: Critical identifiers (Strongest signal)
# STD:  Standard corroboration 
# NONE: No impact
HIGH = 2
STD  = 1
NONE = 0

RULE_BASED_SCORE = {
    "type": {
        "support": NONE,
        "reject": HIGH, # Critical: Person vs Company mismatch
    },
    "primary_name": {
        "support": HIGH, # Name match is critical
        "reject": STD,   # Mismatch is a standard penalty (allows for fuzzy errors)
        "threshold":[66,90] # below 66% is Mismatch, above 90% is Match, between is Uncertain
    },
    "aliases": {
        "support": STD,
        "reject": NONE
    },
    "id_numbers": {
        "support": HIGH, # ID match is critical
        "reject": NONE
    },
    "citizenship": {
        "support": STD,
        "reject": NONE
    },
    "place_of_birth": {
        "support": STD,
        "reject": NONE
    },
    "addresses": {
        "support": STD,
        "reject": NONE
    },
    "domicile": {
        "support": STD,
        "reject": NONE
    },
    "country_of_incorporation": {
        "support": STD,
        "reject": NONE
    },
    "country_of_affiliation": {
        "support": STD,
        "reject": NONE
    },
    "gender": {
        "support": NONE,
        "reject": HIGH # Critical gender mismatch
    },
    "date_of_birth": {
        "support": HIGH,
        "reject": STD
    },
    "deceased": {
        "support": NONE,
        "reject": STD
    },
    "date_of_incorporation": {
        "support": HIGH,
        "reject": STD
    },
    "roles_primary_occupation": {
        "support": HIGH,
        "reject": NONE
    },
    "roles_history_occupation": {
        "support": HIGH,
        "reject": NONE
    },
    "associated_entities": {
        "support": HIGH,
        "reject": NONE
    },
    "associated_persons": {
        "support": HIGH,
        "reject": NONE
    }
}

LLM_BASED_SCORE = {
    # Field Groupings for LLM-based scoring
    "type": {
        # Critical: If Entity Type mismatches, it is a hard stop.
        "support": NONE, 
        "reject":  HIGH 
    },
    "names": {
        # Strong Identity: A name match is definitive.
        "support": HIGH, 
        "reject":  STD 
    },
    "id_numbers": {
        # Strong Identity: ID matches are unique identifiers.
        "support": HIGH, 
        "reject":  NONE 
    },
    "locations": {
       # Location matches confirm identity but don't prove it alone.
        "support": STD, 
        "reject":  NONE 
    },
    "gender": {
        # Critical Check: Mismatch is a strong negative signal.
        "support": NONE, 
        "reject":  HIGH 
    },
    "date_of_birth": {
        # Strong Identity: Exact DOB match is critical.
        "support": HIGH, 
        "reject":  STD 
    },
    "deceased": {
        # Standard check.
        "support": NONE, 
        "reject":  STD 
    },
    "date_of_incorporation": {
        # Strong Identity for companies.
        "support": HIGH, 
        "reject":  STD 
    },
    "roles": {
        # Strong Context: Matching occupation/role is a very specific signal.
        "support": HIGH, 
        "reject":  NONE 
    },
    "relations": {
        # Strong Context: Matching known associates is a very specific signal.
        "support": HIGH, 
        "reject":  NONE 
    }
}
