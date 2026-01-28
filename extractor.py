import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, TypeVar

from dotenv import load_dotenv
from fastapi_poe.client import get_bot_response
from fastapi_poe.types import ProtocolMessage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from models import NewsEntityProfile, StrictBaseModel, Relation
from utils.name_match import calculate_name_score, MATCH_THRESHOLD

# ==========================================================================
# Configuration
# ==========================================================================

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

API_KEY = os.getenv("API_KEY")
BOT_NAME = os.getenv("BOT_NAME", "GPT-4o")

if not API_KEY:
    raise EnvironmentError("API_KEY environment variable is required")

# Field classification by entity type
PERSON_ONLY_FIELDS: set[str] = {"gender","date_of_birth","citizenship","place_of_birth", "deceased", "domicile",
                                "roles_primary_occupation","roles_history_occupation","age","year_of_birth"}

ORG_ONLY_FIELDS: set[str] = {"date_of_incorporation","country_of_incorporation","country_of_affiliation"}

RELATION_FIELDS: set[str] = {"associated_entities", "associated_persons"}

SKIP_FIELDS: set[str] = {"type", "primary_name", "aliases"}



# Generic retry decorator for LLM calls
log_reraise = lambda s: (logger.error(f"LLM failed after {s.attempt_number} attempts: {s.outcome.exception()}"), s.outcome.exception())[1] or (_ for _ in ()).throw(s.outcome.exception())
llm_retry = retry(stop=stop_after_attempt(3), retry_error_callback=log_reraise)

# Retry decorator for business logic errors during extraction
trap_extract_field = retry(
    stop=stop_after_attempt(1),
    retry_error_callback=lambda s: (logger.error(f"Extract Field Failed: {s.outcome.exception()}"), [])[1]
)

trap_extract_relation = retry(
    stop=stop_after_attempt(1),
    retry_error_callback=lambda s: (logger.error(f"Extract Relation Failed: {s.outcome.exception()}"), None)[1]
)

trap_analyze_news = retry(
    stop=stop_after_attempt(1),
    retry_error_callback=lambda s: (logger.error(f"Extract Relation Failed: {s.outcome.exception()}"), None)[1]
)
# ==========================================================================
# Data Models
# ==========================================================================

T = TypeVar("T", bound=BaseModel)


class ExtractedEntity(StrictBaseModel):
    """A single entity with name variants and confidence level."""

    type: Literal["Person", "Organization"] = Field(..., description="Entity type classification")
    names: list[str] = Field( ...,description="List of name variants with the most complete/formal name first",)
    confidence: Literal["High", "Medium", "Low"] = Field(..., description="Confidence level based on evidence quality")


class ResolvedText(StrictBaseModel):
    """Text after coreference resolution."""
    text: str = Field( ..., description="Text with pronouns replaced by entity names")


class EntityContext(StrictBaseModel):
    """Entity with associated text snippets."""
    type: Literal["Person", "Organization"] = Field(..., description="Entity type classification")
    names: list[str] = Field(..., description="Full names, aliases, and name variants")
    focus_text: list[str] = Field(..., description="Text snippets mentioning the entity")

class FieldValidation(StrictBaseModel):
    """Field validation result."""
    is_valid: bool
    feedback: Optional[str] = None

class TextSnippets(StrictBaseModel):
    """List of text snippets."""
    text: list[str] = Field(..., description="Array of text snippets")

class EntityListWrapper(StrictBaseModel):
    """Wrapper for entity list, used for JSON parsing."""
    entities: list[ExtractedEntity] = Field(..., description="List of extracted entities")

class RelationExtract(StrictBaseModel):
    relation: str
    risk_analysis: Optional[str] = None
    risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
# ==========================================================================
# Utility Functions
# ==========================================================================

def serialize(val: Any) -> Any:
    """Recursively serialize Pydantic models to dictionaries."""
    if isinstance(val, dict):
        return {k: serialize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [serialize(v) for v in val]
    if isinstance(val, BaseModel):
        return val.model_dump()
    return val

def extract_json_from_text(text: str) -> str:
    """Extract JSON string from LLM response."""
    patterns = [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    ]
    for pattern in patterns:
        if match := re.search(pattern, text, re.DOTALL):
            return match.group(1)
    return text.strip()


def create_field_response_model(
    field_name: str, field_annotation: Any
) -> type[BaseModel]:
    """Create a dynamic Pydantic model for field extraction responses."""
    return type(
        f"{field_name.capitalize()}Response",
        (StrictBaseModel,),
        {"__annotations__": {"value": field_annotation, "reasoning": str}},
    )


def combine_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Merge entities sharing names using union-find."""
    if not entities:
        return []

    parent = list(range(len(entities)))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    seen: dict[tuple[str, str], int] = {}
    for i, entity in enumerate(entities):
        for name in entity.names:
            key = (name.lower().strip(), entity.type)
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i

    groups: dict[int, dict[str, Any]] = defaultdict(lambda: {"names": {}, "has_high_confidence": False})
    for i, entity in enumerate(entities):
        group = groups[find(i)]
        group["type"] = entity.type
        group["names"] |= dict.fromkeys(entity.names)
        group["has_high_confidence"] |= entity.confidence == "High"

    return [
        ExtractedEntity(
            type=g["type"],
            names=list(g["names"]),
            confidence="High" if g["has_high_confidence"] else "Medium",
        )
        for g in groups.values()
    ]


def find_target_entity(subject_name: str, entities: list[ExtractedEntity]) -> ExtractedEntity:
    """Match target entity using name similarity scoring."""
    if not entities:
        raise ValueError("No entities available for matching")

    def score_entity(entity: ExtractedEntity) -> float:
        return max(
            calculate_name_score(subject_name, name, return_raw=True)
            for name in entity.names
        )

    best_match = max(entities, key=score_entity)
    logger.info("Matched '%s' -> %s", subject_name, best_match.names[0])
    return best_match


def count_populated_fields(profile: NewsEntityProfile) -> int:
    """Count the number of populated fields in a profile."""
    return sum( 1 for v in serialize(profile).values() if v not in (None, [], {}, "") )


def build_feedback_section(feedback_history: list[dict[str, Any]]) -> str:
    """Build prompt text from feedback history."""
    if not feedback_history:
        return ""

    sections = []
    for i, log in enumerate(feedback_history, start=1):
        sections.append(
            f"PREVIOUS REJECTED ATTEMPT {i}:\n"
            f"Rejected Value: {json.dumps(serialize(log['value']), ensure_ascii=False)}\n"
            f"QA FEEDBACK: {log['feedback']}"
        )
    return "\n\n" + "\n\n".join(sections) + "\n\nFIX THE ISSUES ABOVE.\n"

# ==========================================================================
# LLM Communication
# ==========================================================================

async def fetch_llm_response(prompt: str) -> str:
    """Send request to LLM and return raw response."""
    chunks: list[str] = []
    async for partial in get_bot_response(
        messages=[ProtocolMessage(role="user", content=prompt)],
        bot_name=BOT_NAME,
        api_key=API_KEY,
    ):
        chunks.append(partial.text)
    return "".join(chunks)


@llm_retry
async def query_llm(prompt: str, model_class: type[T]) -> T:
    """
    Send prompt to LLM, parse and validate the response.

    Combines network request and JSON parsing to ensure the return value
    conforms to the specified schema.
    """
    raw_response = await fetch_llm_response(prompt)
    json_str = extract_json_from_text(raw_response)
    return model_class.model_validate_json(json_str)


# ==========================================================================
# Pipeline Steps
# ==========================================================================

async def detect_entities(text: str) -> list[ExtractedEntity]:
    """Extract named entities from text."""
    prompt = f"""Extract all person and organization names from the text.

###### Rules:
- Person: named individuals (include titles if present)
- Organization: companies, agencies, institutions (not locations)
- Group name variants only when explicitly referring to the same entity
- List the most complete name first in the "names" array

###### Confidence: High = full name + clear context, Medium = partial name, Low = plausible but uncertain

###### Example: "Dr. Jane Smith of Acme Corp met with Smith yesterday."
Output: {{"entities": [{{"type": "Person", "names": ["Dr. Jane Smith", "Smith"], "confidence": "High"}}, {{"type": "Organization", "names": ["Acme Corp"], "confidence": "High"}}]}}

###### Text:
{text}

Output JSON only:"""

    logger.info("-> Detecting entity names")
    result = await query_llm(prompt, EntityListWrapper)

    person_count = sum(1 for e in result.entities if e.type == "Person")
    org_count = sum(1 for e in result.entities if e.type == "Organization")
    logger.info("Detected %d persons, %d organizations", person_count, org_count)

    return result.entities


async def resolve_coreferences(text: str) -> str:
    """Resolve pronouns to entity names."""
    prompt = f"""Rewrite the text by resolving pronouns to their specific entity names.

###### Rules:
1. Replace pronouns (he, she, it, they, him, her, his, hers, their, them, who, whom, whose, himself, herself, themselves) with the entity's full name
2. For possessive pronouns, make the name possessive (e.g., "his" -> "John's")
3. Do NOT replace dummy pronouns in abstract phrases (e.g., "It is raining")
4. If ambiguous, keep the original pronoun
5. Keep everything else unchanged

###### Text:
{text}

OUTPUT JSON: {{"text": str}}"""

    logger.info("-> Resolving coreferences (%d characters)", len(text))
    result = await query_llm(prompt, ResolvedText)
    return result.text


async def extract_entity_context(entity: ExtractedEntity, text: str) -> EntityContext:
    """Extract text snippets related to the entity."""
    names_list = ", ".join(f'"{name}"' for name in entity.names)

    prompt = f"""Extract ALL text passages mentioning this entity.

###### Entity: {entity.names[0]} ({entity.type})
###### Aliases: {names_list}

###### INCLUDE:
- Whole paragraphs or sentence groups containing any name/alias
- ALL surrounding sentences needed for context (relationships, actions, attributes)
- Pronoun chains clearly referring to this entity

###### Return JSON: {{"text": []}}

###### Example:
Input: "John Smith joined Acme Corp in 2019. He quickly rose through the ranks. By 2020, he became CEO and led a major restructuring. The company later hired Jane Doe."
Entity: John Smith
Output: {{"text": ["John Smith joined Acme Corp in 2019. He quickly rose through the ranks. By 2020, he became CEO and led a major restructuring."]}}

###### Source Text:
{text}"""

    logger.info("-> Extracting context: %s", entity.names[0])
    result = await query_llm(prompt, TextSnippets)

    context = EntityContext(
        names=entity.names,
        type=entity.type,
        focus_text=result.text,
    )
    logger.info(f"Found {len(context.focus_text)} relevant passages")
    return context

async def validate_field(focus_text: str, entity_name: str, field_name: str, extracted_value: Any, reasoning: str) -> FieldValidation:
    """Validate the quality of field extraction."""
    prompt = f"""Role: Data Quality Auditor
###### Task: Find ERRORS or HALLUCINATIONS in the extraction below.

###### Entity: "{entity_name}"
###### Field: {field_name}
###### Text: {focus_text}
###### Proposed Extraction: {json.dumps(serialize(extracted_value), ensure_ascii=False)}
###### Reasoning: {reasoning}

###### RULES:
1. Check if Proposed Extraction is valid by considering the Text and Reasoning
2. Provide super concise feedback on errors and fixes

OUTPUT JSON: {{"is_valid": bool, "feedback": str}}"""

    return await query_llm(prompt, FieldValidation)



@trap_extract_field
async def extract_field(focus_text: str,entity_name: str,field_name: str,use_reflexion: bool = True,max_attempts: int = 2,) -> Any:

    """Extract a single field value with optional self-correction."""
    field_info = NewsEntityProfile.model_fields[field_name]

    format_hint = '["value"]'
    schema_extra = getattr(field_info, "json_schema_extra", None)
    if schema_extra and (examples := schema_extra.get("examples")):
        format_hint = json.dumps(examples[0])

    response_model = create_field_response_model(field_name, field_info.annotation)
    feedback_history: list[dict[str, Any]] = []
    iterations = max_attempts if use_reflexion else 1

    for attempt in range(iterations):
        feedback_section = build_feedback_section(feedback_history)

        prompt = f"""###### Entity: "{entity_name}"
###### Extract: {field_info.description}
###### Focus Text:
{focus_text}

###### Feedback from previous attempts:
{feedback_section}

###### Example JSON:
{{
  "value": {format_hint} or [],
  "reasoning": "Explain concise evidence, reasoning and source"
}}"""

        result = await query_llm(prompt, response_model)

        if not use_reflexion:
            logger.info(f"{entity_name} - {field_name}: {result.reasoning}")
            return result.value

        validation = await validate_field(focus_text, entity_name, field_name, result.value, result.reasoning)

        if validation.is_valid:
            logger.info(f"[{entity_name} - {field_name}] Attempt {attempt + 1} passed: \n Value: {result.value}")
            return result.value
        else:
            logger.info(f"[{entity_name} - {field_name}] Attempt {attempt + 1} failed: {validation.feedback}\n Value: {result.value}")
            feedback_history.append({"value": result.value, "reasoning": result.reasoning, "feedback": validation.feedback})

    logger.warning(f"[{entity_name} - {field_name}] Max retries reached, returning empty value")
    return []


async def validate_relation(
    focus_text: str, 
    entity_name: str, 
    another_entity_name: str, 
    extracted_relation: Relation
) -> FieldValidation:
    """Validate extraction - reject only for factual errors."""
    
    prompt = f"""Validate extraction against text.

TEXT:
{focus_text}

EXTRACTION:
{entity_name} -> {another_entity_name}
Relation: {extracted_relation.relation}
Risk: {extracted_relation.risk_level}
Analysis: {extracted_relation.risk_analysis}

ACCEPT IF:
- "{another_entity_name}" absent from text and relation="None" and risk="LOW"
- Relation and risk accurately reflect text

REJECT IF:
- Relation contradicts text
- Risk=LOW but "{another_entity_name}" is accused/wanted/investigated
- Risk=HIGH but "{another_entity_name}" is victim/law enforcement

OUTPUT:
{{"is_valid": true, "feedback": null}}
{{"is_valid": false, "feedback": "<error>"}}"""

    return await query_llm(prompt, FieldValidation)


@trap_extract_relation
async def extract_relation(
    focus_text: str, 
    entity_name: str, 
    another_entity_name: str, 
    use_reflexion: bool = True, 
    max_attempts: int = 2
) -> Optional[Relation]:
    """Extract relationship information between two entities with optional self-correction."""
    
    iterations = max_attempts if use_reflexion else 1
    feedback_section = ""
    last_relation: Optional[Relation] = None

    for attempt in range(iterations):
        prompt = f"""Analyze relationship: "{entity_name}" → "{another_entity_name}"

### TEXT:
{focus_text}

### FEEDBACK FROM PREVIOUS ATTEMPTS:
{feedback_section}

### INSTRUCTIONS:
1. State their relationship from the text, or relation="None" if unconnected or {another_entity_name} is not mentioned
2. Assess compliance risk for "{another_entity_name}":
   - HIGH: On wanted/sanctions list, accused of financial crimes, under investigation
   - MEDIUM: Close associate of high-risk person, disputed allegations, related litigation
   - LOW: Victim with no accusations, law enforcement, no suspicious ties, or unconnected or not mentioned in the text
3. Accused individuals are MEDIUM or HIGH regardless of denial

### OUTPUT JSON:
{{
  "relation": "<relationship or None>",
  "risk_analysis": "<brief justification for risk level of {another_entity_name}, NOT {entity_name}>",
  "risk_level": "<HIGH|MEDIUM|LOW>"
}}"""

        result = await query_llm(prompt, RelationExtract)
        last_relation = Relation(
            name=another_entity_name,
            relation=result.relation or "None",
            risk_analysis=result.risk_analysis or "No information available.",
            risk_level=result.risk_level or "LOW"
        )

        if not use_reflexion:
            return last_relation

        validation = await validate_relation(focus_text, entity_name, another_entity_name, last_relation)

        if validation.is_valid:
            logger.info(f"[{entity_name} -> {another_entity_name}] Attempt {attempt + 1} passed: {last_relation}")
            return last_relation

        logger.info(f"[{entity_name} -> {another_entity_name}] Attempt {attempt + 1} failed: {last_relation} \n{validation.feedback}")
        feedback_section = f"\n### Previous Attempt Failed:\n{validation.feedback}\nFix this issue.\n"

    logger.warning(f"[{entity_name} -> {another_entity_name}] Max retries reached")
    return last_relation

@trap_analyze_news
async def analyze_news(focus_text: str, entity_name: str) -> str:
    """Analyze compliance risk based on news content."""
    
    class NewsAnalysisResponse(StrictBaseModel):
        risk_summary: str = Field(..., description="Brief risk assessment summary")

    prompt = f"""Analyze compliance risk for: "{entity_name}"

### TEXT:
{focus_text}

### INSTRUCTIONS:
Assess compliance risk level and provide a concise summary:
- HIGH: On wanted/sanctions list, accused of financial crimes, under investigation
- MEDIUM: Close associate of high-risk person, disputed allegations, related litigation
- LOW: Victim with no accusations, law enforcement, no suspicious ties

### OUTPUT JSON:
{{
  "risk_summary": "<risk level and brief justification>"
}}"""

    result = await query_llm(prompt, NewsAnalysisResponse)
    return result.risk_summary

async def validate_dob(
    focus_text: str,
    entity_name: str,
    news_date: str,
    extracted_value: list[str],
    reasoning: str
) -> FieldValidation:
    """Validate DOB extraction against text and news date."""
    prompt = f"""Validate date of birth extraction.

### ENTITY: "{entity_name}"
### NEWS DATE: {news_date}
### TEXT: {focus_text}
### EXTRACTED DOB: {json.dumps(extracted_value)}
### REASONING: {reasoning}

### REJECT IF:
- Calculation error (e.g., age 32 in 2020 → 1987 not 1988)
- DOB after news_date
- Contradicts explicit date in text

### ACCEPT IF:
- Calculation correct
- Matches text evidence
- Format valid ("YYYY-MM-DD" or "YYYY-MM" or "YYYY")

OUTPUT JSON: {{"is_valid": bool, "feedback": str}}"""

    return await query_llm(prompt, FieldValidation)


@trap_extract_field
async def extract_dob(
    focus_text: str,
    entity_name: str,
    news_date: str,
    use_reflexion: bool = True,
    max_attempts: int = 2
) -> list[str]:
    """Extract and derive date of birth with validation loop."""
    
    class DOBResponse(StrictBaseModel):
        value: list[str] = Field(..., description="Derived DOB in YYYY-MM-DD format")
        reasoning: str = Field(..., description="Calculation or evidence")
    
    feedback_history: list[dict[str, Any]] = []
    iterations = max_attempts if use_reflexion else 1

    for attempt in range(iterations):
        feedback_section = build_feedback_section(feedback_history)

        prompt = f"""Extract date of birth for "{entity_name}".

### NEWS DATE: {news_date}
### TEXT:
{focus_text}

### FEEDBACK FROM PREVIOUS ATTEMPTS:
{feedback_section}

### DERIVE FROM:
- Explicit DOB: "born June 15, 1980" → ["1980-06-15"]
- Age: "32 years old" + news_date 2020-10-28 → ["1988"] (birth year = news_year - age)
- Year & Month only: "born in 1985 March" → ["1985-03"]
- Year only: "born in 1985" → ["1985"]
- Return [] if no evidence

### OUTPUT JSON:
{{
  "value": ["YYYY-MM-DD"] or ["YYYY-MM"] or ["YYYY"] or [],
  "reasoning": "Brief calculation/evidence"
}}"""

        result = await query_llm(prompt, DOBResponse)

        if not use_reflexion:
            logger.info(f"{entity_name} - DOB: {result.reasoning}")
            return result.value

        validation = await validate_dob(focus_text, entity_name, news_date, result.value, result.reasoning)

        if validation.is_valid:
            logger.info(f"[{entity_name} - DOB] Attempt {attempt + 1} passed: {result.value}")
            return result.value
        else:
            logger.info(f"[{entity_name} - DOB] Attempt {attempt + 1} failed: {validation.feedback}")
            feedback_history.append({
                "value": result.value,
                "reasoning": result.reasoning,
                "feedback": validation.feedback
            })

    logger.warning(f"[{entity_name} - DOB] Max retries reached, returning empty")
    return []


async def extract_all_fields(entity_context: EntityContext, all_entities: list[ExtractedEntity], news_date: str) -> NewsEntityProfile:
    """Extract a complete structured profile for an entity."""
    entity_name = entity_context.names[0]
    entity_type = entity_context.type
    focus_text = "\n".join(entity_context.focus_text)

    logger.info(f"-> Extracting profile: {entity_name} ({entity_type})")

    # Determine which fields to skip based on entity type
    type_specific_skip = ORG_ONLY_FIELDS if entity_type == "Person" else PERSON_ONLY_FIELDS
    skip_fields = SKIP_FIELDS | RELATION_FIELDS | type_specific_skip | {"news_analysis","date_of_birth"} 

    # Extract regular fields sequentially
    field_values: dict[str, Any] = {}
    for field in NewsEntityProfile.model_fields:
        if field not in skip_fields:
            field_values[field] = await extract_field(focus_text, entity_name, field)

    # Find other entities (exclude self by name matching)
    own_names = {name.lower() for name in entity_context.names}
    other_entities = [
        e for e in all_entities 
        if not any(n.lower() in own_names for n in e.names)
    ]

    # Extract relationships sequentially
    person_relations: list[Relation] = []
    org_relations: list[Relation] = []

    for other_entity in other_entities:
        relation = await extract_relation(focus_text, entity_name, other_entity.names[0])
        if relation is None:
            continue
        if other_entity.type == "Person":
            person_relations.append(relation)
        elif other_entity.type == "Organization":
            org_relations.append(relation)

    # Build profile
    profile = NewsEntityProfile.model_validate({
        "type": [entity_type],
        "primary_name": [entity_context.names[0]],
        "aliases": entity_context.names[1:],
        "date_of_birth": await extract_dob(focus_text, entity_name, news_date),
        "news_analysis": await analyze_news(focus_text, entity_name),
        "associated_persons": person_relations,
        "associated_entities": org_relations,
        **field_values,
        **{field: [] for field in type_specific_skip},
    })

    logger.info(f"Extraction complete: {count_populated_fields(profile)}/{len(NewsEntityProfile.model_fields)} fields")
    return profile

# ==========================================================================
# Main Pipeline
# ==========================================================================


async def extract_profile(
    news_date: str,
    subject_name: str, text: str
) -> NewsEntityProfile:
    """Extract a profile for a single subject from text."""
    logger.info("=== Extracting profile: %s ===\n", subject_name)

    entities = await detect_entities(text)
    text_resolved = await resolve_coreferences(text)
    entities_resolved = await detect_entities(text_resolved)

    all_entities = combine_entities(entities + entities_resolved)
    print("[All Entities]", all_entities)
    target_entity = find_target_entity(subject_name, all_entities)

    context = await extract_entity_context(target_entity, text_resolved)
    print("[Context]", context)

    profile = await extract_all_fields(context, all_entities, news_date)

    logger.info("=== Profile extraction complete ===")
    return profile


# ==========================================================================
# Entry Point
# ==========================================================================


async def main() -> None:
    """Test the extraction pipeline."""
    #test_file = Path(__file__).parent / "news_dashing.txt"
    test_file = Path(__file__).parent / "news_husband.txt"
   # test_file = Path(__file__).parent / "news_age.txt"
    test_text = test_file.read_text(encoding="utf-8").strip()

    profile = await extract_profile("Oct 28, 2020", "Fang Liu", test_text)
    serialized = serialize(profile)
    filtered = {k: v for k, v in serialized.items() if v not in (None, [], {}, "")}
    print(json.dumps(filtered, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

    
