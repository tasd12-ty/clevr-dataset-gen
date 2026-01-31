"""
约束提取的提示词构建器。

Prompt builders for constraint extraction from images.

This module provides prompts for:
- Single-view constraint extraction (Task-3)
- Multi-view constraint extraction (Task-2)
"""

from typing import Dict, List, Optional, Any
from pathlib import Path


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_BASE = """You are a spatial reasoning expert analyzing 3D scenes from images.

## Core Principles

1. **Relative, Not Absolute**: Focus on RELATIVE spatial relationships, not exact coordinates.
2. **3D Reasoning**: Reason about 3D space, not just 2D image positions.
3. **Use Depth Cues**: Leverage occlusion, perspective, shadows, and relative sizes.
4. **Consistency**: Ensure all constraints are globally consistent (no contradictions).

## Output Format

Return ONLY valid JSON. No markdown, no explanation outside JSON.
"""

SYSTEM_PROMPT_SINGLE_VIEW = SYSTEM_PROMPT_BASE + """
## Task: Single-View Constraint Extraction (Task-3)

Analyze the image and extract ALL observable spatial constraints.

### Constraint Types to Extract:

1. **Objects**: Identify all visible objects with:
   - id: Unique identifier (e.g., "red_cube", "blue_sphere")
   - type: Shape type (cube, sphere, cylinder, cone, etc.)
   - color: Object color
   - size_class: tiny / small / medium / large

2. **Axial Relations** (between pairs of objects):
   - left_of / right_of (X-axis in image)
   - above / below (Y-axis in image)
   - in_front_of / behind (depth/Z-axis)

3. **Topology** (RCC-8 subset):
   - disjoint: Objects are separated
   - touching: Objects share a boundary
   - overlapping: Objects partially overlap

4. **Occlusion** (view-dependent):
   - Which object occludes which
   - partial: true/false

5. **Size Comparisons**:
   - Which object appears bigger than which

6. **Closer (Ternary Distance)**:
   - closer(anchor, B, C): B is closer to anchor than C

7. **QRR (Quaternary Distance Comparison)** (optional, if confident):
   - Compare dist(A,B) vs dist(C,D)
   - Use comparator: "<" (less than), "~=" (approximately equal), ">" (greater than)

8. **TRR (Clock Direction)** (optional):
   - Target object's clock position (1-12) relative to reference axis

### Tolerance Parameter (tau)

When comparing distances, use tolerance tau={tau}:
- "<" means strictly less (outside tolerance band)
- "~=" means approximately equal (within tolerance)
- ">" means strictly greater

### Consistency Rules

- If A is left_of B, then B is right_of A
- If A is in_front_of B and B is in_front_of C, then A is in_front_of C
- Distance comparisons must not form contradictory cycles
"""

SYSTEM_PROMPT_MULTI_VIEW = SYSTEM_PROMPT_BASE + """
## Task: Multi-View Constraint Extraction (Task-2)

Analyze multiple images of the SAME scene from different viewpoints.

### Goals:

1. Identify view-INVARIANT constraints (hold across all views):
   - 3D distances, topology, physical sizes

2. Identify view-DEPENDENT constraints (vary by viewpoint):
   - Occlusion, 2D positions, apparent sizes

3. Resolve ambiguities using multiple views

### Process:

1. Analyze each view independently
2. Cross-reference to find consistent 3D relationships
3. Mark view-dependent constraints with camera info
"""


# =============================================================================
# Output Schema
# =============================================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "color": {"type": "string"},
                    "size_class": {"type": "string", "enum": ["tiny", "small", "medium", "large"]},
                },
                "required": ["id", "type", "color"]
            }
        },
        "constraints": {
            "type": "object",
            "properties": {
                "axial": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "obj1": {"type": "string"},
                            "obj2": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": ["left_of", "right_of", "above", "below", "in_front_of", "behind"]
                            }
                        },
                        "required": ["obj1", "obj2", "relation"]
                    }
                },
                "topology": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "obj1": {"type": "string"},
                            "obj2": {"type": "string"},
                            "relation": {"type": "string", "enum": ["disjoint", "touching", "overlapping"]}
                        },
                        "required": ["obj1", "obj2", "relation"]
                    }
                },
                "occlusion": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "occluder": {"type": "string"},
                            "occluded": {"type": "string"},
                            "partial": {"type": "boolean"}
                        },
                        "required": ["occluder", "occluded"]
                    }
                },
                "size": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bigger": {"type": "string"},
                            "smaller": {"type": "string"}
                        },
                        "required": ["bigger", "smaller"]
                    }
                },
                "closer": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "anchor": {"type": "string"},
                            "closer": {"type": "string"},
                            "farther": {"type": "string"}
                        },
                        "required": ["anchor", "closer", "farther"]
                    }
                },
                "qrr": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair1": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                            "pair2": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                            "metric": {"type": "string"},
                            "comparator": {"type": "string", "enum": ["<", "~=", ">"]}
                        },
                        "required": ["pair1", "pair2", "comparator"]
                    }
                },
                "trr": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "ref1": {"type": "string"},
                            "ref2": {"type": "string"},
                            "hour": {"type": "integer", "minimum": 1, "maximum": 12}
                        },
                        "required": ["target", "ref1", "ref2", "hour"]
                    }
                }
            }
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["objects", "constraints", "confidence"]
}


# =============================================================================
# User Prompt Templates
# =============================================================================

SINGLE_VIEW_USER_TEMPLATE = """Analyze this image and extract all spatial constraints.

{object_context}

Parameters:
- Tolerance (tau): {tau}

Extract:
1. All visible objects with id, type, color, size_class
2. Axial relations (left_of, right_of, above, below, in_front_of, behind)
3. Topology (disjoint, touching, overlapping)
4. Occlusion (which object occludes which)
5. Size comparisons (which is bigger)
6. Closer relations (which object is closer to which)
7. QRR if confident (compare distances between pairs)
8. TRR if confident (clock positions)

Return JSON only:
```json
{{
  "objects": [...],
  "constraints": {{
    "axial": [...],
    "topology": [...],
    "occlusion": [...],
    "size": [...],
    "closer": [...],
    "qrr": [...],
    "trr": [...]
  }},
  "confidence": 0.0-1.0
}}
```"""

MULTI_VIEW_USER_TEMPLATE = """Analyze these {n_views} images of the same scene from different viewpoints.

{object_context}

Parameters:
- Tolerance (tau): {tau}

For each constraint, determine if it's view-invariant (3D) or view-dependent.

Return JSON with the same schema as single-view, plus metadata about view consistency."""


# =============================================================================
# Prompt Building Functions
# =============================================================================

def get_system_prompt(mode: str = "single", tau: float = 0.10) -> str:
    """
    获取系统提示词。

    Get system prompt for the specified mode.

    Args:
        mode: "single" or "multi"
        tau: Tolerance parameter

    Returns:
        System prompt string
    """
    if mode == "multi":
        return SYSTEM_PROMPT_MULTI_VIEW.format(tau=tau)
    return SYSTEM_PROMPT_SINGLE_VIEW.format(tau=tau)


def get_output_schema() -> Dict[str, Any]:
    """
    获取输出 JSON Schema。

    Get output JSON schema for validation.
    """
    return OUTPUT_SCHEMA


def format_object_context(objects: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    格式化已知物体信息。

    Format known object information for the prompt.

    Args:
        objects: Optional list of known objects

    Returns:
        Formatted string describing objects
    """
    if not objects:
        return "Objects: Not specified (identify all visible objects)"

    lines = ["Known objects in scene:"]
    for obj in objects:
        obj_id = obj.get("id", "unknown")
        obj_type = obj.get("shape", obj.get("type", "unknown"))
        color = obj.get("color", "unknown")
        size = obj.get("size", "medium")
        lines.append(f"  - {obj_id}: {color} {obj_type} ({size})")

    return "\n".join(lines)


def build_single_view_prompt(
    objects: Optional[List[Dict[str, Any]]] = None,
    tau: float = 0.10,
) -> Dict[str, str]:
    """
    构建单视角提取提示词。

    Build prompt for single-view constraint extraction.

    Args:
        objects: Optional list of known objects
        tau: Tolerance parameter

    Returns:
        Dict with "system" and "user" prompts
    """
    object_context = format_object_context(objects)

    user_prompt = SINGLE_VIEW_USER_TEMPLATE.format(
        object_context=object_context,
        tau=tau,
    )

    return {
        "system": get_system_prompt("single", tau),
        "user": user_prompt,
    }


def build_multi_view_prompt(
    n_views: int,
    objects: Optional[List[Dict[str, Any]]] = None,
    tau: float = 0.10,
) -> Dict[str, str]:
    """
    构建多视角提取提示词。

    Build prompt for multi-view constraint extraction.

    Args:
        n_views: Number of views/images
        objects: Optional list of known objects
        tau: Tolerance parameter

    Returns:
        Dict with "system" and "user" prompts
    """
    object_context = format_object_context(objects)

    user_prompt = MULTI_VIEW_USER_TEMPLATE.format(
        n_views=n_views,
        object_context=object_context,
        tau=tau,
    )

    return {
        "system": get_system_prompt("multi", tau),
        "user": user_prompt,
    }
