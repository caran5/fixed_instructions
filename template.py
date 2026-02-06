"""
Tool Description Template for Function Calling

This template defines the standard structure for tool descriptions
that can be used with AI agents and function calling APIs.
"""

TOOL_TEMPLATE = {
    "name": "<tool_name_snake_case>",
    
    "description": """
        <2–4 sentence high-level explanation of what the tool does.
        Clearly state input → processing → output.
        Mention underlying library/CLI/model and key assumptions
        (e.g., strand scanning, circular DNA, etc.).>
    """,
    
    "required_parameters": [
        {
            "name": "<parameter_name>",
            "type": "str | int | bool | float | list | dict",
            "description": """
                <What the parameter represents.
                Include expected format (e.g., raw string vs FASTA),
                units (nt vs aa),
                allowed ranges,
                and validation constraints.>
            """
        }
    ],
    
    "optional_parameters": [
        {
            "name": "<parameter_name>",
            "type": "str | int | bool | float | list | dict",
            "default": "<actual_default_value>",
            "description": """
                <What changes when enabled/disabled.
                Mention performance, memory, or runtime impact if applicable.>
            """
        }
    ],
    
    "hardware_requirements": {
        "device": "cpu_only | gpu_optional | gpu_required",
        "notes": """
            <Any additional requirements such as RAM needs,
            external binaries, PATH setup, or storage considerations.>
        """
    },
    
    "time_complexity": {
        "assumptions": """
            <Describe benchmarking conditions:
            hardware used, CPU/GPU model,
            input size,
            number of threads,
            cold-start vs warm-start behavior.>
        """,
        "latency_seconds": {
            "n1": None,   # 1 run
            "n2": None,   # 2 runs
            "n10": None   # 10 runs
        }
    },
    
    "outputs": {
        "type": "list | dict | dataframe | json | file_path",
        "schema": """
            <Describe fields, keys, and structure of the output.>
        """,
        "format": """
            <Units, encoding, coordinate system, or file format.>
        """,
        "example": """
            <Minimal example output.>
        """
    },
    
    "failure_modes": [
        {
            "error": "<error_condition>",
            "cause": "<likely_cause>",
            "fix": "<recommended_fix>"
        }
    ],
    
    "dependencies": {
        "requirements_file": "requirements_<tool_name>.txt",
        "description": """
            <List external libraries, binaries, or models.
            Include versions, installation method,
            and whether each is a system binary or Python package.>
        """
    }
}


def get_openai_function_schema(tool: dict) -> dict:
    """
    Convert a tool description to OpenAI function calling format.
    
    Args:
        tool: Tool description dictionary
        
    Returns:
        OpenAI-compatible function schema
    """
    properties = {}
    required = []
    
    for param in tool.get("required_parameters", []):
        properties[param["name"]] = {
            "type": _convert_type(param["type"]),
            "description": param["description"].strip()
        }
        required.append(param["name"])
    
    for param in tool.get("optional_parameters", []):
        properties[param["name"]] = {
            "type": _convert_type(param["type"]),
            "description": param["description"].strip()
        }
        if param.get("default") is not None:
            properties[param["name"]]["default"] = param["default"]
    
    return {
        "name": tool["name"],
        "description": tool["description"].strip(),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def get_anthropic_tool_schema(tool: dict) -> dict:
    """
    Convert a tool description to Anthropic tool use format.
    
    Args:
        tool: Tool description dictionary
        
    Returns:
        Anthropic-compatible tool schema
    """
    properties = {}
    required = []
    
    for param in tool.get("required_parameters", []):
        properties[param["name"]] = {
            "type": _convert_type(param["type"]),
            "description": param["description"].strip()
        }
        required.append(param["name"])
    
    for param in tool.get("optional_parameters", []):
        properties[param["name"]] = {
            "type": _convert_type(param["type"]),
            "description": param["description"].strip()
        }
    
    return {
        "name": tool["name"],
        "description": tool["description"].strip(),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def _convert_type(type_str: str) -> str:
    """Convert Python type hints to JSON Schema types."""
    type_map = {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object"
    }
    return type_map.get(type_str.lower().split()[0], "string")


if __name__ == "__main__":
    # Example usage
    import json
    
    example_tool = {
        "name": "example_tool",
        "description": "An example tool that demonstrates the template structure.",
        "required_parameters": [
            {
                "name": "input_data",
                "type": "string",
                "description": "The input data to process"
            }
        ],
        "optional_parameters": [
            {
                "name": "verbose",
                "type": "boolean",
                "default": False,
                "description": "Enable verbose output"
            }
        ]
    }
    
    print("OpenAI Format:")
    print(json.dumps(get_openai_function_schema(example_tool), indent=2))
    
    print("\nAnthropic Format:")
    print(json.dumps(get_anthropic_tool_schema(example_tool), indent=2))
