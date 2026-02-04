"""
FDA Database - Search Drugs Tool Description
"""

tool_description = {
    "name": "fda_search_drugs",
    "description": (
        "Search the FDA drug database for approved drugs by brand name, generic name, NDC code, or active ingredient. "
        "Uses the openFDA API to retrieve comprehensive drug information including approval dates, labeling, manufacturer details, and regulatory status. "
        "Highly reliable data source directly from FDA; preferred for regulatory compliance and drug information queries. "
        "Returns structured JSON with complete drug profiles suitable for downstream analysis."
    ),
    "required_parameters": [
        {
            "name": "search_term",
            "type": "str",
            "default": None,
            "description": (
                "The search query for finding drugs. Can be a brand name (e.g., 'Lipitor'), "
                "generic name (e.g., 'atorvastatin'), NDC code, or active ingredient. "
                "Partial matches are supported. Case-insensitive."
            ),
        },
    ],
    "optional_parameters": [
        {
            "name": "search_field",
            "type": "str",
            "default": "brand_name",
            "description": (
                "Specifies which field to search in. Options: 'brand_name', 'generic_name', "
                "'active_ingredient', 'ndc', 'application_number'. "
                "Defaults to brand_name for most common use case."
            ),
        },
        {
            "name": "limit",
            "type": "int",
            "default": 100,
            "description": (
                "Maximum number of results to return. Range: 1-1000. "
                "Higher limits increase response time proportionally. "
                "Recommended: 100 for interactive queries, 1000 for batch processing."
            ),
        },
        {
            "name": "skip",
            "type": "int",
            "default": 0,
            "description": (
                "Number of results to skip for pagination. "
                "Use in combination with limit to retrieve large result sets. "
                "Example: skip=100, limit=100 retrieves results 101-200."
            ),
        },
        {
            "name": "api_key",
            "type": "str",
            "default": None,
            "description": (
                "Optional FDA API key for higher rate limits. "
                "Without key: 240 requests/minute, 1000/hour. "
                "With key: 240 requests/minute, unlimited hourly. "
                "Obtain from: https://open.fda.gov/apis/authentication/"
            ),
        },
    ],
    "hardware_requirements": {
        "device": "cpu_only",
        "notes": (
            "No GPU required. Network-based API call. "
            "Requires stable internet connection. "
            "Minimal CPU and memory usage (<10MB RAM per request). "
            "Performance depends on network latency and FDA API availability."
        ),
    },
    "time_complexity": {
        "assumptions": (
            "Wall-clock latency measured with stable internet connection (100 Mbps). "
            "CPU: Apple M1/M2 or equivalent. "
            "Includes API call, JSON parsing, and response formatting. "
            "Cold start includes initial HTTP connection setup (~200ms additional). "
            "Assumes FDA API is responsive (typical: 200-500ms base latency)."
        ),
        "latency_seconds": {
            "n1": 0.5,  # Single query
            "n2": 1.0,  # Two sequential queries
            "n10": 5.0,  # Ten sequential queries
            "batch_10": 2.0,  # If parallelized with asyncio
        },
    },
    "outputs": {
        "type": "dict",
        "schema": {
            "results": "list[dict]",
            "total_count": "int",
            "fields": [
                "brand_name",
                "generic_name",
                "manufacturer_name",
                "product_ndc",
                "application_number",
                "approval_date",
                "marketing_status",
                "dosage_form",
                "route",
                "active_ingredients",
            ],
        },
        "example": {
            "results": [
                {
                    "brand_name": "LIPITOR",
                    "generic_name": "atorvastatin calcium",
                    "manufacturer_name": "Pfizer Inc",
                    "product_ndc": "0071-0155",
                    "application_number": "NDA020702",
                    "approval_date": "1996-12-17",
                    "marketing_status": "Prescription",
                    "dosage_form": "TABLET",
                    "route": "ORAL",
                }
            ],
            "total_count": 1,
        },
    },
    "failure_modes": [
        {
            "error": "HTTPError 404",
            "cause": "No results found for search term or invalid endpoint",
            "fix": "Verify search term spelling, try broader search, or check API endpoint availability",
        },
        {
            "error": "RateLimitError 429",
            "cause": "Exceeded FDA API rate limits (240/min or 1000/hour without API key)",
            "fix": "Implement exponential backoff, add API key, or reduce request frequency",
        },
        {
            "error": "TimeoutError",
            "cause": "API response took longer than timeout threshold (typically 30s)",
            "fix": "Check internet connection, reduce limit parameter, or retry with backoff",
        },
        {
            "error": "JSONDecodeError",
            "cause": "Invalid or malformed response from FDA API",
            "fix": "Log raw response, report to FDA if persistent, implement retry logic",
        },
        {
            "error": "NetworkError",
            "cause": "No internet connection or DNS resolution failure",
            "fix": "Verify network connectivity, check firewall settings, confirm DNS resolution",
        },
    ],
}
