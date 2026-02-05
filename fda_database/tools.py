"""
FDA Database Tool Descriptions for Function Calling

Comprehensive tool definitions for querying the openFDA API.
Covers drugs, devices, foods, animal/veterinary, and substances.
"""

# =============================================================================
# DRUG ENDPOINTS (6)
# =============================================================================

fda_query_drug_events = {
    "name": "fda_query_drug_events",
    "description": """Query drug adverse events from the FDA Adverse Event Reporting System (FAERS).
        Access reports of drug side effects, product use errors, product quality problems,
        and therapeutic failures submitted to the FDA. Useful for safety signal detection,
        post-market surveillance, drug interaction analysis, and comparative safety research.
        API Endpoint: https://api.fda.gov/drug/event.json""",
    
    "required_parameters": [
        {
            "name": "drug_name",
            "type": "string",
            "description": "Name of the drug to search for adverse events (can include wildcards)"
        }
    ],
    
    "optional_parameters": [
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return (1-1000)"},
        {"name": "skip", "type": "integer", "default": 0, "description": "Number of results to skip for pagination"},
        {"name": "serious_only", "type": "boolean", "default": False, "description": "Filter for only serious adverse events"},
        {"name": "count_field", "type": "string", "default": None, "description": "Field to count/aggregate by (e.g., 'patient.reaction.reactionmeddrapt.exact')"},
        {"name": "sort", "type": "string", "default": None, "description": "Field to sort by (e.g., 'receivedate:desc')"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits (optional but recommended)"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of adverse event reports matching the search criteria"},
        {"name": "meta", "type": "dict", "description": "Metadata including total results count, pagination info"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid drug name", "behavior": "Returns empty results array"},
        {"condition": "API rate limit exceeded", "behavior": "HTTP 429 error - use API key or reduce request frequency"},
        {"condition": "Network timeout", "behavior": "Request timeout after 30 seconds"}
    ]
}

fda_query_drug_label = {
    "name": "fda_query_drug_label",
    "description": """Query drug product labeling (Structured Product Labeling - SPL) from FDA.
        Access prescribing information, warnings, indications, usage instructions,
        and comprehensive drug information for FDA-approved and marketed products.
        API Endpoint: https://api.fda.gov/drug/label.json""",
    
    "required_parameters": [
        {"name": "drug_name", "type": "string", "description": "Name of the drug to search for labeling information"}
    ],
    
    "optional_parameters": [
        {"name": "brand", "type": "boolean", "default": True, "description": "Search by brand name (True) or generic name (False)"},
        {"name": "limit", "type": "integer", "default": 1, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of drug labels matching the search criteria"},
        {"name": "indications_and_usage", "type": "string", "description": "Approved indications and usage instructions"},
        {"name": "warnings", "type": "string", "description": "Important safety warnings"},
        {"name": "adverse_reactions", "type": "string", "description": "Known adverse reactions"}
    ],
    
    "failure_modes": [
        {"condition": "Drug not found", "behavior": "Returns empty results array"},
        {"condition": "Multiple brands with same name", "behavior": "Returns multiple results - filter by manufacturer"}
    ]
}

fda_query_drug_ndc = {
    "name": "fda_query_drug_ndc",
    "description": """Query the National Drug Code (NDC) Directory from FDA.
        Access drug product information identified by National Drug Codes including
        manufacturer, dosage form, route of administration, active ingredients.
        API Endpoint: https://api.fda.gov/drug/ndc.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "ndc", "type": "string", "default": None, "description": "National Drug Code (10-digit identifier)"},
        {"name": "manufacturer", "type": "string", "default": None, "description": "Manufacturer/labeler name to search"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of NDC records matching the search"},
        {"name": "product_ndc", "type": "string", "description": "10-digit NDC identifier"},
        {"name": "active_ingredients", "type": "list", "description": "List of active ingredients with strengths"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid NDC format", "behavior": "Returns empty results"},
        {"condition": "Neither ndc nor manufacturer provided", "behavior": "Raises ValueError"}
    ]
}

fda_query_drug_recalls = {
    "name": "fda_query_drug_recalls",
    "description": """Query drug recall enforcement reports from FDA.
        Access information about drug recalls including classification (I, II, III),
        reason for recall, distribution patterns, and status.
        API Endpoint: https://api.fda.gov/drug/enforcement.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "drug_name", "type": "string", "default": None, "description": "Name of drug to search for recalls"},
        {"name": "classification", "type": "string", "default": None, "description": "Recall class to filter by (I, II, or III)"},
        {"name": "status", "type": "string", "default": None, "description": "Recall status filter (Ongoing, Completed, Terminated)"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "sort", "type": "string", "default": "report_date:desc", "description": "Sort field"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of recall enforcement reports"},
        {"name": "classification", "type": "string", "description": "Recall severity classification"},
        {"name": "reason_for_recall", "type": "string", "description": "Detailed reason for the recall"}
    ],
    
    "failure_modes": [
        {"condition": "No matching recalls", "behavior": "Returns empty results array"},
        {"condition": "Invalid classification value", "behavior": "No results returned"}
    ]
}

fda_query_drugs_at_fda = {
    "name": "fda_query_drugs_at_fda",
    "description": """Query the Drugs@FDA database for comprehensive FDA-approved drug information.
        Access historical approval data, regulatory submissions, and approval history
        for drugs approved since 1939.
        API Endpoint: https://api.fda.gov/drug/drugsfda.json""",
    
    "required_parameters": [
        {"name": "search_query", "type": "string", "description": "Search query (e.g., brand name, sponsor name, application number)"}
    ],
    
    "optional_parameters": [
        {"name": "search_field", "type": "string", "default": "openfda.brand_name", "description": "Field to search in"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of drug application records"},
        {"name": "application_number", "type": "string", "description": "FDA application number (NDA/ANDA/BLA)"},
        {"name": "submissions", "type": "list", "description": "History of regulatory submissions"}
    ],
    
    "failure_modes": [
        {"condition": "Drug not in database", "behavior": "Returns empty results"},
        {"condition": "Invalid application number format", "behavior": "No results returned"}
    ]
}

fda_query_drug_shortages = {
    "name": "fda_query_drug_shortages",
    "description": """Query the FDA Drug Shortages database for current and resolved supply issues.
        Access information about drugs in shortage, reasons, resolution dates, and alternatives.
        API Endpoint: https://api.fda.gov/drug/drugshortages.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "product_name", "type": "string", "default": None, "description": "Name of drug to search for shortage status"},
        {"name": "status", "type": "string", "default": None, "description": "Filter by status (Currently in Shortage, Resolved, Discontinued)"},
        {"name": "active_ingredient", "type": "string", "default": None, "description": "Search by active ingredient"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of drug shortage records"},
        {"name": "status", "type": "string", "description": "Current shortage status"},
        {"name": "reason", "type": "string", "description": "Reason for the shortage"}
    ],
    
    "failure_modes": [
        {"condition": "No matching shortages", "behavior": "Returns empty results"},
        {"condition": "Drug not tracked", "behavior": "No results returned"}
    ]
}

# =============================================================================
# DEVICE ENDPOINTS (9)
# =============================================================================

fda_query_device_events = {
    "name": "fda_query_device_events",
    "description": """Query medical device adverse events from the MAUDE database.
        Access reports of device malfunctions, injuries, deaths, and other undesirable effects.
        API Endpoint: https://api.fda.gov/device/event.json""",
    
    "required_parameters": [
        {"name": "device_name", "type": "string", "description": "Name of device to search for adverse events"}
    ],
    
    "optional_parameters": [
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return (1-1000)"},
        {"name": "event_type", "type": "string", "default": None, "description": "Filter by event type (Death, Injury, Malfunction, Other)"},
        {"name": "device_class", "type": "integer", "default": None, "description": "Filter by device class (1, 2, or 3)"},
        {"name": "count_field", "type": "string", "default": None, "description": "Field to count/aggregate by"},
        {"name": "sort", "type": "string", "default": None, "description": "Sort field (e.g., 'date_received:desc')"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of device adverse event reports"},
        {"name": "event_type", "type": "string", "description": "Type of adverse event (Death, Injury, Malfunction)"},
        {"name": "device", "type": "dict", "description": "Device information including brand, manufacturer, class"}
    ],
    
    "failure_modes": [
        {"condition": "Device not found", "behavior": "Returns empty results array"},
        {"condition": "API rate limit exceeded", "behavior": "HTTP 429 error"}
    ]
}

fda_query_device_510k = {
    "name": "fda_query_device_510k",
    "description": """Query 510(k) premarket notification clearances from FDA.
        Access information about medical devices cleared through the 510(k) pathway.
        API Endpoint: https://api.fda.gov/device/510k.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "applicant", "type": "string", "default": None, "description": "Company name to search for 510(k) clearances"},
        {"name": "device_name", "type": "string", "default": None, "description": "Device name to search for"},
        {"name": "product_code", "type": "string", "default": None, "description": "FDA product code"},
        {"name": "k_number", "type": "string", "default": None, "description": "Specific 510(k) number to look up"},
        {"name": "device_class", "type": "integer", "default": None, "description": "Filter by device class (1, 2, or 3)"},
        {"name": "decision_date_start", "type": "string", "default": None, "description": "Start date for decision date range (YYYYMMDD)"},
        {"name": "decision_date_end", "type": "string", "default": None, "description": "End date for decision date range (YYYYMMDD)"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of 510(k) clearance records"},
        {"name": "k_number", "type": "string", "description": "510(k) clearance number"},
        {"name": "decision_description", "type": "string", "description": "FDA clearance decision"}
    ],
    
    "failure_modes": [
        {"condition": "No matching clearances", "behavior": "Returns empty results array"},
        {"condition": "Invalid k_number format", "behavior": "No results returned"}
    ]
}

fda_query_device_classification = {
    "name": "fda_query_device_classification",
    "description": """Query the FDA device classification database.
        Access medical device categories, risk classifications, and regulatory requirements.
        API Endpoint: https://api.fda.gov/device/classification.json""",
    
    "required_parameters": [
        {"name": "product_code", "type": "string", "description": "Three-letter FDA product code (e.g., 'DQY', 'LWS')"}
    ],
    
    "optional_parameters": [
        {"name": "device_name", "type": "string", "default": None, "description": "Search by device name instead of product code"},
        {"name": "device_class", "type": "integer", "default": None, "description": "Filter by device class (1, 2, or 3)"},
        {"name": "medical_specialty", "type": "string", "default": None, "description": "Filter by medical specialty"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of device classification records"},
        {"name": "device_class", "type": "integer", "description": "Risk classification level"},
        {"name": "regulation_number", "type": "string", "description": "CFR regulation reference"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid product code", "behavior": "Returns empty results"},
        {"condition": "Product code not found", "behavior": "No results returned"}
    ]
}

fda_query_device_enforcement = {
    "name": "fda_query_device_enforcement",
    "description": """Query device recall enforcement reports from FDA.
        Access information about device recalls including classification, reason, and status.
        API Endpoint: https://api.fda.gov/device/enforcement.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "device_name", "type": "string", "default": None, "description": "Device name or description to search"},
        {"name": "classification", "type": "string", "default": None, "description": "Recall class to filter (I, II, or III)"},
        {"name": "recalling_firm", "type": "string", "default": None, "description": "Company name to filter recalls"},
        {"name": "status", "type": "string", "default": None, "description": "Recall status filter (Ongoing, Completed)"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "sort", "type": "string", "default": "report_date:desc", "description": "Sort field"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of device recall enforcement reports"},
        {"name": "classification", "type": "string", "description": "Recall severity classification"},
        {"name": "reason_for_recall", "type": "string", "description": "Detailed reason for the recall"}
    ],
    
    "failure_modes": [
        {"condition": "No matching recalls", "behavior": "Returns empty results"},
        {"condition": "Invalid classification", "behavior": "No results returned"}
    ]
}

fda_query_device_recall = {
    "name": "fda_query_device_recall",
    "description": """Query detailed device recall information from FDA.
        Access comprehensive recall data including root cause analysis and associated submissions.
        API Endpoint: https://api.fda.gov/device/recall.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "product_code", "type": "string", "default": None, "description": "FDA product code to search"},
        {"name": "device_name", "type": "string", "default": None, "description": "Device name to search"},
        {"name": "k_number", "type": "string", "default": None, "description": "Associated 510(k) number"},
        {"name": "pma_number", "type": "string", "default": None, "description": "Associated PMA number"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of detailed recall records"},
        {"name": "root_cause_description", "type": "string", "description": "Root cause analysis of the recall"},
        {"name": "k_numbers", "type": "list", "description": "Associated 510(k) clearance numbers"}
    ],
    
    "failure_modes": [
        {"condition": "No matching recalls", "behavior": "Returns empty results"},
        {"condition": "Invalid product code", "behavior": "No results returned"}
    ]
}

fda_query_device_pma = {
    "name": "fda_query_device_pma",
    "description": """Query Premarket Approval (PMA) data from FDA for Class III medical devices.
        API Endpoint: https://api.fda.gov/device/pma.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "applicant", "type": "string", "default": None, "description": "Company name to search"},
        {"name": "pma_number", "type": "string", "default": None, "description": "Specific PMA number to look up"},
        {"name": "device_name", "type": "string", "default": None, "description": "Device name to search"},
        {"name": "product_code", "type": "string", "default": None, "description": "FDA product code"},
        {"name": "advisory_committee", "type": "string", "default": None, "description": "FDA advisory committee"},
        {"name": "decision_date_start", "type": "string", "default": None, "description": "Start date for decision date range (YYYYMMDD)"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of PMA approval records"},
        {"name": "pma_number", "type": "string", "description": "PMA approval number"},
        {"name": "decision_code", "type": "string", "description": "FDA decision code (APPR, etc.)"}
    ],
    
    "failure_modes": [
        {"condition": "No matching approvals", "behavior": "Returns empty results"},
        {"condition": "Invalid PMA number format", "behavior": "No results returned"}
    ]
}

fda_query_device_registration = {
    "name": "fda_query_device_registration",
    "description": """Query device registrations and listings from FDA.
        Access information about manufacturing facilities and their listed devices.
        API Endpoint: https://api.fda.gov/device/registrationlisting.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "registration_number", "type": "string", "default": None, "description": "Facility registration number"},
        {"name": "fei_number", "type": "string", "default": None, "description": "Facility Establishment Identifier"},
        {"name": "facility_name", "type": "string", "default": None, "description": "Name of manufacturing facility"},
        {"name": "product_code", "type": "string", "default": None, "description": "FDA product code"},
        {"name": "state_code", "type": "string", "default": None, "description": "US state code (e.g., 'CA', 'NY')"},
        {"name": "country_code", "type": "string", "default": None, "description": "ISO country code"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of registration/listing records"},
        {"name": "registration", "type": "dict", "description": "Facility registration information"},
        {"name": "products", "type": "list", "description": "Listed products at the facility"}
    ],
    
    "failure_modes": [
        {"condition": "Facility not found", "behavior": "Returns empty results"},
        {"condition": "Invalid registration number", "behavior": "No results returned"}
    ]
}

fda_query_device_udi = {
    "name": "fda_query_device_udi",
    "description": """Query the Unique Device Identification (UDI) database from FDA.
        Access device identification information including UDI, brand names, and characteristics.
        API Endpoint: https://api.fda.gov/device/udi.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "udi", "type": "string", "default": None, "description": "Unique Device Identifier to look up"},
        {"name": "brand_name", "type": "string", "default": None, "description": "Brand name to search"},
        {"name": "company_name", "type": "string", "default": None, "description": "Manufacturer/company name"},
        {"name": "product_code", "type": "string", "default": None, "description": "FDA product code"},
        {"name": "is_rx", "type": "boolean", "default": None, "description": "Filter for prescription devices"},
        {"name": "mri_safety", "type": "string", "default": None, "description": "MRI safety classification (MR Safe, MR Conditional, MR Unsafe)"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of UDI records"},
        {"name": "identifiers", "type": "list", "description": "Device identifiers (GTIN, HIBCC, etc.)"},
        {"name": "mri_safety", "type": "string", "description": "MRI safety classification"}
    ],
    
    "failure_modes": [
        {"condition": "UDI not found", "behavior": "Returns empty results"},
        {"condition": "Invalid UDI format", "behavior": "No results returned"}
    ]
}

fda_query_device_covid19_serology = {
    "name": "fda_query_device_covid19_serology",
    "description": """Query COVID-19 serological (antibody) test performance data from FDA.
        Access evaluation data for COVID-19 antibody tests including sensitivity and specificity.
        API Endpoint: https://api.fda.gov/device/covid19serology.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "manufacturer", "type": "string", "default": None, "description": "Test manufacturer name"},
        {"name": "device_name", "type": "string", "default": None, "description": "Test device name"},
        {"name": "test_method", "type": "string", "default": None, "description": "Testing methodology (ELISA, LFA, CLIA, etc.)"},
        {"name": "min_sensitivity", "type": "number", "default": None, "description": "Minimum sensitivity threshold"},
        {"name": "min_specificity", "type": "number", "default": None, "description": "Minimum specificity threshold"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of COVID-19 serology test records"},
        {"name": "sensitivity", "type": "number", "description": "Test sensitivity percentage"},
        {"name": "specificity", "type": "number", "description": "Test specificity percentage"}
    ],
    
    "failure_modes": [
        {"condition": "No matching tests", "behavior": "Returns empty results"},
        {"condition": "Invalid parameter value", "behavior": "No results returned"}
    ]
}

# =============================================================================
# FOOD ENDPOINTS (2)
# =============================================================================

fda_query_food_events = {
    "name": "fda_query_food_events",
    "description": """Query food adverse events from FDA's CFSAN.
        Access reports of adverse events related to foods, dietary supplements, and cosmetics.
        API Endpoint: https://api.fda.gov/food/event.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "product_name", "type": "string", "default": None, "description": "Product brand name to search"},
        {"name": "industry", "type": "string", "default": None, "description": "Industry category (e.g., 'Dietary Supplements', 'Beverages')"},
        {"name": "outcome", "type": "string", "default": None, "description": "Filter by outcome (Hospitalization, Death, ER Visit)"},
        {"name": "reaction", "type": "string", "default": None, "description": "Filter by reaction type"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "sort", "type": "string", "default": None, "description": "Sort field (e.g., 'date_created:desc')"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of food adverse event reports"},
        {"name": "outcomes", "type": "list", "description": "Array of outcomes (Hospitalization, Death, etc.)"},
        {"name": "reactions", "type": "list", "description": "Array of adverse reactions"}
    ],
    
    "failure_modes": [
        {"condition": "No matching events", "behavior": "Returns empty results"},
        {"condition": "Invalid industry name", "behavior": "No results returned"}
    ]
}

fda_query_food_recalls = {
    "name": "fda_query_food_recalls",
    "description": """Query food recall enforcement reports from FDA.
        Access information about food recalls including allergen and pathogen contamination.
        API Endpoint: https://api.fda.gov/food/enforcement.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "product", "type": "string", "default": None, "description": "Product name or description to search"},
        {"name": "reason", "type": "string", "default": None, "description": "Reason for recall (e.g., 'undeclared peanut', 'listeria')"},
        {"name": "classification", "type": "string", "default": None, "description": "Recall class to filter (I, II, or III)"},
        {"name": "status", "type": "string", "default": None, "description": "Recall status filter (Ongoing, Completed)"},
        {"name": "recalling_firm", "type": "string", "default": None, "description": "Company name to filter"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "sort", "type": "string", "default": "report_date:desc", "description": "Sort field"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of food recall enforcement reports"},
        {"name": "classification", "type": "string", "description": "Recall severity classification"},
        {"name": "reason_for_recall", "type": "string", "description": "Detailed reason for the recall"}
    ],
    
    "failure_modes": [
        {"condition": "No matching recalls", "behavior": "Returns empty results"},
        {"condition": "Invalid classification", "behavior": "No results returned"}
    ]
}

# =============================================================================
# ANIMAL & VETERINARY (1)
# =============================================================================

fda_query_animal_events = {
    "name": "fda_query_animal_events",
    "description": """Query animal drug adverse events from FDA's Center for Veterinary Medicine (CVM).
        Access reports of side effects and therapeutic failures associated with animal drugs.
        API Endpoint: https://api.fda.gov/animalandveterinary/event.json""",
    
    "required_parameters": [],
    
    "optional_parameters": [
        {"name": "species", "type": "string", "default": None, "description": "Animal species (Dog, Cat, Horse, Cattle, etc.)"},
        {"name": "drug_name", "type": "string", "default": None, "description": "Drug brand name or active ingredient"},
        {"name": "breed", "type": "string", "default": None, "description": "Animal breed"},
        {"name": "reaction", "type": "string", "default": None, "description": "Reaction term (VeDDRA term)"},
        {"name": "serious_only", "type": "boolean", "default": False, "description": "Filter for only serious adverse events"},
        {"name": "outcome", "type": "string", "default": None, "description": "Medical outcome (Died, Recovered, Unknown)"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of results to return"},
        {"name": "sort", "type": "string", "default": None, "description": "Sort field (e.g., 'onset_date:desc')"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of animal drug adverse event reports"},
        {"name": "animal", "type": "dict", "description": "Animal information (species, breed, age, weight)"},
        {"name": "reaction", "type": "list", "description": "Array of adverse reactions (VeDDRA terms)"}
    ],
    
    "failure_modes": [
        {"condition": "No matching events", "behavior": "Returns empty results"},
        {"condition": "Invalid species name", "behavior": "No results returned"}
    ]
}

# =============================================================================
# SUBSTANCE ENDPOINTS (2)
# =============================================================================

fda_query_substance_by_unii = {
    "name": "fda_query_substance_by_unii",
    "description": """Query FDA substance data by UNII (Unique Ingredient Identifier) code.
        Access chemical and biological substance information from FDA's GSRS.
        API Endpoint: https://api.fda.gov/other/substance.json""",
    
    "required_parameters": [
        {"name": "unii", "type": "string", "description": "UNII (Unique Ingredient Identifier) code"}
    ],
    
    "optional_parameters": [
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of substance records"},
        {"name": "approvalID", "type": "string", "description": "UNII code"},
        {"name": "names", "type": "list", "description": "Array of substance names"},
        {"name": "molecularFormula", "type": "string", "description": "Chemical molecular formula"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid UNII code", "behavior": "Returns empty results"},
        {"condition": "UNII not found", "behavior": "No results returned"}
    ]
}

fda_query_substance_by_name = {
    "name": "fda_query_substance_by_name",
    "description": """Query FDA substance data by substance name.
        Search for substances to find UNII codes, CAS numbers, and other identifiers.
        API Endpoint: https://api.fda.gov/other/substance.json""",
    
    "required_parameters": [
        {"name": "name", "type": "string", "description": "Substance name to search for"}
    ],
    
    "optional_parameters": [
        {"name": "exact_match", "type": "boolean", "default": False, "description": "Require exact name match"},
        {"name": "limit", "type": "integer", "default": 10, "description": "Maximum number of results to return"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of substance records matching search"},
        {"name": "approvalID", "type": "string", "description": "UNII code"},
        {"name": "codes", "type": "list", "description": "Array of identifier codes (CAS, etc.)"}
    ],
    
    "failure_modes": [
        {"condition": "Substance not found", "behavior": "Returns empty results"},
        {"condition": "Ambiguous name", "behavior": "Returns multiple results - may need refinement"}
    ]
}

# =============================================================================
# UTILITY FUNCTIONS (4)
# =============================================================================

fda_generic_query = {
    "name": "fda_generic_query",
    "description": """Generic query method for any FDA openFDA API endpoint.
        Provides flexible access to all FDA data categories and endpoints with full control
        over search parameters, pagination, sorting, and aggregation.""",
    
    "required_parameters": [
        {"name": "category", "type": "string", "description": "API category (drug, device, food, animalandveterinary, other)"},
        {"name": "endpoint", "type": "string", "description": "Specific endpoint within the category"}
    ],
    
    "optional_parameters": [
        {"name": "search", "type": "string", "default": None, "description": "Search query string (openFDA query syntax)"},
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum results to return (1-1000)"},
        {"name": "skip", "type": "integer", "default": 0, "description": "Number of results to skip (pagination)"},
        {"name": "count", "type": "string", "default": None, "description": "Field to count/aggregate by"},
        {"name": "sort", "type": "string", "default": None, "description": "Sort field (e.g., 'receivedate:desc')"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 10 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of matching records"},
        {"name": "meta", "type": "dict", "description": "Metadata including total count, pagination info"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid category/endpoint", "behavior": "HTTP 404 error"},
        {"condition": "Invalid search syntax", "behavior": "HTTP 400 error with message"},
        {"condition": "Rate limit exceeded", "behavior": "HTTP 429 error"}
    ]
}

fda_count_by_field = {
    "name": "fda_count_by_field",
    "description": """Count/aggregate FDA data by a specific field.
        Perform aggregation queries to get counts of records grouped by a field value.
        Essential for statistical analysis and trend identification.""",
    
    "required_parameters": [
        {"name": "category", "type": "string", "description": "API category (drug, device, food, animalandveterinary, other)"},
        {"name": "endpoint", "type": "string", "description": "Specific endpoint (event, label, enforcement, etc.)"},
        {"name": "search", "type": "string", "description": "Search query to filter records before counting"},
        {"name": "field", "type": "string", "description": "Field to count/aggregate by (use .exact suffix for exact matching)"}
    ],
    
    "optional_parameters": [
        {"name": "exact", "type": "boolean", "default": True, "description": "Whether to use exact field matching (appends .exact)"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 1GB)"],
    "time_complexity": "O(1) - API response typically < 5 seconds",
    
    "outputs": [
        {"name": "results", "type": "list", "description": "Array of term/count pairs"},
        {"name": "term", "type": "string", "description": "Field value"},
        {"name": "count", "type": "integer", "description": "Number of records with this value"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid field name", "behavior": "Returns empty results or error"},
        {"condition": "No matching records", "behavior": "Returns empty results array"}
    ]
}

fda_drug_safety_profile = {
    "name": "fda_drug_safety_profile",
    "description": """Generate a comprehensive drug safety profile by combining multiple FDA data sources.
        Aggregates adverse events, serious events, recalls, and reaction statistics for a drug.""",
    
    "required_parameters": [
        {"name": "drug_name", "type": "string", "description": "Name of the drug to analyze"}
    ],
    
    "optional_parameters": [
        {"name": "limit", "type": "integer", "default": 100, "description": "Maximum number of individual records to retrieve"},
        {"name": "top_reactions", "type": "integer", "default": 10, "description": "Number of top reactions to include"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 2GB)"],
    "time_complexity": "O(1) - Multiple API calls, typically < 15 seconds total",
    
    "outputs": [
        {"name": "total_events", "type": "integer", "description": "Total number of adverse event reports"},
        {"name": "serious_events", "type": "integer", "description": "Number of serious adverse event reports"},
        {"name": "top_reactions", "type": "list", "description": "Array of most common adverse reactions with counts"},
        {"name": "recalls", "type": "list", "description": "Array of recall records for the drug"},
        {"name": "death_events", "type": "integer", "description": "Number of events resulting in death (if available)"}
    ],
    
    "failure_modes": [
        {"condition": "Drug not found", "behavior": "Returns zeros for counts, empty arrays"},
        {"condition": "One API call fails", "behavior": "Partial results returned with error indication"}
    ]
}

fda_device_comprehensive_lookup = {
    "name": "fda_device_comprehensive_lookup",
    "description": """Perform a comprehensive device lookup across multiple FDA databases.
        Searches adverse events, 510(k) clearances, recalls, and UDI information in a single operation.""",
    
    "required_parameters": [
        {"name": "device_name", "type": "string", "description": "Device name or brand name to search"}
    ],
    
    "optional_parameters": [
        {"name": "limit_per_source", "type": "integer", "default": 10, "description": "Maximum results per data source"},
        {"name": "api_key", "type": "string", "default": None, "description": "FDA API key for higher rate limits"}
    ],
    
    "hardware_requirements": ["Internet connection required", "No GPU required", "Minimal RAM (< 2GB)"],
    "time_complexity": "O(1) - Multiple API calls, typically < 20 seconds total",
    
    "outputs": [
        {"name": "adverse_events", "type": "list", "description": "Device adverse event reports"},
        {"name": "clearances_510k", "type": "list", "description": "510(k) premarket clearances"},
        {"name": "recalls", "type": "list", "description": "Recall and enforcement reports"},
        {"name": "udi_info", "type": "list", "description": "Unique Device Identification records"}
    ],
    
    "failure_modes": [
        {"condition": "Device not found", "behavior": "Returns empty arrays for each source"},
        {"condition": "One source fails", "behavior": "Partial results returned"}
    ]
}

# =============================================================================
# COLLECTION OF ALL FDA TOOLS
# =============================================================================

FDA_TOOLS = [
    # Drug endpoints
    fda_query_drug_events,
    fda_query_drug_label,
    fda_query_drug_ndc,
    fda_query_drug_recalls,
    fda_query_drugs_at_fda,
    fda_query_drug_shortages,
    # Device endpoints
    fda_query_device_events,
    fda_query_device_510k,
    fda_query_device_classification,
    fda_query_device_enforcement,
    fda_query_device_recall,
    fda_query_device_pma,
    fda_query_device_registration,
    fda_query_device_udi,
    fda_query_device_covid19_serology,
    # Food endpoints
    fda_query_food_events,
    fda_query_food_recalls,
    # Animal & Veterinary
    fda_query_animal_events,
    # Substance endpoints
    fda_query_substance_by_unii,
    fda_query_substance_by_name,
    # Utility functions
    fda_generic_query,
    fda_count_by_field,
    fda_drug_safety_profile,
    fda_device_comprehensive_lookup,
]


def get_fda_tools_for_openai() -> list:
    """Get all FDA tools in OpenAI function calling format."""
    from template import get_openai_function_schema
    return [get_openai_function_schema(tool) for tool in FDA_TOOLS]


def get_fda_tools_for_anthropic() -> list:
    """Get all FDA tools in Anthropic tool use format."""
    from template import get_anthropic_tool_schema
    return [get_anthropic_tool_schema(tool) for tool in FDA_TOOLS]


if __name__ == "__main__":
    print(f"FDA Database Tools: {len(FDA_TOOLS)} tools defined")
    for tool in FDA_TOOLS:
        print(f"  - {tool['name']}")
