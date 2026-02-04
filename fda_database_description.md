# FDA Database Tool Description

## Overview
The FDA Database tool provides programmatic access to the U.S. Food and Drug Administration's comprehensive databases, including drug approvals, medical device registrations, food safety data, and adverse event reports.

## Key Features
- **Drug Information**: Access to FDA-approved drugs, including NDC codes, labeling, and approval history
- **Medical Devices**: Search device registrations, 510(k) clearances, and recalls
- **Adverse Events**: Query FAERS (FDA Adverse Event Reporting System) data
- **Food Safety**: Access food recalls, outbreak data, and enforcement reports
- **Clinical Trials**: Information on FDA-regulated clinical trials

## Primary Use Cases

### 1. Drug Discovery & Development
- Research approved drug formulations
- Analyze drug approval timelines
- Study drug-drug interactions

### 2. Regulatory Compliance
- Verify FDA approval status
- Check product registrations
- Monitor recalls and enforcement actions

### 3. Pharmacovigilance
- Analyze adverse event patterns
- Safety signal detection
- Post-market surveillance

### 4. Market Intelligence
- Track new drug approvals
- Competitive landscape analysis
- Generic drug availability

## Installation & Setup

### MCP Server Configuration (Claude Desktop)
```json
{
  "mcpServers": {
    "fda-database": {
      "command": "npx",
      "args": ["-y", "@davila7/claude-code-templates", "fda-database"]
    }
  }
}
```

**Configuration File Location (macOS):**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

After adding the configuration, restart Claude Desktop to activate the FDA database skill.

## API Access Points
- **openFDA API**: drugs, devices, foods, animal-veterinary
- **NDC Directory**: National Drug Code database
- **FAERS**: FDA Adverse Event Reporting System
- **Device Registrations**: Medical device database

## Example Queries

### Drug Search
```python
# Search for drug by name
fda.search_drugs(brand_name="Aspirin")

# Search by active ingredient
fda.search_drugs(generic_name="acetylsalicylic acid")

# Search by NDC code
fda.search_drugs(ndc="0363-0123-01")
```

### Adverse Events
```python
# Get adverse events for a drug
fda.get_adverse_events(drug_name="Ibuprofen", limit=100)

# Filter by severity
fda.get_adverse_events(
    drug_name="Warfarin",
    serious=True,
    date_range="2023-01-01:2025-12-31"
)
```

### Device Recalls
```python
# Check device recalls
fda.search_device_recalls(classification="Class I")

# Search by product code
fda.search_device_recalls(product_code="LWI")
```

### Food Safety
```python
# Search food recalls
fda.search_food_recalls(
    reason_for_recall="Salmonella",
    date_range="2024-01-01:2025-12-31"
)
```

## Data Format
- **Response Format**: JSON
- **API Type**: RESTful
- **Rate Limits**: 
  - Without API key: 240 requests/minute, 1000 requests/hour
  - With API key: 240 requests/minute, unlimited hourly

## Common API Endpoints

### Drug Endpoints
- `/drug/event.json` - Adverse event reports
- `/drug/label.json` - Drug labeling
- `/drug/ndc.json` - National Drug Code directory
- `/drug/enforcement.json` - Drug recalls

### Device Endpoints
- `/device/event.json` - Device adverse events
- `/device/recall.json` - Device recalls
- `/device/classification.json` - Device classifications
- `/device/510k.json` - 510(k) clearances
- `/device/pma.json` - Premarket approvals

### Food Endpoints
- `/food/enforcement.json` - Food recalls
- `/food/event.json` - Food adverse events

## Query Syntax

### Basic Search
```
search=brand_name:"Lipitor"
```

### Multiple Conditions
```
search=brand_name:"Lipitor"+AND+generic_name:"atorvastatin"
```

### Date Ranges
```
search=receivedate:[20240101+TO+20251231]
```

### Limiting Results
```
limit=100&skip=0
```

## Best Practices

### 1. API Key Usage
Always include an API key for production applications to avoid rate limiting:
```python
fda.set_api_key("your-api-key-here")
```

### 2. Caching
Cache frequently accessed data to reduce API calls:
```python
import functools

@functools.lru_cache(maxsize=128)
def get_drug_info(ndc_code):
    return fda.search_drugs(ndc=ndc_code)
```

### 3. Error Handling
Implement robust error handling for API failures:
```python
try:
    results = fda.search_drugs(brand_name="Aspirin")
except fda.RateLimitError:
    time.sleep(60)  # Wait before retrying
except fda.APIError as e:
    logger.error(f"FDA API error: {e}")
```

### 4. Data Validation
Always validate data freshness as FDA updates regularly:
```python
# Check last update timestamp
if result['meta']['last_updated'] > cutoff_date:
    process_data(result)
```

## Data Quality Considerations

### Adverse Events
- Reports are voluntary and may contain errors
- Duplicate reports may exist
- Not all adverse events are drug-related
- Underreporting is common

### Drug Labeling
- Labels are submitted by manufacturers
- May not reflect latest safety information
- Check `effective_time` for currency

### Recalls
- Updated daily
- Check `status` field for current state
- Historical data available

## Common Use Cases

### 1. Drug Safety Research
```python
# Analyze adverse event patterns
events = fda.get_adverse_events(
    drug_name="Metformin",
    limit=10000
)

# Group by reaction
from collections import Counter
reactions = Counter(e['reaction'] for e in events)
top_reactions = reactions.most_common(10)
```

### 2. Competitive Intelligence
```python
# Track recent approvals in therapeutic area
approvals = fda.search_drugs(
    approval_date_range="2024-01-01:2025-12-31",
    therapeutic_class="Antineoplastic"
)

# Analyze approval trends
by_company = {}
for drug in approvals:
    company = drug['sponsor_name']
    by_company[company] = by_company.get(company, 0) + 1
```

### 3. Supply Chain Monitoring
```python
# Check for drug shortages
shortages = fda.get_drug_shortages(active=True)

# Monitor device recalls
recalls = fda.search_device_recalls(
    classification="Class I",
    date_range="2025-01-01:2025-12-31"
)
```

## Resources

### Official Documentation
- **openFDA**: https://open.fda.gov/
- **API Reference**: https://open.fda.gov/apis/
- **Interactive Query Tool**: https://open.fda.gov/apis/try-the-api/
- **API Status**: https://open.fda.gov/status/

### Data Downloads
- **Bulk Downloads**: https://open.fda.gov/downloads/
- **Data Dictionary**: https://open.fda.gov/data/

### Support
- **GitHub Issues**: https://github.com/FDA/openfda
- **Email**: open@fda.hhs.gov
- **Updates**: Follow @openFDA on Twitter

## Limitations

1. **Historical Data**: Limited historical data for some endpoints
2. **Real-time**: Not real-time; updates vary by dataset (daily to quarterly)
3. **Completeness**: Not all FDA data is available via openFDA
4. **International**: U.S. data only; no international drug databases
5. **Rate Limits**: Must respect API rate limits

## Troubleshooting

### Common Issues

**Issue**: Rate limit exceeded
```
Solution: Implement exponential backoff or obtain API key
```

**Issue**: Empty results
```
Solution: Check query syntax, verify data availability for date range
```

**Issue**: Slow responses
```
Solution: Reduce result limit, add more specific filters
```

**Issue**: Missing fields
```
Solution: Fields are optional; check for existence before accessing
```

## Updates & Changelog

The FDA continuously updates their databases. Check the API status page for:
- New endpoints
- Schema changes
- Data updates
- Maintenance windows

---

*Last Updated: February 4, 2026*
*Tool Type: REST API / MCP Server*
