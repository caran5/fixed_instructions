#!/usr/bin/env python3
"""
FDA Database Composition Analysis

Queries the FDA API to get record counts from each endpoint
and creates visualizations showing database composition.
"""

import requests
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# FDA API endpoints to query
ENDPOINTS = {
    'Drugs': {
        'Adverse Events': ('drug', 'event'),
        'Product Labels': ('drug', 'label'),
        'NDC Directory': ('drug', 'ndc'),
        'Enforcement': ('drug', 'enforcement'),
        'Drugs@FDA': ('drug', 'drugsfda'),
    },
    'Devices': {
        'Adverse Events': ('device', 'event'),
        '510(k) Clearances': ('device', '510k'),
        'Classification': ('device', 'classification'),
        'Enforcement': ('device', 'enforcement'),
        'Recalls': ('device', 'recall'),
        'PMA Approvals': ('device', 'pma'),
        'Registrations': ('device', 'registrationlisting'),
        'UDI': ('device', 'udi'),
    },
    'Foods': {
        'Adverse Events': ('food', 'event'),
        'Enforcement': ('food', 'enforcement'),
    },
    'Animal & Veterinary': {
        'Adverse Events': ('animalandveterinary', 'event'),
    },
    'Substances': {
        'Substance Data': ('other', 'substance'),
    }
}

def get_endpoint_count(category, endpoint):
    """Query FDA API to get total record count for an endpoint."""
    url = f"https://api.fda.gov/{category}/{endpoint}.json?limit=1"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'meta' in data and 'results' in data['meta']:
                return data['meta']['results']['total']
        return 0
    except Exception as e:
        print(f"Error querying {category}/{endpoint}: {e}")
        return 0

def format_number(num):
    """Format large numbers with K, M suffixes."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

# Collect data
print("Querying FDA database endpoints...")
print("-" * 50)

category_totals = {}
endpoint_data = {}

for category, endpoints in ENDPOINTS.items():
    category_total = 0
    endpoint_data[category] = {}

    for name, (cat, ep) in endpoints.items():
        count = get_endpoint_count(cat, ep)
        endpoint_data[category][name] = count
        category_total += count
        print(f"{category} - {name}: {format_number(count)} records")

    category_totals[category] = category_total
    print(f"  >> {category} Total: {format_number(category_total)}")
    print()

total_records = sum(category_totals.values())
print(f"TOTAL DATABASE RECORDS: {format_number(total_records)}")
print("=" * 50)

# Create visualizations
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#f8f9fa')

# Color scheme
colors = {
    'Drugs': '#4CAF50',
    'Devices': '#2196F3',
    'Foods': '#FF9800',
    'Animal & Veterinary': '#9C27B0',
    'Substances': '#F44336',
}

# 1. Main pie chart - Category composition
ax1 = fig.add_subplot(2, 2, 1)
categories = list(category_totals.keys())
values = list(category_totals.values())
cat_colors = [colors[c] for c in categories]

# Filter out zero values
non_zero = [(c, v, col) for c, v, col in zip(categories, values, cat_colors) if v > 0]
if non_zero:
    categories, values, cat_colors = zip(*non_zero)

wedges, texts, autotexts = ax1.pie(values, labels=categories, autopct='%1.1f%%',
                                    colors=cat_colors, explode=[0.02]*len(values),
                                    shadow=True, startangle=90)
ax1.set_title('FDA Database Composition by Category', fontsize=14, fontweight='bold', pad=20)

# Style the percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 2. Bar chart - Records by category
ax2 = fig.add_subplot(2, 2, 2)
y_pos = np.arange(len(category_totals))
bars = ax2.barh(y_pos, list(category_totals.values()),
                color=[colors[c] for c in category_totals.keys()],
                edgecolor='white', linewidth=1.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(list(category_totals.keys()))
ax2.set_xlabel('Number of Records')
ax2.set_title('Total Records by Category', fontsize=14, fontweight='bold')
ax2.set_facecolor('#f8f9fa')

# Add value labels on bars
for bar, val in zip(bars, category_totals.values()):
    ax2.text(bar.get_width() + max(category_totals.values())*0.01,
             bar.get_y() + bar.get_height()/2,
             format_number(val), va='center', fontsize=10, fontweight='bold')

ax2.set_xlim(0, max(category_totals.values()) * 1.15)

# 3. Detailed breakdown - Drugs
ax3 = fig.add_subplot(2, 2, 3)
drug_data = endpoint_data['Drugs']
drug_names = list(drug_data.keys())
drug_values = list(drug_data.values())

# Sort by value
sorted_pairs = sorted(zip(drug_values, drug_names), reverse=True)
drug_values, drug_names = zip(*sorted_pairs) if sorted_pairs else ([], [])

y_pos = np.arange(len(drug_names))
bars = ax3.barh(y_pos, drug_values, color=colors['Drugs'], alpha=0.8, edgecolor='white')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(drug_names)
ax3.set_xlabel('Number of Records')
ax3.set_title('Drugs Category Breakdown', fontsize=14, fontweight='bold')
ax3.set_facecolor('#f8f9fa')

for bar, val in zip(bars, drug_values):
    if val > 0:
        ax3.text(bar.get_width() + max(drug_values)*0.01 if drug_values else 0,
                 bar.get_y() + bar.get_height()/2,
                 format_number(val), va='center', fontsize=9)

if drug_values:
    ax3.set_xlim(0, max(drug_values) * 1.2)

# 4. Detailed breakdown - Devices
ax4 = fig.add_subplot(2, 2, 4)
device_data = endpoint_data['Devices']
device_names = list(device_data.keys())
device_values = list(device_data.values())

# Sort by value
sorted_pairs = sorted(zip(device_values, device_names), reverse=True)
device_values, device_names = zip(*sorted_pairs) if sorted_pairs else ([], [])

y_pos = np.arange(len(device_names))
bars = ax4.barh(y_pos, device_values, color=colors['Devices'], alpha=0.8, edgecolor='white')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(device_names)
ax4.set_xlabel('Number of Records')
ax4.set_title('Devices Category Breakdown', fontsize=14, fontweight='bold')
ax4.set_facecolor('#f8f9fa')

for bar, val in zip(bars, device_values):
    if val > 0:
        ax4.text(bar.get_width() + max(device_values)*0.01 if device_values else 0,
                 bar.get_y() + bar.get_height()/2,
                 format_number(val), va='center', fontsize=9)

if device_values:
    ax4.set_xlim(0, max(device_values) * 1.2)

# Main title
fig.suptitle(f'FDA openFDA Database Composition\nTotal Records: {format_number(total_records)}',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/Users/ceejayarana/claude/fda_database_composition.png', dpi=150,
            bbox_inches='tight', facecolor='#f8f9fa')
plt.show()

print("\nVisualization saved to: /Users/ceejayarana/claude/fda_database_composition.png")

# Print summary table
print("\n" + "=" * 60)
print("FDA DATABASE COMPOSITION SUMMARY")
print("=" * 60)
print(f"{'Category':<25} {'Records':>15} {'Percentage':>15}")
print("-" * 60)
for cat, total in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
    pct = (total / total_records * 100) if total_records > 0 else 0
    print(f"{cat:<25} {format_number(total):>15} {pct:>14.1f}%")
print("-" * 60)
print(f"{'TOTAL':<25} {format_number(total_records):>15} {'100.0%':>15}")
