#!/usr/bin/env python3
"""
FDA Database Structure Visualization

Creates a visual representation of the FDA openFDA database categories,
endpoints, and their relationships.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

# Color scheme
colors = {
    'drugs': '#4CAF50',      # Green
    'devices': '#2196F3',    # Blue
    'foods': '#FF9800',      # Orange
    'animal': '#9C27B0',     # Purple
    'substances': '#F44336', # Red
    'header': '#1a1a2e',     # Dark blue
}

def draw_category_box(ax, x, y, width, height, title, endpoints, color):
    """Draw a category box with its endpoints."""
    # Main category box
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor=color, edgecolor='white',
                         linewidth=2, alpha=0.9)
    ax.add_patch(box)

    # Category title
    ax.text(x + width/2, y + height - 0.4, title,
            ha='center', va='top', fontsize=14, fontweight='bold',
            color='white')

    # Endpoints
    endpoint_y = y + height - 0.9
    for endpoint in endpoints:
        ax.text(x + 0.3, endpoint_y, f"• {endpoint}",
                fontsize=9, color='white', alpha=0.95)
        endpoint_y -= 0.35

# Title
ax.text(8, 11.5, 'FDA openFDA Database Structure',
        ha='center', va='center', fontsize=22, fontweight='bold',
        color=colors['header'])
ax.text(8, 10.9, 'Comprehensive access to FDA regulatory data through openFDA API',
        ha='center', va='center', fontsize=11, color='#666666', style='italic')

# Draw category boxes

# DRUGS (Top Left)
drug_endpoints = [
    'Adverse Events',
    'Product Labeling',
    'NDC Directory',
    'Enforcement Reports',
    'Drugs@FDA',
    'Drug Shortages'
]
draw_category_box(ax, 0.5, 6.5, 4.5, 4, 'DRUGS', drug_endpoints, colors['drugs'])

# DEVICES (Top Right)
device_endpoints = [
    'Adverse Events',
    '510(k) Clearances',
    'Classification',
    'Enforcement Reports',
    'Recalls',
    'PMA Approvals',
    'Registrations',
    'UDI Database',
    'COVID-19 Serology'
]
draw_category_box(ax, 5.5, 5.5, 4.5, 5, 'DEVICES', device_endpoints, colors['devices'])

# FOODS (Middle Right)
food_endpoints = [
    'Adverse Events',
    'Enforcement Reports'
]
draw_category_box(ax, 10.5, 7.5, 4.5, 2.5, 'FOODS', food_endpoints, colors['foods'])

# ANIMAL & VETERINARY (Bottom Left)
animal_endpoints = [
    'Adverse Events'
]
draw_category_box(ax, 0.5, 4, 4.5, 2, 'ANIMAL & VETERINARY', animal_endpoints, colors['animal'])

# SUBSTANCES (Bottom Right)
substance_endpoints = [
    'Substance Data (UNII)',
    'NSDE (Legacy)'
]
draw_category_box(ax, 10.5, 4.5, 4.5, 2.5, 'SUBSTANCES', substance_endpoints, colors['substances'])

# Central hub - API
hub_x, hub_y = 8, 2.5
hub = plt.Circle((hub_x, hub_y), 1.2, color=colors['header'], alpha=0.9)
ax.add_patch(hub)
ax.text(hub_x, hub_y + 0.15, 'openFDA', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')
ax.text(hub_x, hub_y - 0.25, 'API', ha='center', va='center',
        fontsize=12, color='white')

# Connection lines from hub to categories
connections = [
    ((hub_x - 0.8, hub_y + 0.8), (2.75, 6.5)),    # to Drugs
    ((hub_x, hub_y + 1.2), (7.75, 5.5)),           # to Devices
    ((hub_x + 0.8, hub_y + 0.8), (12.75, 7.5)),    # to Foods
    ((hub_x - 0.8, hub_y + 0.4), (2.75, 4)),       # to Animal
    ((hub_x + 0.8, hub_y + 0.4), (12.75, 4.5)),    # to Substances
]

for start, end in connections:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='#666666',
                               lw=1.5, connectionstyle='arc3,rad=0.1'))

# Legend / Key info at bottom
info_y = 0.8
ax.text(0.5, info_y, 'API Access:', fontsize=10, fontweight='bold', color=colors['header'])
ax.text(0.5, info_y - 0.4, '• Base URL: https://api.fda.gov', fontsize=9, color='#444444')
ax.text(0.5, info_y - 0.7, '• Rate Limit: 240 req/min (with key: 120,000/day)', fontsize=9, color='#444444')

ax.text(6, info_y, 'Key Features:', fontsize=10, fontweight='bold', color=colors['header'])
ax.text(6, info_y - 0.4, '• Real-time adverse event monitoring', fontsize=9, color='#444444')
ax.text(6, info_y - 0.7, '• Product recalls and enforcement actions', fontsize=9, color='#444444')

ax.text(11.5, info_y, 'Data Types:', fontsize=10, fontweight='bold', color=colors['header'])
ax.text(11.5, info_y - 0.4, '• Safety reports, labels, approvals', fontsize=9, color='#444444')
ax.text(11.5, info_y - 0.7, '• Chemical structures, UNII codes', fontsize=9, color='#444444')

plt.tight_layout()
plt.savefig('/Users/ceejayarana/claude/fda_database_structure.png', dpi=150,
            bbox_inches='tight', facecolor='#f8f9fa')
plt.show()
print("Visualization saved to: /Users/ceejayarana/claude/fda_database_structure.png")
