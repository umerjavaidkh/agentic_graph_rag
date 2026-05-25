#!/usr/bin/env python
"""
Print RBAC curl test commands that you can copy/paste.
"""

import json

API_URL = "http://localhost:8000"

tests = [
    {
        "name": "Admin accessing structured data",
        "description": "Admin can query product data",
        "payload": {
            "question": "Which products are most frequently bought together?",
            "role": "admin",
            "user_id": "admin_001",
            "department": "IT"
        },
        "expected": "✓ Query executes, returns results"
    },
    {
        "name": "Regular office accessing structured data",
        "description": "Regular office can query structured (product/order) data",
        "payload": {
            "question": "What are the top 5 selling products?",
            "role": "regular_office",
            "user_id": "regular_001",
            "department": "Sales"
        },
        "expected": "✓ Query executes (regular_office can query 'structured')"
    },
    {
        "name": "Regular office accessing ESG (DENIED)",
        "description": "Regular office should be blocked from ESG data",
        "payload": {
            "question": "What is our ESG compliance status?",
            "role": "regular_office",
            "user_id": "regular_001",
            "department": "Sales"
        },
        "expected": "✗ Access denied - Access control enforced!"
    },
    {
        "name": "Compliance officer accessing ESG",
        "description": "Compliance officer can query ESG knowledge area",
        "payload": {
            "question": "Show ESG compliance data for Q2 2024",
            "role": "compliance_officer",
            "user_id": "compliance_001",
            "department": "Legal"
        },
        "expected": "✓ Query executes (compliance_officer can query 'esg')"
    },
    {
        "name": "Public user accessing data",
        "description": "Public user has limited access",
        "payload": {
            "question": "What are the product categories?",
            "role": "public",
            "user_id": "public_001",
            "department": "External"
        },
        "expected": "Limited results (public role restrictions)"
    }
]

print("\n" + "="*70)
print("RBAC CURL TEST COMMANDS")
print("="*70)
print("\n🚀 START THE API SERVER FIRST:")
print("   ./venv/bin/python -m src.api")
print("\n" + "="*70 + "\n")

for i, test in enumerate(tests, 1):
    payload_json = json.dumps(test["payload"])
    
    print(f"[TEST {i}] {test['name']}")
    print(f"📝 {test['description']}")
    print(f"\ncurl -X POST {API_URL}/query \\")
    print(f"  -H \"Content-Type: application/json\" \\")
    print(f"  -d '{payload_json}'")
    print(f"\nExpected: {test['expected']}")
    print("\n" + "-"*70 + "\n")

print("="*70)
print("RESPONSE FORMAT")
print("="*70)
print("""
{
  "query": "...",
  "answer": "...",
  "strategy": "text2cypher|vector|multi_hop",
  "sources": [...],
  "total_sources": N,
  "_access_level": "admin|compliance_officer|regular_office|public"
}

If access is DENIED:
{
  "answer": "User does not have permission...",
  "_access_level": "regular_office"
}
""")
print("="*70 + "\n")
