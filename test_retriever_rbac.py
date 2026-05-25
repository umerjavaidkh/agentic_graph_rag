#!/usr/bin/env python
"""
test_retriever_rbac.py — Test retriever access control with different roles.

Tests that retrievers enforce access checks before querying.
"""

from src.auth.roles import UserContext, validate_role
from src.structured.retriever import StructuredRetriever
from src.unstructured.retriever import ESGComplianceRetriever


def test_structured_retriever_access():
    """Test structured retriever access control."""
    
    print("\n" + "="*60)
    print("TESTING STRUCTURED RETRIEVER ACCESS CONTROL")
    print("="*60)
    
    retriever = StructuredRetriever()
    
    # Test 1: Admin can access
    print("\n[TEST 1] Admin retrieving structured data")
    admin_ctx = UserContext(
        user_id='admin_001',
        role=validate_role('admin')
    )
    result = retriever.retrieve(
        "What products exist?",
        limit=3,
        user_context=admin_ctx
    )
    if result['chunks'] and result['chunks'][0]['id'] != 'access_denied':
        print(f"  ✓ ALLOWED - Retriever executed")
        print(f"     Strategy: {result['strategy']}")
    else:
        print(f"  ✗ DENIED - Access blocked")
    
    # Test 2: Regular office can access structured
    print("\n[TEST 2] Regular office retrieving structured data")
    regular_ctx = UserContext(
        user_id='regular_001',
        role=validate_role('regular_office')
    )
    result = retriever.retrieve(
        "What products exist?",
        limit=3,
        user_context=regular_ctx
    )
    if result['chunks'] and result['chunks'][0]['id'] != 'access_denied':
        print(f"  ✓ ALLOWED - Retriever executed")
        print(f"     Strategy: {result['strategy']}")
    else:
        print(f"  ✗ DENIED - Access blocked")
    
    retriever.close()


def test_esg_retriever_access():
    """Test ESG retriever access control."""
    
    print("\n" + "="*60)
    print("TESTING ESG RETRIEVER ACCESS CONTROL")
    print("="*60)
    
    retriever = ESGComplianceRetriever()
    
    # Test 1: Admin can access ESG
    print("\n[TEST 1] Admin accessing ESG data")
    admin_ctx = UserContext(
        user_id='admin_001',
        role=validate_role('admin')
    )
    result = retriever.semantic_retrieve(
        "ESG compliance",
        limit=3,
        user_context=admin_ctx
    )
    if result['chunks'] and result['chunks'][0]['id'] != 'access_denied':
        print(f"  ✓ ALLOWED - Retriever executed (got {result['total_available']} results)")
    else:
        print(f"  ✗ DENIED - Access blocked")
    
    # Test 2: Regular office CANNOT access ESG
    print("\n[TEST 2] Regular office accessing ESG data (should be DENIED)")
    regular_ctx = UserContext(
        user_id='regular_001',
        role=validate_role('regular_office')
    )
    result = retriever.semantic_retrieve(
        "ESG compliance",
        limit=3,
        user_context=regular_ctx
    )
    if result['chunks'] and result['chunks'][0]['id'] == 'access_denied':
        print(f"  ✓ DENIED (expected) - Access blocked with message:")
        print(f"     '{result['chunks'][0]['text']}'")
    else:
        print(f"  ✗ UNEXPECTED - Access should have been denied")
    
    # Test 3: Compliance officer can access ESG
    print("\n[TEST 3] Compliance officer accessing ESG data")
    compliance_ctx = UserContext(
        user_id='compliance_001',
        role=validate_role('compliance_officer')
    )
    result = retriever.semantic_retrieve(
        "ESG compliance",
        limit=3,
        user_context=compliance_ctx
    )
    if result['chunks'] and result['chunks'][0]['id'] != 'access_denied':
        print(f"  ✓ ALLOWED - Retriever executed (got {result['total_available']} results)")
    else:
        print(f"  ✗ DENIED - Access blocked")
    
    retriever.close()


if __name__ == "__main__":
    test_structured_retriever_access()
    test_esg_retriever_access()
    
    print("\n" + "="*60)
    print("✓ ALL RETRIEVER ACCESS TESTS COMPLETED")
    print("="*60 + "\n")
