#!/usr/bin/env python
"""
test_rbac_access.py — Test graph-based RBAC with different user roles.

Tests access control WITHOUT re-ingesting actual data.
"""

from src.auth.roles import UserContext, validate_role
from src.auth.rbac_setup import GraphRBAC


def test_rbac_access():
    """Test access control for different roles."""
    
    rbac = GraphRBAC()
    
    print("\n" + "="*60)
    print("TESTING GRAPH-BASED RBAC")
    print("="*60)
    
    # Test case 1: Admin accessing document graph
    print("\n[TEST 1] Admin accessing Agentic Graph RAG knowledge area")
    admin_ctx = UserContext(
        user_id='admin_001',
        role=validate_role('admin')
    )
    can_access = rbac.can_query_knowledge_area(admin_ctx.user_id, 'esg')
    print(f"  User: {admin_ctx.user_id} ({admin_ctx.role.value})")
    print(f"  Knowledge Area: esg")
    print(f"  Result: {'✓ ALLOWED' if can_access else '✗ DENIED'}")
    
    # Test case 2: Regular office accessing document graph (should be denied)
    print("\n[TEST 2] Regular office accessing Agentic Graph RAG (should be DENIED)")
    regular_ctx = UserContext(
        user_id='regular_001',
        role=validate_role('regular_office')
    )
    can_access = rbac.can_query_knowledge_area(regular_ctx.user_id, 'esg')
    print(f"  User: {regular_ctx.user_id} ({regular_ctx.role.value})")
    print(f"  Knowledge Area: esg")
    print(f"  Result: {'✓ ALLOWED' if can_access else '✗ DENIED (expected)'}")
    
    # Test case 3: Regular office accessing structured (should be allowed)
    print("\n[TEST 3] Regular office accessing structured")
    can_access = rbac.can_query_knowledge_area(regular_ctx.user_id, 'structured')
    print(f"  User: {regular_ctx.user_id} ({regular_ctx.role.value})")
    print(f"  Knowledge Area: structured")
    print(f"  Result: {'✓ ALLOWED (expected)' if can_access else '✗ DENIED'}")
    
    # Test case 4: Compliance officer accessing document graph
    print("\n[TEST 4] Compliance officer accessing Agentic Graph RAG")
    compliance_ctx = UserContext(
        user_id='compliance_001',
        role=validate_role('compliance_officer')
    )
    can_access = rbac.can_query_knowledge_area(compliance_ctx.user_id, 'esg')
    print(f"  User: {compliance_ctx.user_id} ({compliance_ctx.role.value})")
    print(f"  Knowledge Area: esg")
    print(f"  Result: {'✓ ALLOWED (expected)' if can_access else '✗ DENIED'}")
    
    # Test case 5: Public user accessing public KA
    print("\n[TEST 5] Public user accessing Agentic Graph RAG")
    public_ctx = UserContext(
        user_id='public_001',
        role=validate_role('public')
    )
    can_access = rbac.can_query_knowledge_area(public_ctx.user_id, 'esg')
    print(f"  User: {public_ctx.user_id} ({public_ctx.role.value})")
    print(f"  Knowledge Area: esg")
    print(f"  Result: {'✓ ALLOWED' if can_access else '✗ DENIED'}")
    
    # Test case 6: Document access
    print("\n[TEST 6] Document access control")
    print(f"\n  Admin can view sensitive doc? {rbac.can_view_document('admin_001', 'doc_data_policy')}")
    print(f"  Regular office can view sensitive doc? {rbac.can_view_document('regular_001', 'doc_data_policy')}")
    print(f"  Regular office can view public doc? {rbac.can_view_document('regular_001', 'doc_product_catalog')}")
    
    # Test case 7: Accessible knowledge areas
    print("\n[TEST 7] Listing accessible knowledge areas by role")
    print(f"\n  Admin accessible KAs:")
    for ka in rbac.get_accessible_knowledge_areas('admin_001'):
        print(f"    - {ka['name']} ({ka['id']})")
    
    print(f"\n  Regular office accessible KAs:")
    for ka in rbac.get_accessible_knowledge_areas('regular_001'):
        print(f"    - {ka['name']} ({ka['id']})")
    
    # Test case 8: Validate comprehensive access
    print("\n[TEST 8] Comprehensive access validation")
    validation = rbac.validate_and_enforce_access(
        user_id='regular_001',
        knowledge_area_id='structured',
        document_ids=['doc_product_catalog', 'doc_data_policy']
    )
    print(f"\n  User: {validation['user_id']}")
    print(f"  Roles: {validation['roles']}")
    print(f"  Can query 'structured'? {validation['can_query_ka']}")
    print(f"  Requested docs: {['doc_product_catalog', 'doc_data_policy']}")
    print(f"  Accessible docs: {validation['accessible_docs']}")
    print(f"  All accessible docs: {len(validation['all_accessible_docs'])} documents")
    
    rbac.close()
    
    print("\n" + "="*60)
    print("✓ RBAC ACCESS TESTS COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_rbac_access()
