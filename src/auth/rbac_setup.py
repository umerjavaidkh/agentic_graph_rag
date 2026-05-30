"""
src/auth/rbac_setup.py — Graph-based RBAC schema setup and queries.

This module provides:
1. Schema initialization (constraints, indexes, sample data)
2. Access control checks via Neo4j graph traversal
3. Query builders that enforce role-based filtering at the database level
"""

from typing import List, Dict, Optional
from neo4j import GraphDatabase

from ..config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class GraphRBAC:
    """Graph-based Role-Based Access Control using Neo4j relationships."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def setup_schema(self, cypher_file: str = "src/auth/rbac_schema.cypher"):
        """
        Load and execute RBAC schema setup from Cypher file.
        Creates: User, Role, KnowledgeArea, Document, Policy, Entity nodes + relationships.
        """
        try:
            with open(cypher_file, 'r') as f:
                cypher = f.read()
            
            with self.driver.session() as session:
                # Split by semicolon and execute statements individually
                statements = [s.strip() for s in cypher.split(';') if s.strip()]
                for stmt in statements:
                    session.run(stmt)
            
            print(f"✓ RBAC schema initialized from {cypher_file}")
            return True
        except Exception as e:
            print(f"✗ Schema setup failed: {e}")
            return False

    def is_initialized(self) -> bool:
        """
        Check whether the RBAC seed schema/data appears to be present.

        This is intentionally a lightweight probe so it can run on app startup.
        """
        cypher = """
            MATCH (ka:KnowledgeArea {id: 'esg'})
            MATCH (r:Role {name: 'public'})
            RETURN count(ka) > 0 AND count(r) > 0 AS ok
        """
        try:
            with self.driver.session() as session:
                row = session.run(cypher).single()
                return bool(row and row["ok"])
        except Exception:
            return False

    def can_query_knowledge_area(self, user_id: str, knowledge_area_id: str) -> bool:
        """
        Check if user can query a knowledge area.
        Traverses: User -[:HAS_ROLE]-> Role -[:CAN_QUERY]-> KnowledgeArea
        """
        cypher = """
            MATCH (u:User {user_id: $user_id})-[:HAS_ROLE]->(r:Role)-[:CAN_QUERY]->(ka:KnowledgeArea {id: $ka_id})
            RETURN count(r) > 0 AS has_access
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id, ka_id=knowledge_area_id)
            return result.single()["has_access"]

    def can_view_document(self, user_id: str, document_id: str) -> bool:
        """
        Check if user can view a document.
        Traverses: User -[:HAS_ROLE]-> Role -[:CAN_VIEW]-> Document
        """
        cypher = """
            MATCH (u:User {user_id: $user_id})-[:HAS_ROLE]->(r:Role)-[:CAN_VIEW]->(d:Document {id: $doc_id})
            RETURN count(r) > 0 AS has_access
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id, doc_id=document_id)
            return result.single()["has_access"]

    def can_edit_policy(self, user_id: str, policy_id: str) -> bool:
        """
        Check if user can edit a policy.
        Traverses: User -[:HAS_ROLE]-> Role -[:CAN_EDIT]-> Policy
        """
        cypher = """
            MATCH (u:User {user_id: $user_id})-[:HAS_ROLE]->(r:Role)-[:CAN_EDIT]->(p:Policy {id: $policy_id})
            RETURN count(r) > 0 AS has_access
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id, policy_id=policy_id)
            return result.single()["has_access"]

    def get_user_roles(self, user_id: str) -> List[str]:
        """Get all role names for a user."""
        cypher = """
            MATCH (u:User {user_id: $user_id})-[:HAS_ROLE]->(r:Role)
            RETURN collect(r.name) AS roles
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id)
            row = result.single()
            return row["roles"] if row else []

    def get_accessible_knowledge_areas(self, user_id: str) -> List[Dict]:
        """Get all knowledge areas accessible by user."""
        cypher = """
            MATCH (u:User {user_id: $user_id})-[:HAS_ROLE]->(r:Role)-[:CAN_QUERY]->(ka:KnowledgeArea)
            RETURN ka.id AS id, ka.name AS name, ka.description AS description
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id)
            return [r.data() for r in result]

    def get_accessible_documents(self, user_id: str) -> List[Dict]:
        """Get all documents accessible by user via CAN_VIEW."""
        cypher = """
            MATCH (u:User {user_id: $user_id})-[:HAS_ROLE]->(r:Role)-[:CAN_VIEW]->(d:Document)
            RETURN d.id AS id, d.title AS title, d.sensitivity AS sensitivity, d.knowledge_area AS knowledge_area
        """
        with self.driver.session() as session:
            result = session.run(cypher, user_id=user_id)
            return [r.data() for r in result]

    def build_cypher_with_access_check(
        self,
        user_id: str,
        knowledge_area_id: str,
        base_cypher: str
    ) -> Optional[str]:
        """
        Enhance a base Cypher query with user access control.
        
        Example:
            base = "MATCH (n:Product) RETURN n LIMIT 10"
            enhanced = build_cypher_with_access_check('user_123', 'structured', base)
            # Returns: 
            # MATCH (u:User {user_id: 'user_123'})-[:HAS_ROLE]->(r:Role)-[:CAN_QUERY]->(ka:KnowledgeArea {id: 'structured'})
            # WITH r
            # [original query here]
        
        Returns None if user lacks access.
        """
        if not self.can_query_knowledge_area(user_id, knowledge_area_id):
            return None
        
        # Wrap query with access check
        wrapped = f"""
            MATCH (u:User {{user_id: '{user_id}'}})-[:HAS_ROLE]->(r:Role)-[:CAN_QUERY]->(ka:KnowledgeArea {{id: '{knowledge_area_id}'}})
            WITH r, ka
            {base_cypher}
        """
        return wrapped.strip()

    def build_document_filter_cypher(self, user_id: str) -> str:
        """
        Build a Cypher WHERE clause that restricts results to documents user can view.
        
        Usage:
            filter_clause = build_document_filter_cypher('user_123')
            cypher = f"MATCH (d:Document) {filter_clause} RETURN d"
        """
        return f"""
            WHERE EXISTS ((User {{user_id: '{user_id}'}})-[:HAS_ROLE]->(r:Role)-[:CAN_VIEW]->(d))
        """

    def validate_and_enforce_access(
        self,
        user_id: str,
        knowledge_area_id: str,
        document_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive access validation.
        
        Returns:
            {
                'user_id': str,
                'can_query_ka': bool,
                'accessible_docs': [list of accessible docs from the provided list],
                'all_accessible_docs': [all docs user can view],
                'roles': [user's roles]
            }
        """
        result = {
            'user_id': user_id,
            'can_query_ka': self.can_query_knowledge_area(user_id, knowledge_area_id),
            'roles': self.get_user_roles(user_id),
            'all_accessible_docs': self.get_accessible_documents(user_id),
        }
        
        if document_ids:
            accessible = result['all_accessible_docs']
            accessible_ids = {d['id'] for d in accessible}
            result['accessible_docs'] = [
                doc_id for doc_id in document_ids if doc_id in accessible_ids
            ]
        
        return result

    def close(self):
        self.driver.close()


# Convenience function for RBAC schema setup
def initialize_rbac_schema(
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    cypher_file: str = "src/auth/rbac_schema.cypher"
) -> bool:
    """Initialize RBAC schema from Cypher file."""
    rbac = GraphRBAC(uri, user, password)
    success = rbac.setup_schema(cypher_file)
    rbac.close()
    return success


# Test / example usage
if __name__ == "__main__":
    rbac = GraphRBAC()
    
    # Setup schema (comment out after first run)
    print("Setting up RBAC schema...")
    rbac.setup_schema()
    
    # Test access checks
    print("\n--- Testing Access Checks ---")
    print(f"admin_001 can query Agentic Graph RAG? {rbac.can_query_knowledge_area('admin_001', 'esg')}")
    print(f"regular_001 can query Agentic Graph RAG? {rbac.can_query_knowledge_area('regular_001', 'esg')}")
    print(f"regular_001 can query structured? {rbac.can_query_knowledge_area('regular_001', 'structured')}")
    
    print("\nRoles for admin_001:", rbac.get_user_roles('admin_001'))
    print("Roles for regular_001:", rbac.get_user_roles('regular_001'))
    
    print("\nAccessible KAs for regular_001:", rbac.get_accessible_knowledge_areas('regular_001'))
    print("Accessible docs for regular_001:", rbac.get_accessible_documents('regular_001'))
    
    rbac.close()
