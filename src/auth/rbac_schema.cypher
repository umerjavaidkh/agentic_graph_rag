// src/auth/rbac_schema.cypher
// Graph-based Role-Based Access Control Schema
// Supports user → role → permissions → data access

// ─────────────────────────────────────────
// CONSTRAINTS & INDEXES
// ─────────────────────────────────────────

CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE u.user_id IS UNIQUE;

CREATE CONSTRAINT role_name IF NOT EXISTS
FOR (r:Role) REQUIRE r.name IS UNIQUE;

CREATE CONSTRAINT knowledge_area_id IF NOT EXISTS
FOR (ka:KnowledgeArea) REQUIRE ka.id IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT policy_id IF NOT EXISTS
FOR (p:Policy) REQUIRE p.id IS UNIQUE;

// Indexes for faster queries
CREATE INDEX user_id_idx IF NOT EXISTS FOR (u:User) ON (u.user_id);
CREATE INDEX role_name_idx IF NOT EXISTS FOR (r:Role) ON (r.name);
CREATE INDEX ka_id_idx IF NOT EXISTS FOR (ka:KnowledgeArea) ON (ka.id);
CREATE INDEX doc_sensitivity_idx IF NOT EXISTS FOR (d:Document) ON (d.sensitivity);
CREATE INDEX chunk_sensitivity_idx IF NOT EXISTS FOR (c:Chunk) ON (c.sensitivity);

// ─────────────────────────────────────────
// SAMPLE ROLES & PERMISSIONS
// ─────────────────────────────────────────

// Create Roles
MERGE (admin:Role {name: 'admin', description: 'Full access to all data'})
MERGE (compliance:Role {name: 'compliance_officer', description: 'Access to sensitive/compliance data'})
MERGE (regular:Role {name: 'regular_office', description: 'Access to public/internal data'})
MERGE (public_role:Role {name: 'public', description: 'Access to public data only'});

// Create Knowledge Areas (query domains)
MERGE (esg:KnowledgeArea {
  id: 'esg',
  name: 'ESG Compliance',
  description: 'Environmental, Social, Governance data'
})
MERGE (structured:KnowledgeArea {
  id: 'structured',
  name: 'Structured Data',
  description: 'Product, customer, order data'
})
MERGE (policies:KnowledgeArea {
  id: 'policies',
  name: 'Company Policies',
  description: 'Internal policies and procedures'
});

// Role → KnowledgeArea (CAN_QUERY permissions)
MATCH (admin:Role {name: 'admin'})
MATCH (esg:KnowledgeArea {id: 'esg'})
MATCH (struct:KnowledgeArea {id: 'structured'})
MATCH (pol:KnowledgeArea {id: 'policies'})
MERGE (admin)-[:CAN_QUERY]->(esg)
MERGE (admin)-[:CAN_QUERY]->(struct)
MERGE (admin)-[:CAN_QUERY]->(pol);

MATCH (compliance:Role {name: 'compliance_officer'})
MATCH (esg:KnowledgeArea {id: 'esg'})
MATCH (struct:KnowledgeArea {id: 'structured'})
MERGE (compliance)-[:CAN_QUERY]->(esg)
MERGE (compliance)-[:CAN_QUERY]->(struct);

MATCH (regular:Role {name: 'regular_office'})
MATCH (struct:KnowledgeArea {id: 'structured'})
MERGE (regular)-[:CAN_QUERY]->(struct);

MATCH (public_role:Role {name: 'public'})
MATCH (esg:KnowledgeArea {id: 'esg'})
MERGE (public_role)-[:CAN_QUERY]->(esg);

// ─────────────────────────────────────────
// SAMPLE USERS
// ─────────────────────────────────────────

MERGE (user_admin:User {
  user_id: 'admin_001',
  name: 'Admin User',
  department: 'IT',
  created_at: timestamp()
})
MERGE (user_compliance:User {
  user_id: 'compliance_001',
  name: 'Compliance Officer',
  department: 'Legal',
  created_at: timestamp()
})
MERGE (user_regular:User {
  user_id: 'regular_001',
  name: 'Regular Office',
  department: 'Sales',
  created_at: timestamp()
})
MERGE (user_public:User {
  user_id: 'public_001',
  name: 'Public User',
  department: 'External',
  created_at: timestamp()
});

// User → Role (HAS_ROLE assignments)
MATCH (user_admin:User {user_id: 'admin_001'})
MATCH (admin:Role {name: 'admin'})
MERGE (user_admin)-[:HAS_ROLE]->(admin);

MATCH (user_compliance:User {user_id: 'compliance_001'})
MATCH (compliance:Role {name: 'compliance_officer'})
MERGE (user_compliance)-[:HAS_ROLE]->(compliance);

MATCH (user_regular:User {user_id: 'regular_001'})
MATCH (regular:Role {name: 'regular_office'})
MERGE (user_regular)-[:HAS_ROLE]->(regular);

MATCH (user_public:User {user_id: 'public_001'})
MATCH (public_role:Role {name: 'public'})
MERGE (user_public)-[:HAS_ROLE]->(public_role);

// ─────────────────────────────────────────
// POLICIES (CAN_EDIT for admin/compliance)
// ─────────────────────────────────────────

MERGE (policy_data:Policy {
  id: 'policy_data_retention',
  name: 'Data Retention Policy',
  sensitivity: 'confidential'
})
MERGE (policy_access:Policy {
  id: 'policy_access_control',
  name: 'Access Control Policy',
  sensitivity: 'sensitive'
});

MATCH (admin:Role {name: 'admin'})
MATCH (compliance:Role {name: 'compliance_officer'})
MATCH (policy_data:Policy {id: 'policy_data_retention'})
MATCH (policy_access:Policy {id: 'policy_access_control'})
MERGE (admin)-[:CAN_EDIT]->(policy_data)
MERGE (admin)-[:CAN_EDIT]->(policy_access)
MERGE (compliance)-[:CAN_EDIT]->(policy_access);

// ─────────────────────────────────────────
// DOCUMENTS (categorized by sensitivity)
// ─────────────────────────────────────────

// Sample document structure linking to knowledge areas
MERGE (doc_esg:Document {
  id: 'doc_esg_report_2024',
  title: 'ESG Report 2024',
  sensitivity: 'sensitive',
  knowledge_area: 'esg'
})
MERGE (doc_policy:Document {
  id: 'doc_data_policy',
  title: 'Data Retention Policy',
  sensitivity: 'confidential',
  knowledge_area: 'policies'
})
MERGE (doc_product:Document {
  id: 'doc_product_catalog',
  title: 'Product Catalog',
  sensitivity: 'public',
  knowledge_area: 'structured'
});

MATCH (ka_esg:KnowledgeArea {id: 'esg'})
MATCH (ka_pol:KnowledgeArea {id: 'policies'})
MATCH (ka_struct:KnowledgeArea {id: 'structured'})
MATCH (doc_esg:Document {id: 'doc_esg_report_2024'})
MATCH (doc_policy:Document {id: 'doc_data_policy'})
MATCH (doc_product:Document {id: 'doc_product_catalog'})
MERGE (ka_esg)-[:CONTAINS]->(doc_esg)
MERGE (ka_pol)-[:CONTAINS]->(doc_policy)
MERGE (ka_struct)-[:CONTAINS]->(doc_product);

// Role → Document (CAN_VIEW permissions by document)
MATCH (admin:Role {name: 'admin'})
MATCH (doc_esg:Document {id: 'doc_esg_report_2024'})
MATCH (doc_policy:Document {id: 'doc_data_policy'})
MATCH (doc_product:Document {id: 'doc_product_catalog'})
MERGE (admin)-[:CAN_VIEW]->(doc_esg)
MERGE (admin)-[:CAN_VIEW]->(doc_policy)
MERGE (admin)-[:CAN_VIEW]->(doc_product);

MATCH (compliance:Role {name: 'compliance_officer'})
MATCH (doc_esg:Document {id: 'doc_esg_report_2024'})
MATCH (doc_policy:Document {id: 'doc_data_policy'})
MERGE (compliance)-[:CAN_VIEW]->(doc_esg)
MERGE (compliance)-[:CAN_VIEW]->(doc_policy);

MATCH (regular:Role {name: 'regular_office'})
MATCH (doc_product:Document {id: 'doc_product_catalog'})
MERGE (regular)-[:CAN_VIEW]->(doc_product);

// ─────────────────────────────────────────
// ENTITY & CHUNK LINKS (for queries)
// ─────────────────────────────────────────

MERGE (entity_compliance:Entity {name: 'Compliance', category: 'org'})
MERGE (entity_product:Entity {name: 'Product', category: 'business'});

MATCH (doc_esg:Document {id: 'doc_esg_report_2024'})
MATCH (entity_compliance:Entity {name: 'Compliance'})
MERGE (doc_esg)-[:MENTIONS]->(entity_compliance);

MATCH (doc_product:Document {id: 'doc_product_catalog'})
MATCH (entity_product:Entity {name: 'Product'})
MERGE (doc_product)-[:MENTIONS]->(entity_product);
