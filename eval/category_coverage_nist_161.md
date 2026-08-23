# Category coverage — one question per query type

Document: **NIST SP 800-161r1** (`doc_nist_sp800_161`) — hardest in the corpus:
887 nodes, 418 sections, 327 pages, 15 chapters, 88 tables, 39 figures.

Run 2026-08-23 against `graph_rag_vector_first`, after the plan-authoritative,
address-matching, honesty-gate and vector-backfill changes.

**Correct 22/28 · right document 27/28 · median 17.4s**

Progression across the day: 16/28 -> 19/28 (plan enforced, addresses matched)
-> 22/28 (vector coverage backfilled from 49% to 100%).

| # | Category | s | Right doc | Verdict |
|---|---|---|---|---|
| 1 | Fact / Lookup | 16.1 | yes | correct |
| 2 | Definition | 21.0 | yes | correct |
| 3 | Entity / Attribute | 14.5 | yes | correct |
| 4 | Structural / Navigation | 14.0 | yes | correct |
| 5 | List / Enumerative | 15.7 | yes | correct |
| 6 | Filtering / Selection | 22.1 | yes | correct |
| 7 | Aggregation / Count | 15.9 | yes | incomplete |
| 8 | Comparison | 22.6 | yes | correct |
| 9 | Temporal / Version | 17.3 | yes | said not covered |
| 10 | Multi-hop / Relational | 21.1 | yes | correct |
| 11 | Causal / Why | 15.5 | yes | correct |
| 12 | Thematic / Synthesis | 17.5 | yes | correct |
| 13 | Summarization | 18.6 | yes | correct |
| 14 | Procedural / How-to | 23.6 | yes | correct |
| 15 | Instruction / Requirements | 21.6 | yes | correct |
| 16 | Conditional / Rule-based | 14.6 | yes | said not covered |
| 17 | Exception / Edge case | 20.4 | yes | said not covered |
| 18 | Numeric / Calculation | 14.6 | yes | said not covered |
| 19 | Table / Structured-data | 25.3 | yes | correct |
| 20 | Chart / Figure | 15.7 | yes | correct |
| 21 | Cross-document | 21.4 | yes | correct |
| 22 | Cross-entity | 15.1 | yes | correct |
| 23 | Reference / Citation | 21.8 | yes | correct |
| 24 | Verification / Validation | 21.3 | yes | correct |
| 25 | Contradiction / Conflict | 15.2 | yes | said not covered |
| 26 | Recommendation / Decision support | 22.5 | yes | correct |
| 27 | Ambiguous / Underspecified | 3.2 | **no** (doc_arxiv_2608_02031) | correct |
| 28 | Unanswerable / Out-of-corpus | 8.3 | yes | correct |

## What the backfill fixed

Page nodes were 1.0% embedded and Region nodes 0%, so the dense channel could
not see 51% of the corpus. Three categories were failing purely on that:

| # | before | after |
|---|---|---|
| 3 Entity/Attribute | "does not cover the publisher" | names NIST |
| 22 Cross-entity | "does not explicitly name agencies" | NIST, OMB, FAR Council, FASC |
| 27 Ambiguous | answered from an unrelated IRS publication | declines |

## Remaining six

**Not built (4)** — `Temporal/Version` needs CanonicalSection and SUPERSEDES;
`Contradiction` and `Conditional` need cross-passage reasoning; `Aggregation`
needs counting over a retrieved set rather than reading a stated number.

**Retrieval (2)** — `Numeric/Calculation` and `Exception` report content missing
that the document contains. Both now fail honestly rather than inventing an
answer, which is the improvement; finding it is still open.

## Answers

### 1. Fact / Lookup — correct
*What is the publication date of NIST SP 800-161r1?*

> The publication date of NIST SP 800-161r1 is May 5, 2022.

### 2. Definition — correct
*In NIST SP 800-161r1, what does C-SCRM mean?*

> C-SCRM stands for Cyber Supply Chain Risk Management. It refers to the processes and practices that organizations implement to manage risks associated with the supply chain, particularly in relation to cybersecurity threats and vulnerabilities. This includes assessing and mitigating risks posed by suppliers, products, and services that are integral to an enterprise's operations. The concept emphasizes the importance of understanding and managing the cybersecurity risks that can arise from the supply chain, ensuring the integrity, availability, and confidentiality of the products and services a

### 3. Entity / Attribute — correct
*Who published NIST SP 800-161r1 and under what authority?*

> NIST SP 800-161r1 was published by the National Institute of Standards and Technology (NIST). The document is part of the NIST Special Publication series, which provides guidelines and standards for various aspects of information security and risk management.

### 4. Structural / Navigation — correct
*What does Section 1.1 of NIST SP 800-161r1 say?*

> Section 1.1 of NIST SP 800-161r1 outlines the purpose of the publication, which is to provide guidance on Cybersecurity Supply Chain Risk Management (C-SCRM). It describes C-SCRM as a systematic process for managing exposure to cybersecurity risks throughout the supply chain and developing appropriate response strategies, policies, processes, and procedures. The section emphasizes that the guidance is not one-size-fits-all and should be tailored to the unique size, resources, and risk circumstances of each enterprise. It also discusses the importance of adopting and maturing C-SCRM practices a

### 5. List / Enumerative — correct
*List all the appendices in NIST SP 800-161r1.*

> The appendices in NIST SP 800-161r1 are as follows: - APPENDIX H: GLOSSARY - APPENDIX I: ACRONYMS - APPENDIX J: RESOURCES - RELATIONSHIP TO OTHER PROGRAMS AND PUBLICATIONS - NIST Publications - Regulatory and Legislative Guidance - Other U.S. Government Reports - Standards, Guidelines, and Best Practices

### 6. Filtering / Selection — correct
*Which C-SCRM controls in NIST SP 800-161r1 apply at the operational level?*

> The C-SCRM controls in NIST SP 800-161r1 that apply at the operational level (Level 3) include: - SA-9: External System Services - SA-5: System Documentation - SA-8: Security and Privacy Engineering Principles - SR-2: Supply Chain Risk Management Plan - SR-3: Supply Chain Controls and Processes - SR-5: Acquisition Strategies, Tools, and Methods - SR-6: Supplier Assessments and Reviews - SR-7: Supply Chain Operations Security - SR-8: Notification Agreements - SR-10: Inspection of Systems or Components - SR-11: Component Authenticity - SR-12: Component Disposal - SR-13: Supplier Inventory These 

### 7. Aggregation / Count — incomplete
*How many critical success factors does NIST SP 800-161r1 identify?*

> The document does not specify a numerical count of critical success factors identified in NIST SP 800-161r1. It states that the publication highlights a non-exhaustive set of factors that contribute to an enterprise’s successful execution of C-SCRM, indicating that critical success factors are fluid and will evolve over time as the environment and the enterprise’s own capability advances.

### 8. Comparison — correct
*In NIST SP 800-161r1, how does C-SCRM differ from traditional information security?*

> The document outlines several key differences between Cyber Supply Chain Risk Management (C-SCRM) and traditional information security: 1. **Focus on Supply Chain**: C-SCRM specifically addresses risks associated with the supply chain, including the vulnerabilities and threats that arise from suppliers, developers, and system integrators. Traditional information security typically focuses on protecting the organization's internal systems and data without as much emphasis on external supply chain risks. 2. **Multidisciplinary Approach**: C-SCRM requires a multidisciplinary approach that integra

### 9. Temporal / Version — said not covered
*What changed in NIST SP 800-161r1 compared with the original SP 800-161?*

> This document does not cover the specific changes made in NIST SP 800-161r1 compared with the original SP 800-161.

### 10. Multi-hop / Relational — correct
*In NIST SP 800-161r1, which enterprise level is responsible for the C-SCRM strategy that the operational level implements?*

> In NIST SP 800-161r1, the enterprise level responsible for the C-SCRM strategy that the operational level implements is the **enterprise level** itself. This level defines the C-SCRM assumptions, constraints, risk appetite/tolerance, and priorities/trade-offs, which are then integrated into the operational level's activities. This is outlined in the section discussing the interrelationships among the risk management process steps, including the Frame, Assess, Respond, and Monitor steps.

### 11. Causal / Why — correct
*Why was NIST SP 800-161r1 developed?*

> NIST SP 800-161r1 was developed to provide guidance on Cybersecurity Supply Chain Risk Management (C-SCRM) practices for systems and organizations. It aims to assist enterprises in defining the purpose and objectives for supply chain risk assessments, outlining key information required for understanding systems, operations, and supporting architecture, and establishing policies and procedures to address supply chain concerns, particularly in relation to personally identifiable information (PII) processing and transparency. This publication is part of NIST's responsibilities under the Federal I

### 12. Thematic / Synthesis — correct
*What are the major supply chain risks discussed in NIST SP 800-161r1?*

> The major supply chain risks discussed in NIST SP 800-161, Rev. 1, include: - **Adversarial Threats**: Risks arising from the intent and capability of adversarial threat actors to conduct malicious activities or cause harm within the supply chain. - **Non-Adversarial Risks**: Broader risks that may not be directly linked to adversarial actions but still pose significant threats to the supply chain. - **Vulnerabilities**: Weaknesses within the supply chain that could be exploited by adversaries or lead to failures in security. - **Dependency Risks**: Risks associated with reliance on specific s

### 13. Summarization — correct
*Summarize Chapter 2 of NIST SP 800-161r1.*

> Chapter 2 of NIST SP 800-161r1 focuses on the integration of Cybersecurity Supply Chain Risk Management (C-SCRM) into enterprise-wide risk management. It outlines a continuous and iterative risk management process that includes the following key steps: 1. **Frame Risk**: Establish the context for risk-based decisions regarding the enterprise's information and communications technology and services. 2. **Assess Risk**: Review criticality, threats, vulnerabilities, likelihood, and impact. 3. **Respond to Risk**: Implement mitigation controls based on risk assessment findings. 4. **Monitor Risk**

### 14. Procedural / How-to — correct
*According to NIST SP 800-161r1, how should an organization implement a C-SCRM programme?*

> To implement a Cyber Supply Chain Risk Management (C-SCRM) program, NIST SP 800-161r1 outlines several strategic objectives and essential capabilities: 1. **Establish a Risk Management Hierarchy**: Implement a structured approach to risk management that integrates C-SCRM requirements into enterprise policies. 2. **Enterprise Governance Structure**: Create a governance framework that incorporates C-SCRM requirements and ensures alignment across the organization. 3. **Supplier Risk Assessment**: Define a systematic approach for assessing risks associated with suppliers, including their criticali

### 15. Instruction / Requirements — correct
*What does NIST SP 800-161r1 require an enterprise to document for C-SCRM?*

> NIST SP 800-161r1 requires an enterprise to document the following for Cyber Supply Chain Risk Management (C-SCRM): - **Supply Chain Risk Assessments (SCRAs)**: Agencies must conduct and prioritize SCRAs to assess the supply chain risk posed by the acquisition and use of covered articles. - **Criticality Analysis Procedures**: Enterprises should define documented procedures to guide criticality analysis across different levels. - **Risk Management Hierarchy and Approach**: Establishing a risk management hierarchy and approach that integrates C-SCRM requirements into enterprise policies. - **Su

### 16. Conditional / Rule-based — said not covered
*According to NIST SP 800-161r1, if a supplier fails a criticality assessment, what should happen?*

> This document does not cover the specific actions that should be taken if a supplier fails a criticality assessment according to NIST SP 800-161r1.

### 17. Exception / Edge case — said not covered
*Does NIST SP 800-161r1 describe any exceptions to applying its C-SCRM controls?*

> The retrieved sections do not explicitly mention any exceptions to applying the C-SCRM controls in NIST SP 800-161r1. The document primarily focuses on the implementation and requirements of the controls without detailing any exceptions.

### 18. Numeric / Calculation — said not covered
*In the risk exposure framework of NIST SP 800-161r1, how is a risk exposure score calculated?*

> This document does not cover the specific calculation method for a risk exposure score within the risk exposure framework of NIST SP 800-161r1.

### 19. Table / Structured-data — correct
*What does the C-SCRM control summary table in Appendix B of NIST SP 800-161r1 contain?*

> The C-SCRM control summary table in Appendix B of NIST SP 800-161r1 contains a list of C-SCRM controls and maps them to their corresponding controls from [NIST SP 800-53, Rev. 5]. The table includes the following information for each control: - **Control Identifier**: A unique identifier for each control. - **Control (or Control Enhancement) Name**: The name of the control or enhancement. - **C-SCRM Baseline**: Indicates whether the control is part of the C-SCRM baseline. - **Flow Down Control**: Indicates if the control should flow down from prime contractors to sub-tier contractors. - **Leve

### 20. Chart / Figure — correct
*What does Figure 1 in NIST SP 800-161r1 show?*

> Figure 1-1 in NIST SP 800-161r1 illustrates the "Dimensions of C-SCRM" (Cybersecurity Supply Chain Risk Management). It highlights various aspects that are critical for effective C-SCRM, including: - **Culture and Awareness**: The shared values, practices, goals, and attitudes of the organization that support successful C-SCRM. - **Security**: Ensuring confidentiality, integrity, and availability of information related to the supply chain. - **Suitability**: The appropriateness of the supply chain products and services for the enterprise's needs. - **Safety**: Ensuring that products and servic

### 21. Cross-document — correct
*How does NIST SP 800-161r1 relate to the NIST Cybersecurity Framework?*

> NIST SP 800-161r1 builds on concepts promoted within the NIST Cybersecurity Framework (CSF) Version 1.1. It tailors these concepts specifically for use within Cybersecurity Supply Chain Risk Management (C-SCRM). The CSF provides voluntary guidance based on existing standards, guidelines, and practices that help organizations manage and reduce cybersecurity risk. Additionally, it is designed to foster risk and cybersecurity management communications among both internal and external organizational stakeholders. This relationship allows NIST SP 800-161r1 to inherit many of the foundational concep

### 22. Cross-entity — correct
*Which federal agencies or bodies are named in NIST SP 800-161r1?*

> The federal agencies and bodies named in NIST SP 800-161r1 include: - National Institute of Standards and Technology (NIST) - Office of Management and Budget (OMB) - Federal Acquisition Regulatory (FAR) Council - Federal Acquisition Security Council (FASC) These entities are mentioned in the context of developing guidelines and standards for cybersecurity supply chain risk management.

### 23. Reference / Citation — correct
*Where does NIST SP 800-161r1 mention Executive Order 14028?*

> NIST SP 800-161r1 mentions Executive Order 14028 in several sections, specifically: 1. **Chunk 1**: It states that departments and agencies should refer to Appendix F to implement guidance in accordance with Executive Order 14028, Improving the Nation’s Cybersecurity. 2. **Chunk 2**: This chunk also references the need for enterprises to refer to Appendix F to implement guidance in accordance with Executive Order 14028. 3. **Chunk 3**: Similar to the previous chunks, it mentions that departments and agencies should refer to Appendix F to implement this guidance in accordance with Executive Ord

### 24. Verification / Validation — correct
*Does NIST SP 800-161r1 actually require a C-SCRM plan, or only recommend one?*

> NIST SP 800-161r1 includes guidance that suggests the development of a Cyber Supply Chain Risk Management (C-SCRM) Plan of Action and Milestones (POA&M) as part of the authorization package. The document states that the C-SCRM POA&M should include tasks to be accomplished, resources required, milestones, and scheduled completion dates. This implies a strong recommendation for organizations to create a C-SCRM plan to effectively manage supply chain risks. However, it does not explicitly state that a C-SCRM plan is a mandatory requirement. Instead, it emphasizes the importance of including C-SCR

### 25. Contradiction / Conflict — said not covered
*Does NIST SP 800-161r1 contain any conflicting guidance about control applicability?*

> This document does not cover any conflicting guidance about control applicability.

### 26. Recommendation / Decision support — correct
*According to NIST SP 800-161r1, which C-SCRM control baseline applies to a high-impact system?*

> The C-SCRM control baseline that applies to a high-impact system is defined in the context of security control baselines, which include one for each system impact level (i.e., low-impact, moderate-impact, and high-impact). This information is referenced in the section discussing security control baselines, indicating that specific controls are tailored to meet the requirements of high-impact systems. For further details, you may refer to the relevant sections in the document that outline the specific controls and their applicability to high-impact systems.

### 27. Ambiguous / Underspecified — correct
*What is the deadline?*

> This document does not cover the specific deadline.

### 28. Unanswerable / Out-of-corpus — correct
*What will NIST publish about supply chain risk in 2027?*

> This document does not cover what NIST will publish about supply chain risk in 2027.
