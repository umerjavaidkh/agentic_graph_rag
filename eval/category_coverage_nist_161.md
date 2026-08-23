# Category coverage — one question per query type

Document: **NIST SP 800-161r1** (`doc_nist_sp800_161`) — the hardest in the corpus:
887 nodes, 418 sections, 327 pages, 15 chapters, 88 tables, 39 figures.

Run 2026-08-23 against `graph_rag_vector_first` with the indexed-scoping fixes.
One question per category from the 28-category taxonomy. Verdicts assigned by
reading each answer against the document's known content.

| # | Category | s | Right doc | Verdict |
|---|---|---|---|---|
| 1 | Fact / Lookup | 15.4 | yes | correct |
| 2 | Definition | 20.7 | yes | correct |
| 3 | Entity / Attribute | 14.7 | yes | said not covered |
| 4 | Structural / Navigation | 15.0 | yes | wrong |
| 5 | List / Enumerative | 16.1 | yes | incomplete |
| 6 | Filtering / Selection | 21.4 | yes | correct |
| 7 | Aggregation / Count | 15.6 | yes | incomplete |
| 8 | Comparison | 22.3 | yes | correct |
| 9 | Temporal / Version | 17.7 | yes | said not covered |
| 10 | Multi-hop / Relational | 20.7 | yes | correct |
| 11 | Causal / Why | 15.1 | yes | correct |
| 12 | Thematic / Synthesis | 17.4 | yes | correct |
| 13 | Summarization | 19.0 | yes | wrong |
| 14 | Procedural / How-to | 22.8 | yes | correct |
| 15 | Instruction / Requirements | 21.9 | yes | correct |
| 16 | Conditional / Rule-based | 14.6 | yes | said not covered |
| 17 | Exception / Edge case | 20.8 | yes | said not covered |
| 18 | Numeric / Calculation | 15.7 | **no** (doc_nist_ir_8286) | answered from wrong document |
| 19 | Table / Structured-data | 25.2 | yes | correct |
| 20 | Chart / Figure | 16.3 | yes | correct |
| 21 | Cross-document | 22.3 | yes | correct |
| 22 | Cross-entity | 14.8 | yes | said not covered |
| 23 | Reference / Citation | 22.1 | yes | correct |
| 24 | Verification / Validation | 21.2 | yes | correct |
| 25 | Contradiction / Conflict | 14.4 | yes | said not covered |
| 26 | Recommendation / Decision support | 21.4 | yes | correct |
| 27 | Ambiguous / Underspecified | 5.0 | **no** (doc_irs_p3) | wrong |
| 28 | Unanswerable / Out-of-corpus | 5.8 | yes | correct |

**Correct 16/28 · right document 26/28 · median 17.7s**

## Answers

### 1. Fact / Lookup — correct
*What is the publication date of NIST SP 800-161r1?*

> The publication date of NIST SP 800-161r1 is May 5, 2022.

### 2. Definition — correct
*In NIST SP 800-161r1, what does C-SCRM mean?*

> In NIST SP 800-161r1, C-SCRM stands for Cyber Supply Chain Risk Management. It refers to the processes and practices that organizations implement to manage risks associated with the supply chain, particularly in relation to cybersecurity threats and vulnerabilities that may affect the integrity, availability, and confidentiality of products and services sourced from suppliers.

### 3. Entity / Attribute — said not covered
*Who published NIST SP 800-161r1 and under what authority?*

> This document does not cover the specific publisher or authority under which NIST SP 800-161r1 was published.

### 4. Structural / Navigation — wrong
*What does Section 1.1 of NIST SP 800-161r1 say?*

> This document does not cover the content of Section 1.1 of NIST SP 800-161r1.

### 5. List / Enumerative — incomplete
*List all the appendices in NIST SP 800-161r1.*

> The appendices in NIST SP 800-161r1 are as follows: - APPENDIX H: GLOSSARY - APPENDIX I: ACRONYMS - APPENDIX J: RESOURCES - RELATIONSHIP TO OTHER PROGRAMS AND PUBLICATIONS - NIST Publications - Regulatory and Legislative Guidance - Other U.S. Government Reports - Standards, Guidelines, and Best Practices

### 6. Filtering / Selection — correct
*Which C-SCRM controls in NIST SP 800-161r1 apply at the operational level?*

> The C-SCRM controls in NIST SP 800-161r1 that apply at the operational level (Level 3) include: - SA-9: External System Services - SA-5: System Documentation - SA-8: Security and Privacy Engineering Principles - SR-2: Supply Chain Risk Management Plan - SR-3: Supply Chain Controls and Processes - SR-10: Inspection of Systems or Components - SR-11: Component Authenticity - SR-12: Component Disposal These controls are designed to manage operational-level risk exposure resulting from ICT/OT-related products and services provided through the supply chain.

### 7. Aggregation / Count — incomplete
*How many critical success factors does NIST SP 800-161r1 identify?*

> NIST SP 800-161r1 identifies a non-exhaustive set of critical success factors for successfully addressing evolving cybersecurity risks throughout the supply chain. However, the document does not specify an exact number of these factors. It emphasizes the importance of engaging multiple internal processes and capabilities, communicating and collaborating across enterprise levels, and ensuring that all individuals within the enterprise understand their roles in managing cybersecurity risks.

### 8. Comparison — correct
*In NIST SP 800-161r1, how does C-SCRM differ from traditional information security?*

> The document outlines several key differences between Cyber Supply Chain Risk Management (C-SCRM) and traditional information security: - **Focus on Supply Chain**: C-SCRM specifically addresses risks associated with the supply chain, including the vulnerabilities and threats posed by suppliers and third-party service providers. Traditional information security typically focuses on protecting the organization's internal systems and data without as much emphasis on external supply chain risks. - **Multidisciplinary Approach**: C-SCRM requires a multidisciplinary approach that integrates various

### 9. Temporal / Version — said not covered
*What changed in NIST SP 800-161r1 compared with the original SP 800-161?*

> The document does not explicitly detail the changes made in NIST SP 800-161r1 compared to the original SP 800-161. It mentions that the revision builds upon concepts described in previous publications and aims to facilitate integration with existing enterprise-wide activities, as well as legislative developments following its initial release. However, specific changes or updates are not outlined in the retrieved sections.

### 10. Multi-hop / Relational — correct
*In NIST SP 800-161r1, which enterprise level is responsible for the C-SCRM strategy that the operational level implements?*

> In NIST SP 800-161r1, the enterprise level responsible for the C-SCRM strategy that the operational level implements is the **enterprise level** itself. This level defines the C-SCRM assumptions, constraints, risk appetite/tolerance, and priorities/trade-offs, which are then integrated into the operational level's implementation of the C-SCRM strategy. This is detailed in the section discussing the interrelationships among the risk management process steps, particularly in the context of the Frame, Assess, Respond, and Monitor steps of the Risk Management Process.

### 11. Causal / Why — correct
*Why was NIST SP 800-161r1 developed?*

> NIST SP 800-161r1 was developed to provide guidance on Cybersecurity Supply Chain Risk Management (C-SCRM) practices for systems and organizations. The document aims to facilitate the integration of supply chain risk management into existing enterprise-wide activities and to address the growing concerns regarding supply chain risks in the context of cybersecurity. It builds upon concepts described in various NIST and other publications, ensuring that organizations can protect their operations and assets against increasingly sophisticated threats. The publication also seeks to remain consistent

### 12. Thematic / Synthesis — correct
*What are the major supply chain risks discussed in NIST SP 800-161r1?*

> The major supply chain risks discussed in NIST SP 800-161, Rev. 1, include: 1. **Adversarial Threats**: Risks arising from malicious activities conducted by adversarial threat actors with intent and capability to cause harm. 2. **Non-Adversarial Risks**: Risks that are not necessarily linked to adversarial actions but can still impact the supply chain, such as operational failures or natural disasters. 3. **Critical Suppliers and Systems**: The compromise of critical suppliers or systems can lead to significant harm, necessitating prioritization in assessments. 4. **Vulnerabilities and Threats

### 13. Summarization — wrong
*Summarize Chapter 2 of NIST SP 800-161r1.*

> This document does not cover Chapter 2 of NIST SP 800-161r1. The retrieved sections primarily focus on specific categories such as anomaly detection, response planning, standards, guidelines, and best practices related to cybersecurity and supply chain risk management. If you need information on a specific aspect of Chapter 2, please provide more details.

### 14. Procedural / How-to — correct
*According to NIST SP 800-161r1, how should an organization implement a C-SCRM programme?*

> To implement a Cyber Supply Chain Risk Management (C-SCRM) program, NIST SP 800-161r1 outlines several strategic objectives and essential capabilities: 1. **Establish a Risk Management Hierarchy**: Implement a structured approach to risk management that integrates C-SCRM requirements into enterprise policies. 2. **Governance Structure**: Create an enterprise governance structure that incorporates C-SCRM requirements and ensures alignment with overall organizational objectives. 3. **Supplier Risk Assessment**: Define a comprehensive approach for assessing supplier risks, including the evaluatio

### 15. Instruction / Requirements — correct
*What does NIST SP 800-161r1 require an enterprise to document for C-SCRM?*

> NIST SP 800-161r1 requires an enterprise to document several key elements for Cyber Supply Chain Risk Management (C-SCRM), including: - **Consequences and Impact Assumptions**: Enterprises should refine and update their consequences and impact assumptions to reflect the role that the availability, confidentiality, and integrity of supplier-provided products or services have on the enterprise’s operations, assets, and individuals (Chunk 3). - **Criticality Analysis Procedures**: There should be a set of documented procedures to guide the enterprise’s criticality analysis across different levels

### 16. Conditional / Rule-based — said not covered
*According to NIST SP 800-161r1, if a supplier fails a criticality assessment, what should happen?*

> This document does not cover the specific actions that should be taken if a supplier fails a criticality assessment according to NIST SP 800-161r1.

### 17. Exception / Edge case — said not covered
*Does NIST SP 800-161r1 describe any exceptions to applying its C-SCRM controls?*

> The retrieved sections do not explicitly mention any exceptions to applying the C-SCRM controls outlined in NIST SP 800-161r1. The document primarily focuses on the implementation and guidance of these controls without detailing any exceptions.

### 18. Numeric / Calculation — answered from wrong document
*In the risk exposure framework of NIST SP 800-161r1, how is a risk exposure score calculated?*

> In the risk exposure framework of NIST SP 800-161r1, a risk exposure score is calculated by multiplying the impact of a risk event by its likelihood of occurrence. This is expressed as: **Risk Exposure = Impact x Likelihood** The document emphasizes that understanding both the likelihood of risk events and their potential impacts is critical for effective risk management. Each risk scenario should be assessed to determine these factors, which then contribute to the overall risk exposure score.

### 19. Table / Structured-data — correct
*What does the C-SCRM control summary table in Appendix B of NIST SP 800-161r1 contain?*

> The C-SCRM control summary table in Appendix B of NIST SP 800-161r1 contains a list of C-SCRM controls and maps them to their corresponding controls from [NIST SP 800-53, Rev. 5]. The table includes the following information for each control: - **Control Identifier**: A unique identifier for each control. - **Control (or Control Enhancement) Name**: The name of the control or enhancement. - **C-SCRM Baseline**: Indicates whether the control is part of the C-SCRM baseline. - **Flow Down Control**: Indicates if the control should flow down from prime contractors to sub-tier contractors. - **Leve

### 20. Chart / Figure — correct
*What does Figure 1 in NIST SP 800-161r1 show?*

> Figure 1-1 in NIST SP 800-161r1 illustrates the "Dimensions of C-SCRM" (Cyber Supply Chain Risk Management). It highlights various aspects that are critical for effective C-SCRM, including: - **Culture and Awareness**: The shared values, practices, goals, and attitudes within an organization that support successful C-SCRM. - **Security**: Ensuring confidentiality, integrity, and availability of information related to the supply chain. - **Suitability**: The appropriateness of the supply chain products and services for the enterprise's needs. - **Safety**: Ensuring that products and services do

### 21. Cross-document — correct
*How does NIST SP 800-161r1 relate to the NIST Cybersecurity Framework?*

> NIST SP 800-161r1, titled "Cybersecurity Supply Chain Risk Management Practices for Systems and Organizations," builds upon the concepts promoted within the NIST Cybersecurity Framework (CSF) Version 1.1. It tailors these concepts specifically for use within Cybersecurity Supply Chain Risk Management (C-SCRM). The relationship includes the following key points: - **Voluntary Guidance**: The CSF provides voluntary guidance based on existing standards, guidelines, and practices to help organizations manage and reduce cybersecurity risk. SP 800-161r1 leverages this guidance to enhance C-SCRM prac

### 22. Cross-entity — said not covered
*Which federal agencies or bodies are named in NIST SP 800-161r1?*

> The document does not explicitly name specific federal agencies or bodies in NIST SP 800-161r1. It discusses general guidance and frameworks applicable to federal enterprises but does not provide a list of named agencies.

### 23. Reference / Citation — correct
*Where does NIST SP 800-161r1 mention Executive Order 14028?*

> NIST SP 800-161r1 mentions Executive Order 14028 in several sections: 1. In the context of **Supplemental C-SCRM Guidance** regarding anti-counterfeit policies and procedures, it states that departments and agencies should refer to Appendix F to implement this guidance in accordance with Executive Order 14028, Improving the Nation’s Cybersecurity (Chunk 1). 2. The same reference to Executive Order 14028 is made in the **C-SCRM in Acquisition** section, emphasizing the importance of integrating C-SCRM considerations into acquisition activities and referring to Appendix F for implementation guid

### 24. Verification / Validation — correct
*Does NIST SP 800-161r1 actually require a C-SCRM plan, or only recommend one?*

> NIST SP 800-161r1 includes guidance that suggests the development of a Cyber Supply Chain Risk Management (C-SCRM) Plan of Action and Milestones (POA&M) as part of the authorization package. The document states that the C-SCRM POA&M should include tasks to be accomplished, resources required, milestones, and scheduled completion dates. This implies a strong recommendation for having a C-SCRM plan in place, particularly for addressing supply chain risks and ensuring that they are included in authorization decisions. While the document emphasizes the importance of including C-SCRM in authorizati

### 25. Contradiction / Conflict — said not covered
*Does NIST SP 800-161r1 contain any conflicting guidance about control applicability?*

> This document does not cover any conflicting guidance about control applicability in NIST SP 800-161r1.

### 26. Recommendation / Decision support — correct
*According to NIST SP 800-161r1, which C-SCRM control baseline applies to a high-impact system?*

> The C-SCRM control baseline that applies to a high-impact system is defined in the context of security control baselines, which include one for each system impact level (i.e., low-impact, moderate-impact, and high-impact). This information is referenced in the section discussing security control baselines, but specific details about the controls themselves are not provided in the retrieved sections. For further details on the specific controls applicable to high-impact systems, you may need to refer to the relevant sections of NIST SP 800-53, Rev. 5, which outlines the security controls for di

### 27. Ambiguous / Underspecified — wrong
*What is the deadline?*

> The deadline for filing your return, paying your tax, claiming a refund, and taking other actions with the IRS is extended in two steps. First, your deadline is extended for 180 days after the later of the following: 1. The last day you are in a combat zone, have qualifying service outside of the combat zone, or serve in a contingency operation (or the last day the area qualifies as a combat zone or the operation qualifies as a contingency operation). 2. The last day of any continuous qualified hospitalization for injury from service in the combat zone or contingency operation or while perform

### 28. Unanswerable / Out-of-corpus — correct
*What will NIST publish about supply chain risk in 2027?*

> This document does not cover what NIST will publish about supply chain risk in 2027.
