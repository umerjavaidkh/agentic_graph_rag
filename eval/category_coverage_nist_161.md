# Category coverage — one question per query type

Document: **NIST SP 800-161r1** (`doc_nist_sp800_161`) — the hardest in the corpus:
887 nodes, 418 sections, 327 pages, 15 chapters, 88 tables, 39 figures.

Run 2026-08-23 against `graph_rag_vector_first`, after the plan-authoritative
and address-matching fixes. One question per category from the 28-category
taxonomy. Verdicts assigned by reading each answer against the document.

**Correct 19/28 (was 16/28) · right document 26/28 · median 17.5s**

| # | Category | s | Right doc | Verdict |
|---|---|---|---|---|
| 1 | Fact / Lookup | 16.0 | yes | correct |
| 2 | Definition | 21.4 | yes | correct |
| 3 | Entity / Attribute | 14.9 | yes | said not covered |
| 4 | Structural / Navigation | 14.9 | yes | correct |
| 5 | List / Enumerative | 16.7 | yes | correct |
| 6 | Filtering / Selection | 21.9 | yes | correct |
| 7 | Aggregation / Count | 15.3 | yes | incomplete |
| 8 | Comparison | 22.4 | yes | correct |
| 9 | Temporal / Version | 17.5 | yes | said not covered |
| 10 | Multi-hop / Relational | 21.3 | yes | correct |
| 11 | Causal / Why | 15.0 | yes | correct |
| 12 | Thematic / Synthesis | 17.5 | yes | correct |
| 13 | Summarization | 19.6 | yes | correct |
| 14 | Procedural / How-to | 22.7 | yes | correct |
| 15 | Instruction / Requirements | 21.3 | yes | correct |
| 16 | Conditional / Rule-based | 14.6 | yes | said not covered |
| 17 | Exception / Edge case | 21.2 | yes | said not covered |
| 18 | Numeric / Calculation | 15.5 | **no** (doc_nist_ir_8286) | answered from wrong document |
| 19 | Table / Structured-data | 25.4 | yes | correct |
| 20 | Chart / Figure | 16.0 | yes | correct |
| 21 | Cross-document | 21.6 | yes | correct |
| 22 | Cross-entity | 14.9 | yes | said not covered |
| 23 | Reference / Citation | 21.7 | yes | correct |
| 24 | Verification / Validation | 21.6 | yes | correct |
| 25 | Contradiction / Conflict | 14.4 | yes | said not covered |
| 26 | Recommendation / Decision support | 22.1 | yes | correct |
| 27 | Ambiguous / Underspecified | 4.6 | **no** (doc_irs_p3) | wrong |
| 28 | Unanswerable / Out-of-corpus | 6.0 | yes | correct |

## Remaining failures, by cause

**Not built** — `Temporal/Version` (no CanonicalSection or SUPERSEDES),
`Cross-entity` (Entity has 2 nodes), `Contradiction` and `Conditional`
(need cross-passage reasoning).

**Recall** — `Entity/Attribute` and `Cross-entity` report content missing
that the document contains. Page nodes are 1.0% embedded and Region nodes 0%,
so the dense channel cannot see 51% of the corpus.

**Honesty** — `Numeric/Calculation` answered from `doc_nist_ir_8286` and
attributed the formula to 800-161. `Ambiguous/Underspecified` answered from an
arbitrary IRS publication instead of asking which document was meant; the
ambiguity gate only fires on a close runner-up, and a question naming no
document at all is a different signal that is not yet detected.

## Answers

### 1. Fact / Lookup — correct
*What is the publication date of NIST SP 800-161r1?*

> The publication date of NIST SP 800-161r1 is May 5, 2022.

### 2. Definition — correct
*In NIST SP 800-161r1, what does C-SCRM mean?*

> C-SCRM stands for Cyber Supply Chain Risk Management. It focuses on managing risks associated with the supply chain of products and services, particularly in the context of cybersecurity. The document emphasizes the importance of integrating C-SCRM practices into enterprise risk management processes to address potential vulnerabilities and threats that may arise from suppliers and service providers.

### 3. Entity / Attribute — said not covered
*Who published NIST SP 800-161r1 and under what authority?*

> This document does not cover the specific publisher or authority under which NIST SP 800-161r1 was published.

### 4. Structural / Navigation — correct
*What does Section 1.1 of NIST SP 800-161r1 say?*

> Section 1.1 of NIST SP 800-161r1 is titled "Purpose." It outlines the purpose of the Cybersecurity Supply Chain Risk Management (C-SCRM) guidance, which is to provide a systematic process for managing exposure to cybersecurity risks throughout the supply chain. The section emphasizes the importance of developing appropriate response strategies, policies, processes, and procedures to help enterprises identify, assess, select, and implement risk management processes and mitigating controls. Key points from this section include: - The guidance is not a one-size-fits-all solution; it should be tai

### 5. List / Enumerative — correct
*List all the appendices in NIST SP 800-161r1.*

> The appendices in NIST SP 800-161r1 are as follows: - APPENDIX A: C-SCRM SECURITY CONTROLS - APPENDIX B: C-SCRM CONTROL SUMMARY - APPENDIX C: RISK EXPOSURE FRAMEWORK - APPENDIX D: C-SCRM TEMPLATES - APPENDIX E: FASCSA - APPENDIX F: RESPONSE TO EXECUTIVE ORDER 14028’s CALL TO PUBLISH GUIDELINES FOR ENHANCING SOFTWARE SUPPLY CHAIN SECURITY - APPENDIX G: C-SCRM ACTIVITIES IN THE RISK MANAGEMENT PROCESS - APPENDIX H: GLOSSARY - APPENDIX I: ACRONYMS - APPENDIX J: RESOURCES

### 6. Filtering / Selection — correct
*Which C-SCRM controls in NIST SP 800-161r1 apply at the operational level?*

> The C-SCRM controls in NIST SP 800-161r1 that apply at the operational level (Level 3) include: - SA-9: External System Services - SA-5: System Documentation - SA-8: Security and Privacy Engineering Principles - SR-2: Supply Chain Risk Management Plan - SR-3: Supply Chain Controls and Processes - SR-10: Inspection of Systems or Components - SR-11: Component Authenticity - SR-12: Component Disposal These controls are designed to manage operational-level risk exposure resulting from ICT/OT-related products and services provided through the supply chain.

### 7. Aggregation / Count — incomplete
*How many critical success factors does NIST SP 800-161r1 identify?*

> NIST SP 800-161r1 identifies a non-exhaustive set of critical success factors for successfully addressing evolving cybersecurity risks throughout the supply chain. However, the document does not specify an exact number of these factors. It emphasizes the importance of engaging multiple internal processes and capabilities, communicating and collaborating across enterprise levels, and ensuring that all individuals within the enterprise understand their role in managing cybersecurity risks.

### 8. Comparison — correct
*In NIST SP 800-161r1, how does C-SCRM differ from traditional information security?*

> The document outlines several key differences between Cyber Supply Chain Risk Management (C-SCRM) and traditional information security: 1. **Focus on Supply Chain**: C-SCRM specifically addresses risks associated with the supply chain, including the vulnerabilities and threats posed by suppliers, developers, and system integrators. Traditional information security typically focuses on protecting the organization's internal systems and data without as much emphasis on external supply chain factors. 2. **Holistic Approach**: C-SCRM takes a more holistic view by considering the entire lifecycle o

### 9. Temporal / Version — said not covered
*What changed in NIST SP 800-161r1 compared with the original SP 800-161?*

> This document does not cover the specific changes made in NIST SP 800-161r1 compared with the original SP 800-161.

### 10. Multi-hop / Relational — correct
*In NIST SP 800-161r1, which enterprise level is responsible for the C-SCRM strategy that the operational level implements?*

> In NIST SP 800-161r1, the enterprise level responsible for the C-SCRM strategy that the operational level implements is the **enterprise level** itself. This level defines the C-SCRM assumptions, constraints, risk appetite/tolerance, and priorities/trade-offs, which are then integrated into the operational level's activities. This is outlined in the section discussing the interrelationships among the risk management process steps, particularly in the context of defining and tailoring C-SCRM strategies and policies.

### 11. Causal / Why — correct
*Why was NIST SP 800-161r1 developed?*

> NIST SP 800-161r1 was developed to provide guidance on Cybersecurity Supply Chain Risk Management (C-SCRM) practices for systems and organizations. The document aims to facilitate the integration of supply chain risk management into existing enterprise-wide activities and to address the increasing sophistication of threats faced by organizations. It builds upon concepts described in various NIST and other publications, ensuring that enterprises can establish risk-based information security programs to protect their operations and assets. The publication is intended to help organizations manage

### 12. Thematic / Synthesis — correct
*What are the major supply chain risks discussed in NIST SP 800-161r1?*

> The major supply chain risks discussed in NIST SP 800-161, Rev. 1, include both adversarial and non-adversarial-related risks. The document emphasizes the importance of assessing risks that arise from: 1. **Adversarial Threats**: Risks stemming from malicious activities conducted by threat actors with intent and capability to cause harm. 2. **Non-Adversarial Risks**: Broader risks that may not involve malicious intent but can still impact the supply chain, such as operational failures or natural disasters. Additionally, the document outlines various baseline risk factors that agencies should c

### 13. Summarization — correct
*Summarize Chapter 2 of NIST SP 800-161r1.*

> Chapter 2 of NIST SP 800-161r1 focuses on the integration of Cybersecurity Supply Chain Risk Management (C-SCRM) into enterprise-wide risk management. It outlines a continuous and iterative risk management process that includes the following steps: 1. **Frame Risk**: Establish the context for risk-based decisions regarding the enterprise’s information and communications technology and services. 2. **Assess Risk**: Review criticality, threats, vulnerabilities, likelihood, and impact. 3. **Respond to Risk**: Implement mitigation controls based on risk assessment findings. 4. **Monitor Risk**: Co

### 14. Procedural / How-to — correct
*According to NIST SP 800-161r1, how should an organization implement a C-SCRM programme?*

> To implement a Cyber Supply Chain Risk Management (C-SCRM) program, NIST SP 800-161r1 outlines several strategic objectives and essential capabilities: 1. **Establish a Risk Management Hierarchy**: Implement a structured risk management approach that integrates C-SCRM requirements into enterprise policies. 2. **Governance Structure**: Create an enterprise governance structure that incorporates C-SCRM requirements and ensures alignment with overall organizational policies. 3. **Supplier Risk Assessment**: Define a comprehensive supplier risk assessment approach to evaluate and manage risks asso

### 15. Instruction / Requirements — correct
*What does NIST SP 800-161r1 require an enterprise to document for C-SCRM?*

> NIST SP 800-161r1 requires an enterprise to document several key aspects for Cyber Supply Chain Risk Management (C-SCRM), including: - **Supplier Risk Assessment Approach**: Establishing a defined approach for assessing risks associated with suppliers. - **Consequences and Impact Assumptions**: Documenting the consequences and impact assumptions related to the availability, confidentiality, and integrity of supplier-provided products or services. - **Criticality Analysis Procedures**: Defining procedures for conducting criticality analysis across different levels of the enterprise. - **Risk Ma

### 16. Conditional / Rule-based — said not covered
*According to NIST SP 800-161r1, if a supplier fails a criticality assessment, what should happen?*

> This document does not cover the specific actions that should be taken if a supplier fails a criticality assessment according to NIST SP 800-161r1.

### 17. Exception / Edge case — said not covered
*Does NIST SP 800-161r1 describe any exceptions to applying its C-SCRM controls?*

> The retrieved sections do not explicitly mention any exceptions to applying the C-SCRM controls in NIST SP 800-161r1. The document primarily focuses on the implementation and guidance of these controls without detailing any exceptions.

### 18. Numeric / Calculation — answered from wrong document
*In the risk exposure framework of NIST SP 800-161r1, how is a risk exposure score calculated?*

> In the risk exposure framework of NIST SP 800-161r1, a risk exposure score is calculated by multiplying the impact of a risk event by its likelihood of occurrence. This is expressed as: **Risk Exposure = Impact x Likelihood** This calculation helps organizations assess the potential financial and operational consequences of risks, allowing them to prioritize risk management efforts effectively.

### 19. Table / Structured-data — correct
*What does the C-SCRM control summary table in Appendix B of NIST SP 800-161r1 contain?*

> The C-SCRM control summary table in Appendix B of NIST SP 800-161r1 contains a list of C-SCRM controls and maps them to their corresponding controls from [NIST SP 800-53, Rev. 5]. It includes the following information: - **Control Identifier**: A unique identifier for each control. - **Control (or Control Enhancement) Name**: The name of the control or enhancement. - **C-SCRM Baseline**: Indicates whether the control is part of the C-SCRM baseline. - **Flow Down Control**: Indicates if the control should flow down from prime contractors to sub-tier contractors. - **Levels**: Specifies the ente

### 20. Chart / Figure — correct
*What does Figure 1 in NIST SP 800-161r1 show?*

> Figure 1-1 in NIST SP 800-161r1 illustrates the "Dimensions of C-SCRM" (Cyber Supply Chain Risk Management). It highlights several key dimensions that are critical for effective C-SCRM, including: - **Culture and Awareness**: The shared values, practices, goals, and attitudes within an organization that support successful C-SCRM. - **Security**: Ensuring the confidentiality, integrity, and availability of information related to the supply chain. - **Suitability**: The appropriateness of the supply chain products and services for the enterprise's needs. - **Safety**: Ensuring that products and 

### 21. Cross-document — correct
*How does NIST SP 800-161r1 relate to the NIST Cybersecurity Framework?*

> NIST SP 800-161r1 builds on the concepts promoted within the NIST Cybersecurity Framework (CSF) Version 1.1. It tailors these concepts specifically for use within Cybersecurity Supply Chain Risk Management (C-SCRM). The relationship allows NIST SP 800-161r1 to inherit many of the frameworks, concepts, and methodologies established in the CSF, thereby enhancing the guidance for organizations to better manage and reduce cybersecurity risks associated with their supply chains. The CSF is designed to foster risk and cybersecurity management communications among both internal and external organizat

### 22. Cross-entity — said not covered
*Which federal agencies or bodies are named in NIST SP 800-161r1?*

> The document does not explicitly name specific federal agencies or bodies in NIST SP 800-161r1. It discusses general guidance and frameworks applicable to federal enterprises but does not provide a list of named agencies.

### 23. Reference / Citation — correct
*Where does NIST SP 800-161r1 mention Executive Order 14028?*

> NIST SP 800-161r1 mentions Executive Order 14028 in several sections: 1. In the context of **Supplemental C-SCRM Guidance** related to anti-counterfeit policies and procedures, it states that departments and agencies should refer to Appendix F to implement this guidance in accordance with Executive Order 14028, Improving the Nation’s Cybersecurity. 2. The same reference to Executive Order 14028 is made in the **Vulnerability Monitoring and Scanning** section, where it advises departments and agencies to refer to Appendix F for implementation guidance. 3. Additionally, in the **Developer Config

### 24. Verification / Validation — correct
*Does NIST SP 800-161r1 actually require a C-SCRM plan, or only recommend one?*

> NIST SP 800-161r1 includes guidance that suggests the development of a Cyber Supply Chain Risk Management (C-SCRM) Plan of Action and Milestones (POA&M) as part of the authorization package. It emphasizes that authorizing officials should include C-SCRM in authorization decisions and that the C-SCRM POA&M should include tasks to be accomplished, resources required, and milestones for completion. This indicates a strong recommendation for a C-SCRM plan, but it does not explicitly state that it is a requirement. Thus, it can be concluded that NIST SP 800-161r1 recommends the creation of a C-SCRM

### 25. Contradiction / Conflict — said not covered
*Does NIST SP 800-161r1 contain any conflicting guidance about control applicability?*

> This document does not cover any conflicting guidance about control applicability.

### 26. Recommendation / Decision support — correct
*According to NIST SP 800-161r1, which C-SCRM control baseline applies to a high-impact system?*

> The C-SCRM control baseline that applies to a high-impact system is defined in the context of security control baselines for different impact levels, including low-impact, moderate-impact, and high-impact systems. This information is typically detailed in the section discussing security control baselines, but specific control identifiers or names for high-impact systems are not explicitly listed in the retrieved sections. For further details, you may refer to the relevant sections in the document that outline the control baselines for high-impact systems.

### 27. Ambiguous / Underspecified — wrong
*What is the deadline?*

> The deadline for filing your return, paying your tax, claiming a refund, and taking other actions with the IRS is extended in two steps: 1. Your deadline is extended for 180 days after the later of: - The last day you are in a combat zone, have qualifying service outside of the combat zone, or serve in a contingency operation. - The last day of any continuous qualified hospitalization for injury from service in the combat zone or contingency operation. 2. In addition to the 180 days, your deadline is extended by the number of days that were left for you to take the action with the IRS when you

### 28. Unanswerable / Out-of-corpus — correct
*What will NIST publish about supply chain risk in 2027?*

> This document does not cover what NIST will publish about supply chain risk in 2027.
