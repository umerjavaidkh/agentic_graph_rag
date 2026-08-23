# Category coverage — verified measurement

Document: **NIST SP 800-161r1** (`doc_nist_sp800_161`), 887 nodes.
Run 2026-08-23, `graph_rag_vector_first`, after all fixes.

## Verification status

Earlier tables in this file scored by reading the answers. That was wrong
four times: #9 asked about a document not in the corpus, #17 was graded by
keyword presence, #16 rested on an unchecked premise. Below, a claim of
correct means a check was actually run against the document, and anything
not checked says so.

| status | count |
|---|---|
| VERIFIED CORRECT | 7 |
| CORRECT REFUSAL | 5 |
| VERIFIED WRONG | 1 |
| NOT VERIFIED | 14 |
| UNVERIFIABLE | 1 |

**Verified correct 12/28 · verified wrong 1/28 · not established 15/28**

| # | Category | Status | Evidence |
|---|---|---|---|
| 1 | Fact / Lookup | VERIFIED CORRECT | 'Publication Date(s) | May 5, 2022' found in doc |
| 2 | Definition | NOT VERIFIED | read as plausible; definition not checked against doc |
| 3 | Entity / Attribute | VERIFIED CORRECT | NIST named as publisher; found after backfill |
| 4 | Structural / Navigation | VERIFIED CORRECT | Section 1.1 title is 'Purpose'; matches answer |
| 5 | List / Enumerative | VERIFIED CORRECT | all 10 appendices A-J listed; matches hierarchy |
| 6 | Filtering / Selection | NOT VERIFIED | SA-9 (10 nodes), 'Level 3' (86) exist; full claim unchecked |
| 7 | Aggregation / Count | VERIFIED WRONG | chapter 3 has sections 3.1-3.6 = six factors; answer said no count given |
| 8 | Comparison | NOT VERIFIED | read as plausible; not checked |
| 9 | Temporal / Version | CORRECT REFUSAL | original SP 800-161 is not in the corpus at all |
| 10 | Multi-hop / Relational | NOT VERIFIED | read as plausible; not checked |
| 11 | Causal / Why | NOT VERIFIED | read as plausible; not checked |
| 12 | Thematic / Synthesis | NOT VERIFIED | read as plausible; not checked |
| 13 | Summarization | VERIFIED CORRECT | Chapter 2 is 'INTEGRATION OF C-SCRM...'; matches |
| 14 | Procedural / How-to | NOT VERIFIED | read as plausible; not checked |
| 15 | Instruction / Requirements | NOT VERIFIED | read as plausible; not checked |
| 16 | Conditional / Rule-based | CORRECT REFUSAL | 'criticality assessment' appears only as input to response actions, never pass/fail |
| 17 | Exception / Edge case | CORRECT REFUSAL | all 14 'exception' hits incidental (reporting, 'exceptional time', export licence) |
| 18 | Numeric / Calculation | VERIFIED CORRECT | Table C-1 exists; impact x likelihood confirmed |
| 19 | Table / Structured-data | NOT VERIFIED | Appendix B exists; table contents not checked field by field |
| 20 | Chart / Figure | NOT VERIFIED | Figure 1-1 exists; its content not checked |
| 21 | Cross-document | NOT VERIFIED | read as plausible; not checked |
| 22 | Cross-entity | VERIFIED CORRECT | NIST, OMB, FAR Council, FASC — agencies are named in doc |
| 23 | Reference / Citation | NOT VERIFIED | EO 14028 appears (Appendix F is titled for it); passages not checked |
| 24 | Verification / Validation | NOT VERIFIED | read as plausible; not checked |
| 25 | Contradiction / Conflict | UNVERIFIABLE | would require me to do the contradiction analysis to know ground truth |
| 26 | Recommendation / Decision support | NOT VERIFIED | read as plausible; not checked |
| 27 | Ambiguous / Underspecified | CORRECT REFUSAL | query names no document; declining is the correct behaviour |
| 28 | Unanswerable / Out-of-corpus | CORRECT REFUSAL | asks about a 2027 publication; cannot be in any corpus |