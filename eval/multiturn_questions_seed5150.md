# Multi-turn question set — seed 5150

Eight documents, one conversation each, three turns per conversation.

- **T1** names the document — must resolve it
- **T2** names nothing — must STAY on the same document
- **T3** names a DIFFERENT document — must SWITCH to it

T2 and T3 fail in opposite ways, so a system cannot pass both by always
trusting the thread or always ignoring it.

Reproduce: `python scripts/eval_multiturn.py --docs 10 --seed 5150`

Result: **24/24 right document** (T1 8/8, T2 8/8, T3 8/8), 23/24 answered.

## Documents

- `arxiv_2608_17360` (`doc_arxiv_2608_17360`)
- `arxiv_2608_17214` (`doc_arxiv_2608_17214`)
- `irs_p907` (`doc_irs_p907`)
- `arxiv_2608_18268` (`doc_arxiv_2608_18268`)
- `arxiv_2608_01192` (`doc_arxiv_2608_01192`)
- `arxiv_2608_18149` (`doc_arxiv_2608_18149`)
- `nist_sp800_161` (`doc_nist_sp800_161`)
- `arxiv_2608_16043` (`doc_arxiv_2608_16043`)

## Questions

### arxiv_2608_17360

**1. T1 names it**

> In the document "arxiv_2608_17360", fill in the blank with the exact value from the text: "We evaluate on two datasets: the  ______  standard text behaviors from HarmBench (Mazeika et al., 2024) and the 100 harmful requests from JailbreakBench (Chao et al., 2024)."

expected: `200` · result: doc=OK, answered=yes

**2. T2 follow-up**

> Fill in the blank with the exact value from the source: "For stochastic repeated-sampling and LLM-driven attacks, we report ASR@ ______  and ATC ."

expected: `100` · result: doc=OK, answered=yes

**3. T3 names another** — **must switch document**

> In the document "arxiv_2608_17214", fill in the blank with the exact value from the text: "A speed-table mutation taking  ______  to 200 knots shrinks the bank-limited radius by (200/230)2 = 0.76, and panel (b) is drawn at that scale."

expected: `230` · result: doc=OK, answered=yes

### arxiv_2608_17214

**4. T1 names it**

> In the document "arxiv_2608_17214", fill in the blank with the exact value from the text: "A speed-table mutation taking  ______  to 200 knots shrinks the bank-limited radius by (200/230)2 = 0.76, and panel (b) is drawn at that scale."

expected: `230` · result: doc=OK, answered=yes

**5. T2 follow-up**

> Fill in the blank with the exact value from the source: "3 mutants of  ______  as the suites were found, 4 once one of them is re-anchored."

expected: `366` · result: doc=OK, answered=no

**6. T3 names another** — **must switch document**

> In the document "irs_p907", fill in the blank with the exact value from the text: "Tax Highlights Future Developments Publication  ______  For the latest information about developments related to Pub."

expected: `907` · result: doc=OK, answered=yes

### irs_p907

**7. T1 names it**

> In the document "irs_p907", fill in the blank with the exact value from the text: "Tax Highlights Future Developments Publication  ______  For the latest information about developments related to Pub."

expected: `907` · result: doc=OK, answered=yes

**8. T2 follow-up**

> Fill in the blank with the exact value from the source: "Or, you can write to the Internal Revenue Service, Tax Forms and Publications,  ______  Constitution Ave."

expected: `1111` · result: doc=OK, answered=yes

**9. T3 names another** — **must switch document**

> In the document "arxiv_2608_18268", fill in the blank with the exact value from the text: "The resulting dataset contains  ______  tweets of the 56 Manifesto categories."

expected: `331,819` · result: doc=OK, answered=yes

### arxiv_2608_18268

**10. T1 names it**

> In the document "arxiv_2608_18268", fill in the blank with the exact value from the text: "The resulting dataset contains  ______  tweets of the 56 Manifesto categories."

expected: `331,819` · result: doc=OK, answered=yes

**11. T2 follow-up**

> Fill in the blank with the exact value from the source: "Articles exceeding a length of  ______  tokens were split into non-overlapping chunks of up to 512 tokens."

expected: `512` · result: doc=OK, answered=yes

**12. T3 names another** — **must switch document**

> In the document "arxiv_2608_01192", fill in the blank with the exact value from the text: "Therefore, the m(ajori)ty of vector-search services rely on approximate NNS where 9× slower than BNTM, roughly two orders of magnitude below Plaintext, on a corpus subset with  ______  k passages."

expected: `100` · result: doc=OK, answered=yes

### arxiv_2608_01192

**13. T1 names it**

> In the document "arxiv_2608_01192", fill in the blank with the exact value from the text: "Therefore, the m(ajori)ty of vector-search services rely on approximate NNS where 9× slower than BNTM, roughly two orders of magnitude below Plaintext, on a corpus subset with  ______  k passages."

expected: `100` · result: doc=OK, answered=yes

**14. T2 follow-up**

> Fill in the blank with the exact value from the source: "We also deterministically sample 1  ______  questions from the MS MARCO dataset and use these as query workload."

expected: `000` · result: doc=OK, answered=yes

**15. T3 names another** — **must switch document**

> In the document "arxiv_2608_18149", fill in the blank with the exact value from the text: "• A predeclared scope gate that accepts energy transfer but 81 Across  ______  real H100 serving executions, the central result is not of 21.43% in P95 TTFT relative to the predeclared expert baseline."

expected: `154` · result: doc=OK, answered=yes

### arxiv_2608_18149

**16. T1 names it**

> In the document "arxiv_2608_18149", fill in the blank with the exact value from the text: "• A predeclared scope gate that accepts energy transfer but 81 Across  ______  real H100 serving executions, the central result is not of 21.43% in P95 TTFT relative to the predeclared expert baseline."

expected: `154` · result: doc=OK, answered=yes

**17. T2 follow-up**

> Fill in the blank with the exact value from the source: "The executor emits  ______ -ms telemetry, benchmark output, server metadata, model and tokenizer revisions, image digest, configuration, and power-limit readback, then seals the emitted-flie hashes in a campaign manifest."

expected: `100` · result: doc=OK, answered=yes

**18. T3 names another** — **must switch document**

> In the document "nist_sp800_161", fill in the blank with the exact value from the text: "This guideline is consistent with the requirements of the Office of Management and Budget (OMB) Circular A- ______ ."

expected: `130` · result: doc=OK, answered=yes

### nist_sp800_161

**19. T1 names it**

> In the document "nist_sp800_161", fill in the blank with the exact value from the text: "This guideline is consistent with the requirements of the Office of Management and Budget (OMB) Circular A- ______ ."

expected: `130` · result: doc=OK, answered=yes

**20. T2 follow-up**

> Fill in the blank with the exact value from the source: "National Institute of Standards and Technology Special Publication  ______ -161r1 Natl."

expected: `800` · result: doc=OK, answered=yes

**21. T3 names another** — **must switch document**

> In the document "arxiv_2608_16043", fill in the blank with the exact value from the text: "The mechanism is that an approximate sen- |𝑆|= ______  to 0.22 ms at |𝑆|=20000) but not in 𝑁, whereas the ANN tinel retrieval covers the hub less often (𝑝 falls with recall), so cover cover insert grows with 𝑁."

expected: `1000` · result: doc=OK, answered=yes

### arxiv_2608_16043

**22. T1 names it**

> In the document "arxiv_2608_16043", fill in the blank with the exact value from the text: "The mechanism is that an approximate sen- |𝑆|= ______  to 0.22 ms at |𝑆|=20000) but not in 𝑁, whereas the ANN tinel retrieval covers the hub less often (𝑝 falls with recall), so cover cover insert grows with 𝑁."

expected: `1000` · result: doc=OK, answered=yes

**23. T2 follow-up**

> Fill in the blank with the exact value from the source: "Coverage Is Not Redundancy: Maintenance Cost and Exposure of Qeury-Aware Admission Indexes in Vector Databases Under Workload Drift Table 6: Ingest-path overhead at 8.8M, |𝑆|= ______ ."

expected: `5000` · result: doc=OK, answered=yes

**24. T3 names another** — **must switch document**

> In the document "arxiv_2608_17360", fill in the blank with the exact value from the text: "We evaluate on two datasets: the  ______  standard text behaviors from HarmBench (Mazeika et al., 2024) and the 100 harmful requests from JailbreakBench (Chao et al., 2024)."

expected: `200` · result: doc=OK, answered=yes
