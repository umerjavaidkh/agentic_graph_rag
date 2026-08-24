# Corpus-500 QA log — 20 random documents, 5 questions each

Run 2026-08-23 against 499 documents / 11,019 pages. Questions are generated
deterministically from each document's own text (cloze over a stated fact), so
the expected answer is verifiable and no LLM is spent generating them.

Score counts an answer correct when the expected span appears in it.
`doc` = how often retrieval resolved the document the question came from.

| Document | Score | Nodes | Avg s | Right doc |
|---|---|---|---|---|
| arxiv_2608_20106 | 5/5 | 118 | 35 | 5/5 |
| arxiv_2608_19000 | 5/5 | 72 | 51 | 5/5 |
| arxiv_2608_19973 | 5/5 | 88 | 48 | 5/5 |
| arxiv_2608_19408 | 5/5 | 48 | 28 | 5/5 |
| arxiv_2608_19982 | 5/5 | 31 | 29 | 5/5 |
| irs_p946 | 5/5 | 317 | 36 | 5/5 |
| arxiv_2608_09185 | 5/5 | 71 | 42 | 5/5 |
| arxiv_2608_06043 | 5/5 | 71 | 34 | 5/5 |
| arxiv_2608_17355 | 5/5 | 63 | 48 | 5/5 |
| arxiv_2608_18833 | 5/5 | 47 | 29 | 5/5 |
| arxiv_2608_16422 | 5/5 | 41 | 71 | 5/5 |
| arxiv_2608_18307 | 5/5 | 99 | 37 | 5/5 |
| arxiv_2608_09028 | 4/5 | 88 | 47 | 4/5 |
| nist_csf_2 | 4/5 | 62 | 126 | 5/5 |
| arxiv_2608_19598 | 4/5 | 49 | 101 | 5/5 |
| arxiv_rag | 4/5 | 56 | 34 | 5/5 |
| arxiv_2608_16026 | 4/5 | 23 | 15 | 4/5 |
| arxiv_2608_18907 | 4/5 | 75 | 69 | 5/5 |
| arxiv_2608_19693 | 3/5 | 83 | 42 | 5/5 |
| nist_cswp_27 | no questions built | 40 | — | 0/0 |

**Total 87/95 (92%)**

## Failures

Each failure classified as a retrieval defect or a bad question — a single
percentage hides which one you are actually looking at.

### arxiv_2608_09028 — page 11
- expected: `93.7%`
- resolved document: `arxiv_2608_08751`  **(wrong document)**
- latency: 35.2s
- question: The pipeline processes 1,663 sentences →443 classified rules in 34.9 minutes wall-clock (4.73 s per rule, 1.26 s per sen
- answer: This document does not cover the mean GPU utilization value.

### nist_csf_2 — page 12
- expected: `3.2`
- resolved document: `nist_csf_2`
- latency: 34.6s
- question: Also, a Target Profile can help express the organization’s cybersecurity risk management requirements and expectations t
- answer: Also, a Target Profile can help express the organization’s cybersecurity risk management requirements and expectations to suppliers, partners, and other third parties as a target for those parties to 

### arxiv_2608_19598 — page 5
- expected: `1.1 Models`
- resolved document: `arxiv_2608_19598`
- latency: 255.0s
- question: Full derivations and an em- pirical validation showing how 𝐺𝑚and 𝐺𝑟shift before and after training are provided in Appen
- answer: Full derivations and an empirical validation showing how 𝐺𝑚 and 𝐺𝑟 shift before and after training are provided in Appendix 6.1.

### arxiv_rag — page 3
- expected: `2.1 Models`
- resolved document: `arxiv_rag`
- latency: 37.8s
- question: In the following, we formally introduce both models and then describe the pη and pθ components, as well as the training 
- answer: In the following, we formally introduce both models and then describe the pη and pθ components, as well as the training and decoding procedure. **RAG-Sequence Model** The RAG-Sequence model uses the s

### arxiv_2608_16026 — page 7
- expected: `1`
- resolved document: `arxiv_2608_13136`  **(wrong document)**
- latency: 9.3s
- question: We summarize the paradigms that can be used for multi-turn watermark design as follows. ___) Orchestrating tool-call seq
- answer: This document does not cover the specific statement regarding the paradigms for multi-turn watermark design.

### arxiv_2608_18907 — page 4
- expected: `4.1 Sample`
- resolved document: `arxiv_2608_18907`
- latency: 221.5s
- question: The overall algorithm is summarized in Appendix A. ___ Learning State Estimation Existing dynamic GDA methods typically 
- answer: The overall algorithm is summarized in Appendix A. **Sample** Learning State Estimation Existing dynamic GDA methods typically use downstream model predictions as feedback to select original samples f

### arxiv_2608_19693 — page 21
- expected: `21`
- resolved document: `arxiv_2608_19693`
- latency: 80.4s
- question: RIPE++: Reinforced Keypoint Learning from Positive Pairs Only ___ where Ku p is the detection heatmap logit at location 
- answer: In document "arxiv_2608_19693", the completed statement is: RIPE++: Reinforced Keypoint Learning from Positive Pairs Only **where Ku is the detection heatmap logit at location p**.

### arxiv_2608_19693 — page 9
- expected: `10`
- resolved document: `arxiv_2608_19693`
- latency: 25.1s
- question: We define the loss of the matcher as Lmatch = −EMA↔B[R(MA↔B)] and the final training loss as \mathc a l {L}=\mathcal {L}
- answer: The document does not provide a specific value for η controlling the strength of the non-matchable regularization term.


## Run 2026-08-23 16:18 — strategy `graph_rag_vector_first`

20 documents sampled (seed 20260823) from the most recently ingested 500, 5 questions each.

**Answer accuracy 80/94 (85%) · right document 94/94 (100%) · median 16s**

| Document | Score | Nodes | Avg s | Right doc |
|---|---|---|---|---|
| arxiv_2608_07947 | 5/5 | 57 | 18 | 5/5 |
| arxiv_2608_10517 | 5/5 | 89 | 15 | 5/5 |
| arxiv_2608_12440 | 5/5 | 48 | 9 | 5/5 |
| arxiv_2608_12929 | 5/5 | 36 | 16 | 5/5 |
| arxiv_2608_13612 | 5/5 | 41 | 13 | 5/5 |
| arxiv_2608_17214 | 5/5 | 58 | 8 | 5/5 |
| arxiv_2608_17613 | 5/5 | 72 | 21 | 5/5 |
| arxiv_2608_18779 | 5/5 | 77 | 11 | 5/5 |
| arxiv_2608_19269 | 5/5 | 48 | 13 | 5/5 |
| arxiv_2608_19680 | 5/5 | 70 | 17 | 5/5 |
| arxiv_2608_11034 | 4/5 | 52 | 13 | 5/5 |
| arxiv_2608_11840 | 4/5 | 64 | 16 | 5/5 |
| arxiv_2608_17694 | 4/5 | 33 | 18 | 5/5 |
| arxiv_2608_18329 | 4/5 | 42 | 16 | 5/5 |
| arxiv_2608_10555 | 3/3 | 28 | 17 | 3/3 |
| arxiv_2608_13632 | 3/5 | 61 | 18 | 5/5 |
| arxiv_2608_13681 | 3/5 | 60 | 25 | 5/5 |
| arxiv_2608_16742 | 3/5 | 61 | 24 | 5/5 |
| arxiv_2608_15943 | 1/5 | 75 | 20 | 5/5 |
| arxiv_2608_16618 | 1/1 | 21 | 23 | 1/1 |


## Run 2026-08-23 17:52 — strategy `graph_rag_vector_first`

18 documents sampled (seed 20260823) from the most recently ingested 500, 5 questions each.

**Answer accuracy 61/62 (98%) · right document 62/62 (100%) · median 9s**

| Document | Score | Nodes | Avg s | Right doc |
|---|---|---|---|---|
| arxiv_2608_10517 | 5/5 | 89 | 9 | 5/5 |
| arxiv_2608_12440 | 5/5 | 48 | 5 | 5/5 |
| arxiv_2608_13612 | 5/5 | 41 | 6 | 5/5 |
| arxiv_2608_17214 | 5/5 | 58 | 6 | 5/5 |
| arxiv_2608_17613 | 5/5 | 72 | 17 | 5/5 |
| arxiv_2608_18329 | 5/5 | 42 | 10 | 5/5 |
| arxiv_2608_18779 | 5/5 | 77 | 5 | 5/5 |
| arxiv_2608_19269 | 5/5 | 48 | 5 | 5/5 |
| arxiv_2608_11034 | 4/4 | 52 | 14 | 4/4 |
| arxiv_2608_16742 | 4/4 | 61 | 18 | 4/4 |
| arxiv_2608_19680 | 4/4 | 70 | 17 | 4/4 |
| arxiv_2608_11840 | 3/3 | 64 | 9 | 3/3 |
| arxiv_2608_13681 | 2/2 | 60 | 21 | 2/2 |
| arxiv_2608_07947 | 1/1 | 57 | 21 | 1/1 |
| arxiv_2608_10555 | 1/1 | 28 | 21 | 1/1 |
| arxiv_2608_12929 | 1/1 | 36 | 13 | 1/1 |
| arxiv_2608_16618 | 1/1 | 21 | 20 | 1/1 |
| arxiv_2608_13632 | 0/1 | 61 | 9 | 1/1 |


## Run 2026-08-24 13:37 — strategy `graph_rag_vector_first`

18 documents sampled (seed 20260823) from the most recently ingested 500, 5 questions each.

**Answer accuracy 61/62 (98%) · right document 62/62 (100%) · median 10s**

| Document | Score | Nodes | Avg s | Right doc |
|---|---|---|---|---|
| arxiv_2608_10517 | 5/5 | 89 | 10 | 5/5 |
| arxiv_2608_12440 | 5/5 | 48 | 5 | 5/5 |
| arxiv_2608_13612 | 5/5 | 41 | 6 | 5/5 |
| arxiv_2608_17214 | 5/5 | 58 | 6 | 5/5 |
| arxiv_2608_17613 | 5/5 | 72 | 19 | 5/5 |
| arxiv_2608_18329 | 5/5 | 42 | 10 | 5/5 |
| arxiv_2608_18779 | 5/5 | 77 | 4 | 5/5 |
| arxiv_2608_19269 | 5/5 | 48 | 5 | 5/5 |
| arxiv_2608_11034 | 4/4 | 52 | 16 | 4/4 |
| arxiv_2608_16742 | 4/4 | 61 | 20 | 4/4 |
| arxiv_2608_19680 | 4/4 | 70 | 21 | 4/4 |
| arxiv_2608_11840 | 3/3 | 64 | 10 | 3/3 |
| arxiv_2608_13681 | 2/2 | 60 | 25 | 2/2 |
| arxiv_2608_07947 | 1/1 | 57 | 25 | 1/1 |
| arxiv_2608_10555 | 1/1 | 28 | 27 | 1/1 |
| arxiv_2608_12929 | 1/1 | 36 | 14 | 1/1 |
| arxiv_2608_16618 | 1/1 | 21 | 26 | 1/1 |
| arxiv_2608_13632 | 0/1 | 61 | 11 | 1/1 |
