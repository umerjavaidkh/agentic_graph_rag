# Coverage questions — doc_arxiv_2608_19689 (34 pages)

The largest document of the last 100 ingested. 18 questions drawn evenly
across the page range -- pages 1, 6, 10, 14, 18 and 30 -- so a document
that is answerable at the front and invisible at the back cannot score
well by accident.

Result when run this way: **18/18 correct, 18/18 right document, 18/18
source node retrieved.** Naming the document in every question scores the
same, which is the thread memory doing its job.

Note #10 and #18: the generator drew the same sentence twice for page 30,
and #10's source sentence begins "A Limitations" where the parser merged
an appendix heading into the body text. Both are generator/ingestion
artifacts, not retrieval ones.
Ask these in ONE conversation, in order. Only the first names the document; the rest rely on the thread holding it.

| # | Page | Expected | Question |
|---|---|---|---|
| 1 | 1 | `subjective` | In the document "arxiv_2608_19689", fill in the blank with the exact value from the source: "distinguishes objective tasks such as coding from  ______  tasks such as social simulation, and then use it to systematically analyze how accuracy-based evaluation s c[ and hard-label training fail as subjectivity grows." |
| 2 | 1 | `SubjSim` | Fill in the blank with the exact value from the source: "Moreover, since existing datasets record 69 only single observed responses and thus cannot support distributional evaluation, we construct  ______ , a benchmark of 19,300 contexts covering 193 annotators 1." |
| 3 | 1 | `SubjSim` | Fill in the blank with the exact value from the source: "Extensive results on  ______  2: demonstrate the advantages of our method." |
| 4 | 6 | `subjectivity` | Fill in the blank with the exact value from the source: "It further satisfies K∗ (x) ≥ Ks(x), so it increases with the  ______  of the context; eff eff this bound and the remaining properties used below are established in Remark 1." |
| 5 | 6 | `tradeof` | Fill in the blank with the exact value from the source: "Combining the decomposition with statistical and bias bounds yields the main  ______ .f Theorem 1 (Aggregation–Estimation Tradeoff)." |
| 6 | 6 | `subjective` | Fill in the blank with the exact value from the source: "Second, r∗ increases with K∗ (x): the more  ______  a eff eff context, the larger its neighborhood should be." |
| 7 | 10 | `193` | Fill in the blank with the exact value from the source: "All  ______  annotators participated voluntarily; before starting the annotation task, they were informed of the purpose of the study, the type of data collected, and their right to withdraw at any time." |
| 8 | 10 | `participated` | Fill in the blank with the exact value from the source: "All 193 annotators  ______  voluntarily; before starting the annotation task, they were informed of the purpose of the study, the type of data collected, and their right to withdraw at any time." |
| 9 | 10 | `screened` | Fill in the blank with the exact value from the source: "The survey questions themselves were  ______  for cultural suitability for the annotator population during dataset construction, and questions flagged as unsuitable were removed (Appendix G)." |
| 10 | 14 | `SubjSim` | Fill in the blank with the exact value from the source: "A Limitations  ______  uses elicited response propensities rather than direct repeated-choice frequencies." |
| 11 | 14 | `personas` | Fill in the blank with the exact value from the source: "The annotator-level split tests unseen  ______  within known questions rather than transfer to entirely new decision contexts." |
| 12 | 14 | `smoothness` | Fill in the blank with the exact value from the source: "SALT also depends on behavioral  ______  in the representation space, so poor embeddings or discontinuous response patterns can make aggregation harmful." |
| 13 | 18 | `precomputed` | Fill in the blank with the exact value from the source: "Embeddings are  ______  once before training and cached on disk; they are not updated during training." |
| 14 | 18 | `situational` | Fill in the blank with the exact value from the source: "All contexts within the same action-space group share the same  ______  description s." |
| 15 | 18 | `persona` | Fill in the blank with the exact value from the source: "As a result, the embeddings primarily capture  ______  similarity within each group rather than situational variation." |
| 16 | 30 | `fatigue` | Fill in the blank with the exact value from the source: "Exact repeated measurement of the same person is not a viable route to the distributional ground truth, since repetition can change memory, reflection,  ______ , and demand effects." |
| 17 | 30 | `topical` | Fill in the blank with the exact value from the source: "These programs were selected for their broad  ______  coverage, institutional authority, and diversity of question types." |
| 18 | 30 | `fatigue` | Fill in the blank with the exact value from the source: "Exact repeated measurement of the same person is not a viable route to the distributional ground truth, since repetition can change memory, reflection,  ______ , and demand effects." |
