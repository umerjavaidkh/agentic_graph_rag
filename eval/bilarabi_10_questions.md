# BilArabi — 10 questions, and why 3 fail

Corpus: `bilarabi_v120` (rtldoc 1.2.0). Result: **7/10** answered from BilArabi.

## Run one question

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"language":"ar","retrieval_mode":"unstructured","user_id":"admin_001","role":"admin",
       "thread_id":"'"$(uuidgen)"'",
       "question":"ما هو موضوع دليل المعلم؟"}' | python3 -m json.tool
```

> Use a **fresh `thread_id` every time**. Thread memory pins the previous
> question's document, and reusing `default` silently answers from whatever
> you asked about last — that cost an hour of wrong conclusions once.

## Run all ten

```bash
for q in \
 "ما هو موضوع دليل المعلم؟" \
 "من مؤلفة دليل المعلم؟" \
 "كيف نحكم على صحة المعلومات التي نحصل عليها؟" \
 "ما هي مفاتيح التقييم؟" \
 "ما هو الذكاء الاصطناعي في هذا الدليل؟" \
 "ما هي أسئلة الاستكشاف؟" \
 "ما معنى الهويات في الوحدة الأولى؟" \
 "ما هي أنواع المصادر المستخدمة؟" \
 "ما هي عناوين الوحدات في الدليل؟" \
 "ما هو سعر النفط في عام 2025؟" ; do
  echo "── $q"
  curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
    -d "{\"language\":\"ar\",\"retrieval_mode\":\"unstructured\",\"user_id\":\"admin_001\",\"role\":\"admin\",\"thread_id\":\"$(uuidgen)\",\"question\":\"$q\"}" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);s=d.get('sources') or [];print('  chunks',d.get('total_chunks'),'docs',sorted({x.get('id','').split(':')[0] for x in s}));print('  ',(d.get('answer') or '')[:120])"
done
```

## The 7 that pass

| # | question | from |
|---|---|---|
| 1 | ما هو موضوع دليل المعلم؟ | bilarabi ✅ |
| 3 | كيف نحكم على صحة المعلومات التي نحصل عليها؟ | bilarabi ✅ |
| 4 | ما هي مفاتيح التقييم؟ | bilarabi ✅ (returns the assessment table) |
| 6 | ما هي أسئلة الاستكشاف؟ | bilarabi ✅ |
| 7 | ما معنى الهويات في الوحدة الأولى؟ | bilarabi ✅ |
| 8 | ما هي أنواع المصادر المستخدمة؟ | bilarabi ✅ |
| 10 | ما هو سعر النفط في عام 2025؟ | correctly refuses ✅ |

## The 3 that fail, and why

### 2. `من مؤلفة دليل المعلم؟` — answer CORRECT, sources mixed

Returns `أ.د. هنادا طه تامير` (right), but 6 chunks span `bilarabi_v120` **and**
`doc_arwiki_education`. Not a defect so much as loose scoping — "دليل المعلم"
(teacher's guide) is generic enough that an education article also matches.

### 5. `ما هو الذكاء الاصطناعي في هذا الدليل؟` — WRONG DOCUMENT

Answers from `doc_arwiki_ai`, not BilArabi — even though the question says
**"in this guide"**.

Cause: the parser stores the word with two characters transposed.

```
you type   الاصطناعي     stored in BilArabi   االصطناعي
```

`CONTAINS` cannot match, so BilArabi loses to the Wikipedia AI article, which
spells it correctly. Verify:

```bash
docker exec graphrag-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
 "MATCH (n) WHERE n.logical_doc_id CONTAINS 'bilarabi'
  RETURN count(CASE WHEN n.search_text CONTAINS 'الاصطناعي' THEN 1 END) AS correct,
         count(CASE WHEN n.search_text CONTAINS 'االصطناعي' THEN 1 END) AS reversed;"
```

Expect `correct 0, reversed >0`. Root cause and fix:
[docs/ISSUE_rtldoc_arabic_lam_alef.md](ISSUE_rtldoc_arabic_lam_alef.md) —
**still open in rtldoc 1.2.0** (2,719 reversed pairs in its own output).

### 9. `ما هي عناوين الوحدات في الدليل؟` — GARBAGE STRUCTURE

Returns `1. الوحدة 2 الوحد / 2. الوحدة 2 الوحد` — truncated and duplicated.

Cause: chapter titles in the graph are corrupt. 239 pages produced **512
chapters**, and **187** titles are literally `"Preamble"`.

```bash
docker exec graphrag-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
 "MATCH (n:Chapter) WHERE n.logical_doc_id CONTAINS 'bilarabi'
  RETURN n.title LIMIT 15;"
```

**The surprise:** rtldoc 1.2.0 parses this document into **42 headings, 0
unusable, longest 163 chars**. The graph still gets 512 chapters and 187
`Preamble` — identical to 1.1.0. So the parser's heading fix is real and the
pipeline is not consuming it; chapters are rebuilt by `Axis1StructuralBuilder`
instead. Check standalone:

```bash
rtldoc parse BilArabi_TG07.pdf --json out.json
python3 -c "
import json;d=json.load(open('out.json'));h=[]
def w(o):
  if isinstance(o,dict):
    if str(o.get('role','')).find('head')>=0: h.append(o.get('text',''))
    [w(v) for v in o.values()]
  elif isinstance(o,list): [w(v) for v in o]
w(d);print(len(h),'headings, longest',max(map(len,h)))"
```

## Summary

| failure | layer | open? |
|---|---|---|
| 2 — mixed sources | retrieval scoping | minor |
| 5 — wrong document | **parser** (lam-alef transposition) | yes, in 1.2.0 |
| 9 — garbage headings | **pipeline** ignores rtldoc's headings | yes |
