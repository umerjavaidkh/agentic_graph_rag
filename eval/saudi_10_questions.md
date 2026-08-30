# arwiki_saudi (doc_key `saudi_fix`) — 10 questions

Ingested with rtldoc @6204ea2. 32 pages, 6 chapters, 30 sections.
Result: **8/10** — 8 correct + 1 correct refusal; 1 real failure (structure).

## Run them all

```bash
for q in \
 "عن ماذا تتحدث مقالة السعودية؟" \
 "ما أصل تسمية السعودية؟" \
 "ما هي رؤية السعودية 2030؟" \
 "ما هي أهم المدن في السعودية؟" \
 "كيف يوصف اقتصاد السعودية؟" \
 "ما هي جغرافيا السعودية؟" \
 "ما هي التقسيمات الإدارية في السعودية؟" \
 "ماذا تقول مقالة السعودية عن العلاقات الخارجية؟" \
 "ما هي عناوين أقسام مقالة السعودية؟" \
 "كم عدد سكان اليابان حسب مقالة السعودية؟" ; do
  echo "── $q"
  curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' \
    -d "{\"language\":\"ar\",\"retrieval_mode\":\"unstructured\",\"user_id\":\"admin_001\",\"role\":\"admin\",\"thread_id\":\"$(uuidgen)\",\"question\":\"$q\"}" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);s=d.get('sources') or [];print(' docs',sorted({x.get('id','').split(':')[0] for x in s}));print(' ',(d.get('answer') or '')[:140])"
done
```

Fresh `thread_id` every time — reusing one pins the previous question's document.

## Expected

| # | question | expected |
|---|---|---|
| 1 | عن ماذا تتحدث مقالة السعودية؟ | overview, capital الرياض |
| 2 | ما أصل تسمية السعودية؟ | named after سعود بن محمد بن مقرن |
| 3 | ما هي رؤية السعودية 2030؟ | development plan, 25 إبريل 2016 |
| 4 | ما هي أهم المدن في السعودية؟ | مكة المكرمة, etc. |
| 5 | كيف يوصف اقتصاد السعودية؟ | GDP ≈ $2.246 trillion |
| 6 | ما هي جغرافيا السعودية؟ | terrain, Arabian peninsula |
| 7 | ما هي التقسيمات الإدارية في السعودية؟ | **13 مناطق إدارية** |
| 8 | ماذا تقول مقالة السعودية عن العلاقات الخارجية؟ | non-alignment (عدم الانحياز) |
| 9 | ما هي عناوين أقسام مقالة السعودية؟ | **FAILS — answers from BilArabi** |
| 10 | كم عدد سكان اليابان حسب مقالة السعودية؟ | correctly refuses |

## Why #9 fails

Structure questions hit `bilarabi_fix`, which has **494 Chapter nodes for 239
pages** against Saudi's 6 for 32. One badly-parsed document dominates the
structural index corpus-wide, so *any* "what are the sections" question lands
there regardless of which document is named.

Cause: `_is_heading` (`src/unstructured/graph/axis1_structural.py:823`) trusts
rtldoc's positive heading call but lets a negative fall through to a font-size
rescue. Measured on BilArabi: 113 chapters from `parse_numbered_title`, 0 from
`_COMMON_HEADING`, **381 from the font branches**.
