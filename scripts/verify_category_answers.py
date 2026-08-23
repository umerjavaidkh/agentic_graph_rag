"""Score each answer by whether its key claims appear in the document."""
import json, re
from neo4j import GraphDatabase

D = "doc_nist_sp800_161"
d = GraphDatabase.driver("bolt://localhost:17687", auth=("neo4j", "password123"))
with d.session() as s:
    CORPUS = " ".join(
        r["t"].lower() for r in s.run(
            "MATCH (n:Section|Page|Chapter|Region) WHERE n.logical_doc_id=$d "
            "RETURN coalesce(n.title,'')+' '+coalesce(n.search_text,'') AS t", d=D)
    )
CORPUS = re.sub(r"\s+", " ", CORPUS)

# Refusals: correct only if the content genuinely is absent (verified earlier).
REFUSAL_OK = {9, 16, 17, 25, 27, 28}   # 25: no self-declared conflict exists in the doc
# Verified wrong: chapter 3 has sections 3.1-3.6 = six factors, answer gave none.
KNOWN_WRONG = {7}

STOP = {"the","and","for","that","this","with","from","are","which","include","includes",
        "document","information","systems","organizations","management","risk","supply",
        "chain","cybersecurity","practices","nist","section","appendix","provides"}

def claims(ans):
    """Distinctive tokens a correct answer should be grounded in."""
    out = set()
    out |= set(re.findall(r"\b[A-Z]{2,}(?:-\d+)?\b", ans))            # acronyms, SA-9
    out |= set(re.findall(r"\b[A-Z][a-z]{4,}(?:\s[A-Z][a-z]{3,})+\b", ans))  # Proper Phrases
    out |= set(re.findall(r"\bTable [A-Z]-\d+\b|\bFigure \d+-?\d*\b", ans))
    out |= set(re.findall(r"\b[A-Z][a-z]+ \d{1,2}, \d{4}\b", ans))     # dates
    out |= set(re.findall(r"\b\d+\.\d+\b", ans))                        # section nums, values
    return {c for c in out if c.lower() not in STOP and len(c) > 2}

rows = json.load(open("/tmp/cat28.json"))
correct = wrong = 0
lines = []
for r in rows:
    n, ans = r["n"], r["ans"]
    if n in KNOWN_WRONG:
        v, why = "WRONG", "six factors exist (3.1-3.6); answer gave no count"
    elif n in REFUSAL_OK:
        v, why = "CORRECT", "refusal; content verified absent from document"
    else:
        c = claims(ans)
        hit = [x for x in c if x.lower() in CORPUS]
        ratio = len(hit) / max(len(c), 1)
        if not c:
            v, why = "WRONG", "no checkable claim in answer"
        elif ratio >= 0.6:
            v, why = "CORRECT", f"{len(hit)}/{len(c)} claims found in document"
        else:
            v, why = "WRONG", f"only {len(hit)}/{len(c)} claims found: " \
                              f"missing {sorted(set(c)-set(hit))[:4]}"
    correct += v == "CORRECT"; wrong += v == "WRONG"
    lines.append(f"| {n} | {r['cat']} | {v} | {why} |")

print(f"CORRECT {correct}/28   WRONG {wrong}/28")
print()
for l in lines: print(l)
