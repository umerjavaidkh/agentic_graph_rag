"""28 categories against IRS Pub 225 (Farmer's Tax Guide).

Every question is grounded in content verified present in the document
BEFORE it was written, or is a deliberate refusal test where the content was
verified ABSENT. `expect` is a token that must appear in a correct answer;
`refuse` marks questions whose correct answer is a decline.
"""
import json, time, urllib.request

Q = [
 (1,"Fact / Lookup","What is the business standard mileage rate for 2025 in IRS Publication 225, the Farmer's Tax Guide?","70 cents",False),
 (2,"Definition","In IRS Publication 225, what is the uniform capitalization rule?","capitaliz",False),
 (3,"Entity / Attribute","Which schedule does a farmer use to report farm income in IRS Publication 225?","Schedule F",False),
 (4,"Structural / Navigation","What does Chapter 12 of IRS Publication 225 cover?","Self-Employment",False),
 (5,"List / Enumerative","List all the chapters in IRS Publication 225, the Farmer's Tax Guide.","Estimated Tax",False),
 (6,"Filtering / Selection","Which chapters of IRS Publication 225 deal with taxes rather than farm income or expenses?","Employment",False),
 (7,"Aggregation / Count","How many numbered chapters does IRS Publication 225 have?","15",False),
 (8,"Comparison","In IRS Publication 225, how do personal expenses differ from business expenses?","business",False),
 (9,"Temporal / Version","What is new for 2026 in IRS Publication 225 compared with what is new for 2025?","2026",False),
 (10,"Multi-hop / Relational","According to IRS Publication 225, which form must a farmer file if they have to pay self-employment tax?","1040",False),
 (11,"Causal / Why","Why does IRS Publication 225 say farmers must keep records?","records",False),
 (12,"Thematic / Synthesis","What subjects does IRS Publication 225 cover overall?","farm",False),
 (13,"Summarization","Summarize Chapter 15 of IRS Publication 225.","Estimated",False),
 (14,"Procedural / How-to","How does IRS Publication 225 say a farmer figures net earnings from self-employment?","earnings",False),
 (15,"Instruction / Requirements","What records does IRS Publication 225 say a farmer must keep?","records",False),
 (16,"Conditional / Rule-based","According to IRS Publication 225, if a farmer has to pay self-employment tax, what must they file?","Form 1040",False),
 (17,"Exception / Edge case","What exception to the economic performance rule does IRS Publication 225 describe?","economic performance",False),
 (18,"Numeric / Calculation","In IRS Publication 225, what is the business standard mileage rate per mile for 2025?","70",False),
 (19,"Table / Structured-data","What does Table 12-1 in IRS Publication 225 show?","Earnings",False),
 (20,"Chart / Figure","What does Figure 3 in IRS Publication 225 show?",None,True),
 (21,"Cross-document","Which other IRS forms or publications does IRS Publication 225 tell farmers to see?","Form",False),
 (22,"Cross-entity","Which IRS forms are named in IRS Publication 225?","943",False),
 (23,"Reference / Citation","Where does IRS Publication 225 mention Form 943?","943",False),
 (24,"Verification / Validation","Does IRS Publication 225 require farmers to keep records, or only suggest it?","records",False),
 (25,"Contradiction / Conflict","Does IRS Publication 225 give conflicting guidance about accounting methods?",None,True),
 (26,"Recommendation / Decision support","Which chapter of IRS Publication 225 should a farmer read about depreciation?","7",False),
 (27,"Ambiguous / Underspecified","What is the rate?",None,True),
 (28,"Unanswerable / Out-of-corpus","What will the standard mileage rate be in 2030?",None,True),
]
out=[]
for n,cat,q,exp,refuse in Q:
    b=json.dumps({"question":q,"user_id":"admin_001","role":"admin","tenant_id":"","thread_id":f"p225-{n}"}).encode()
    r=urllib.request.Request("http://127.0.0.1:8000/query",b,{"Content-Type":"application/json"})
    t=time.perf_counter()
    try:
        d=json.load(urllib.request.urlopen(r,timeout=240)); el=time.perf_counter()-t
        ans=" ".join((d.get("answer") or "").split())
        rec={"n":n,"cat":cat,"q":q,"expect":exp,"refuse":refuse,"s":round(el,1),
             "doc":d.get("document_id"),"ans":ans}
    except Exception as e:
        rec={"n":n,"cat":cat,"q":q,"expect":exp,"refuse":refuse,"s":round(time.perf_counter()-t,1),
             "doc":None,"ans":f"ERROR {type(e).__name__}"}
    out.append(rec); print(f"{n:>2}. {cat:<34}{rec['s']:>6}s doc={rec['doc']}",flush=True)
json.dump(out,open("/tmp/cat28b.json","w"),indent=1)
