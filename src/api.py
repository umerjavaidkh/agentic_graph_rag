from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from .bridge import ask
from .unstructured.retriever import ESGComplianceRetriever
from .structured.retriever import StructuredRetriever

app = FastAPI(title="ESG Compliance Agent API")

# for shutdown cleanup
_unstructured_retriever = ESGComplianceRetriever()
_structured_retriever   = StructuredRetriever()


class QueryRequest(BaseModel):
    question:  str           = Field(..., description="User's question")
    thread_id: Optional[str] = Field(default="default")


class QueryResponse(BaseModel):
    answer:       str
    sources:      list
    keywords:     list
    total_chunks: int
    agent:        str   # "unstructured" | "structured" | "hybrid"
    strategy:     str   # "semantic" | "text2cypher" | "multi_hop" | "vector"


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = ask(request.question)

        return QueryResponse(
            answer       = result.get("answer", "No answer generated."),
            sources      = result.get("sources", []),
            keywords     = result.get("keywords", []),
            total_chunks = len(result.get("sources", [])),
            agent        = result.get("agent", "unstructured"),
            strategy     = result.get("strategy", "semantic"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("shutdown")
async def shutdown():
    _unstructured_retriever.close()
    _structured_retriever.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)