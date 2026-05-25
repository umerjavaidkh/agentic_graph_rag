from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from .bridge import ask
from .auth.roles import UserContext, validate_role
from .unstructured.retriever import ESGComplianceRetriever
from .structured.retriever import StructuredRetriever

app = FastAPI(title="ESG Compliance Agent API")

# for shutdown cleanup
_unstructured_retriever = ESGComplianceRetriever()
_structured_retriever   = StructuredRetriever()


class QueryRequest(BaseModel):
    question:    str           = Field(..., description="User's question")
    role:        Optional[str] = Field(default="public", description="User role for access control")
    user_id:     Optional[str] = Field(default="anonymous", description="User identifier")
    department:  Optional[str] = Field(default=None, description="User department")
    thread_id:   Optional[str] = Field(default="default")


class QueryResponse(BaseModel):
    answer:       str
    sources:      list
    keywords:     list
    total_chunks: int
    agent:        str   # "unstructured" | "structured" | "hybrid"
    strategy:     str   # "semantic" | "text2cypher" | "multi_hop" | "vector"
    access_level: str


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        role = validate_role(request.role or "public")
        context = UserContext(
            user_id=request.user_id or "anonymous",
            role=role,
            department=request.department,
        )

        result = ask(request.question, user_context=context)

        return QueryResponse(
            answer       = result.get("answer", "No answer generated."),
            sources      = result.get("sources", []),
            keywords     = result.get("keywords", []),
            total_chunks = len(result.get("sources", [])),
            agent        = result.get("agent", "unstructured"),
            strategy     = result.get("strategy", "semantic"),
            access_level = result.get("_access_level", role.value),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
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