from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
