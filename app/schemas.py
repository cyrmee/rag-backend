from datetime import datetime

from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: int


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


class ConversationSummary(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    first_question: str | None = None


class ConversationMessage(BaseModel):
    role: str
    content: str


class SourceInfo(BaseModel):
    content: str
    filename: str
    source_type: str
    source_format: str
    page_number: int | None = None
    document_url: str | None = None


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
