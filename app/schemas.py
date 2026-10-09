from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: int


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    # The message this turn should attach under - omit to continue the
    # conversation normally. Pass an earlier message's parent id to edit
    # that question or regenerate its answer: the new turn becomes a
    # sibling branch instead of overwriting what's there, and the old one
    # stays reachable by branching from the same parent again. Pass "" (not
    # null/omitted - those mean "continue normally") to branch from before
    # the very first message, i.e. to edit/regenerate the first turn.
    parent_message_id: str | None = None
    # Opt-in per request: exposes the web_search tool to the model for
    # this turn only. Off by default - the model can't reach the public
    # internet unless the caller explicitly asks for it here.
    web_search: bool = False
    # Ids from prior POST /attachments responses - their extracted text is
    # folded into this turn's question for the model, then discarded (not
    # persisted with the turn - see app/agent.py's _augment_with_attachments).
    attachment_ids: list[str] = []
    # Council mode (app/council.py) - parallel planned searches run before
    # the model's first turn, plus a citation check on the answer (reported
    # in the done event's citation_warnings). true/false forces it on/off;
    # omitted (null) routes by question: complex questions get council mode,
    # simple ones the regular agent. The done event's `council` says which
    # one answered.
    council: bool | None = None
    # Names the mode outright, overriding `council`: "agent", "council", or
    # "research" - app/research.py's slow, multi-round read of the corpus
    # for complex questions (minutes, not seconds; progress arrives as
    # research_plan/research_progress events). Research is never chosen
    # automatically. Omitted, the council/auto-routing rules above apply.
    mode: Literal["agent", "council", "research"] | None = None


class AttachmentInfo(BaseModel):
    id: str
    filename: str
    char_count: int


class ConversationSummary(BaseModel):
    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    first_question: str | None = None


class ConversationMessageNode(BaseModel):
    id: str
    parent_message_id: str | None = None
    role: str
    content: str
    created_at: datetime


class ConversationDetail(BaseModel):
    id: str
    title: str | None = None
    active_message_id: str | None = None
    # Every message in the conversation (all branches), not just the
    # active path - a client walks parent_message_id from
    # active_message_id to render the current transcript, and can offer
    # switching to a sibling branch using whatever else is in here.
    messages: list[ConversationMessageNode]


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
