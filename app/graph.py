"""Knowledge-graph retrieval, after HippoRAG 2 (Gutiérrez et al., 2025),
in our own schema and with our own models. In the trial behind it
(ENG-3310, October 2026) it found 32 of 46 known answer passages in the
top 5 against 26 for document_routed_search, the whole gain on questions
whose answer spans several documents; on exact figures inside tables it
ranked lower, which is why it is fused with the existing search as one
more evidence source (council and research mode) rather than replacing it.

Index (prepare_graph + store_graph; called from app/ingestion.py and
scripts/maintenance/backfill_graph.py): per chunk, two thinking-off calls -
named entities, then subject-relation-object triples conditioned on them.
Each distinct normalised subject or object is a phrase node with an
embedding (graph_phrases); each distinct triple a fact with an embedding
(graph_facts) that joins its two phrases; graph_triples records which
chunks state which fact, which is the passage-phrase edge; and phrases
whose embeddings are close get a synonym edge (graph_edges) - what joins
"amhara bank" in one document to "amhara bank s c" in another.

Query (graph_search): the question's embedding picks the closest facts,
one thinking-off call keeps those that bear on the question, and their
phrases - plus every passage, weighted lightly by plain embedding
similarity - seed Personalized PageRank over the whole graph. Passages are
ranked by PageRank score: the ones the chosen facts point at directly, and
ones a hop or two away through shared or synonymous phrases. If the filter
keeps nothing, the ranking is the embedding similarity alone. The graph is
held in process (igraph) and reloaded from Postgres when the tables change.
"""

import asyncio
import difflib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import igraph
import numpy as np
from pgvector import Vector

from app.config import settings
from app.db import get_connection
from app.embeddings import embed_texts
from app.generation import generate_from_messages

logger = logging.getLogger(__name__)

# Extraction calls for one document run in parallel; vLLM batches them.
EXTRACTION_CONCURRENCY = 8
NER_MAX_TOKENS = 600
TRIPLES_MAX_TOKENS = 1500

# Synonym edges: phrases whose embeddings are at least this similar, at
# most this many per phrase (HippoRAG 2's defaults).
SYNONYM_THRESHOLD = 0.8
SYNONYM_MAX_NEIGHBOURS = 100

# Query side, HippoRAG 2's defaults: facts offered to the filter call and
# phrases kept as PageRank seeds; the weight of the passage-similarity
# seeds next to the phrase seeds (a phrase seed is up to 1.0); PageRank's
# damping (0.5: half the walk restarts at the seeds every step, so scores
# stay close to them).
LINKING_TOP_K = 5
PASSAGE_NODE_WEIGHT = 0.05
DAMPING = 0.5
FILTER_MAX_TOKENS = 512
# Fact scores are min-max normalised over every fact in the graph. The
# minimum is taken over this many facts held in memory, not a scan of
# every fact embedding per query; the top facts' scores barely move.
FACT_SAMPLE = 2000


# ----------------------------------------------------------------------
# Extraction


def normalise_phrase(text) -> str:
    """Lower case, punctuation to spaces, one space between words - so
    "Amhara Bank S.C." and "amhara bank s c" are the same node. Letters
    and digits of any script are kept (HippoRAG keeps ASCII only, which
    would turn every Amharic phrase into nothing)."""
    return " ".join(re.sub(r"[^\w ]+", " ", str(text).lower()).split())


def passage_text(filename: str, page_number: int | None, source_type: str, content: str) -> str:
    """What the extractor reads: the chunk with its document's name in
    front, so a form that never names its subject in the body ("Device:
    Cisco FTD" in Amhara Bank's VPN form) still yields a fact about the
    right party, and captions are marked as descriptions of an image."""
    page = f" (page {page_number})" if page_number else ""
    kind = {
        "image_caption": " [description of an image in the document]",
        "chart_data": " [data of a chart in the document]",
    }.get(source_type, "")
    return f"Document: {Path(filename).name}{page}{kind}\n{content}"


# HippoRAG 2's one-shot extraction prompts, kept as they are: the trial
# measured extraction quality with exactly these.
_NER_SYSTEM = (
    "Your task is to extract named entities from the given paragraph. \n"
    "Respond with a JSON list of entities.\n"
)
_EXAMPLE_PASSAGE = (
    "Radio City\n"
    "Radio City is India's first private FM radio station and was started on 3 July 2001.\n"
    "It plays Hindi, English and regional songs.\n"
    "Radio City recently forayed into New Media in May 2008 with the launch of a music portal - "
    "PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."
)
_EXAMPLE_ENTITIES = (
    '{"named_entities":\n'
    '    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]\n'
    "}\n"
)
_TRIPLES_SYSTEM = (
    "Your task is to construct an RDF (Resource Description Framework) graph from the given passages "
    "and named entity lists. \n"
    "Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. \n\n"
    "Pay attention to the following requirements:\n"
    "- Each triple should contain at least one, but preferably two, of the named entities in the list "
    "for each passage.\n"
    "- Clearly resolve pronouns to their specific names to maintain clarity.\n\n"
)
_TRIPLES_FRAME = (
    "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\n"
    "Paragraph:\n```\n{passage}\n```\n\n{entities}\n"
)
_EXAMPLE_TRIPLES = (
    '{"triples": [\n'
    '            ["Radio City", "located in", "India"],\n'
    '            ["Radio City", "is", "private FM radio station"],\n'
    '            ["Radio City", "started on", "3 July 2001"],\n'
    '            ["Radio City", "plays songs in", "Hindi"],\n'
    '            ["Radio City", "plays songs in", "English"],\n'
    '            ["Radio City", "forayed into", "New Media"],\n'
    '            ["Radio City", "launched", "PlanetRadiocity.com"],\n'
    '            ["PlanetRadiocity.com", "launched in", "May 2008"],\n'
    '            ["PlanetRadiocity.com", "is", "music portal"],\n'
    '            ["PlanetRadiocity.com", "offers", "news"],\n'
    '            ["PlanetRadiocity.com", "offers", "videos"],\n'
    '            ["PlanetRadiocity.com", "offers", "songs"]\n'
    "    ]\n"
    "}\n"
)


def _parse_json_reply(reply: str):
    """The model's JSON, whether bare, fenced, or wrapped in a sentence;
    None if there is none. Never eval()s the text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL) or re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _parse_entities(reply: str) -> list[str]:
    data = _parse_json_reply(reply)
    if isinstance(data, dict):
        data = data.get("named_entities", [])
    if not isinstance(data, list):
        return []
    return [str(e).strip() for e in data if isinstance(e, (str, int, float)) and str(e).strip()]


def _parse_triples(reply: str) -> list[tuple[str, str, str]]:
    """Normalised, de-duplicated (subject, relation, object) triples; ones
    with a missing part, or whose subject and object are the same phrase,
    are dropped."""
    data = _parse_json_reply(reply)
    if isinstance(data, dict):
        data = data.get("triples", [])
    if not isinstance(data, list):
        return []
    triples: list[tuple[str, str, str]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        subject, relation, obj = (normalise_phrase(part) for part in item)
        if not subject or not relation or not obj or subject == obj:
            continue
        if (subject, relation, obj) not in triples:
            triples.append((subject, relation, obj))
    return triples


async def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """The two extraction calls for one passage, at temperature 0 so the
    same chunk yields the same graph on re-upload. Raises if a call fails;
    an empty list means the model found nothing."""
    entities = _parse_entities(await generate_from_messages([
        {"role": "system", "content": _NER_SYSTEM},
        {"role": "user", "content": _EXAMPLE_PASSAGE},
        {"role": "assistant", "content": _EXAMPLE_ENTITIES},
        {"role": "user", "content": text},
    ], max_tokens=NER_MAX_TOKENS, temperature=0.0))
    entities_json = json.dumps({"named_entities": entities}, ensure_ascii=False)
    return _parse_triples(await generate_from_messages([
        {"role": "system", "content": _TRIPLES_SYSTEM},
        {"role": "user", "content": _TRIPLES_FRAME.format(passage=_EXAMPLE_PASSAGE, entities=_EXAMPLE_ENTITIES)},
        {"role": "assistant", "content": _EXAMPLE_TRIPLES},
        {"role": "user", "content": _TRIPLES_FRAME.format(passage=text, entities=entities_json)},
    ], max_tokens=TRIPLES_MAX_TOKENS, temperature=0.0))


@dataclass
class GraphBatch:
    """Extraction results for a set of chunks, with the embeddings they
    need, ready to be stored: everything that takes model calls happens
    in prepare_graph, outside any transaction; store_graph only writes."""

    triples: dict  # chunk key -> triples, for chunks whose extraction ran (possibly to nothing)
    phrase_embeddings: dict[str, list[float]]
    fact_embeddings: dict[tuple[str, str, str], list[float]]

    @property
    def is_empty(self) -> bool:
        return not self.triples


async def prepare_graph(texts: dict) -> GraphBatch:
    """Extracts triples from every text (keyed however the caller likes)
    and embeds the phrases and facts found. A chunk whose extraction
    fails is left out, logged, and stays unextracted for a later backfill;
    it doesn't fail the document."""
    semaphore = asyncio.Semaphore(EXTRACTION_CONCURRENCY)

    async def one(key, text: str):
        async with semaphore:
            try:
                return key, await extract_triples(text)
            except Exception:
                logger.warning("graph extraction failed for chunk %r", key, exc_info=True)
                return key, None

    results = await asyncio.gather(*(one(key, text) for key, text in texts.items()))
    triples = {key: found for key, found in results if found is not None}
    facts = sorted({t for found in triples.values() for t in found})
    phrases = sorted({p for s, _, o in facts for p in (s, o)})
    # The embedding server returns an all-zero vector for the odd short
    # text, consistently for some (see app/embeddings.py). A zero vector
    # can't be compared with anything, so such a phrase, and every fact
    # that uses it, is left out of the graph rather than stored.
    phrase_embeddings = {
        p: v for p, v in zip(phrases, await embed_texts(phrases, allow_zero=True)) if any(v)
    }
    facts = [f for f in facts if f[0] in phrase_embeddings and f[2] in phrase_embeddings]
    fact_embeddings = {
        f: v for f, v in zip(facts, await embed_texts([" ".join(f) for f in facts], allow_zero=True)) if any(v)
    }
    triples = {key: [f for f in found if f in fact_embeddings] for key, found in triples.items()}
    return GraphBatch(triples, phrase_embeddings, fact_embeddings)


async def store_graph(cur, batch: GraphBatch, chunk_ids: dict) -> dict:
    """Writes a prepared batch inside the caller's transaction: upserts
    phrases and facts, links facts to their chunks, adds synonym edges for
    phrases new to the graph, and stamps the chunks as extracted. Returns
    counts for logging."""
    chunks = {chunk_ids[key]: found for key, found in batch.triples.items() if key in chunk_ids}
    stats = {"chunks": len(chunks), "facts": 0, "new_phrases": 0, "synonym_edges": 0}
    if not chunks:
        return stats
    await cur.execute(
        "update documents set graph_extracted_at = now() where id = any(%s)", (list(chunks),),
    )
    facts = sorted({t for found in chunks.values() for t in found})
    if not facts:
        return stats
    phrases = sorted({p for s, _, o in facts for p in (s, o)})

    # xmax = 0 on the returned row means it was inserted, not found.
    phrase_ids: dict[str, str] = {}
    new_phrases: list[str] = []
    for phrase in phrases:
        await cur.execute(
            """
            insert into graph_phrases (phrase, embedding) values (%s, %s)
            on conflict (phrase) do update set phrase = excluded.phrase
            returning id, (xmax = 0) as inserted
            """,
            (phrase, Vector(batch.phrase_embeddings[phrase])),
        )
        phrase_id, inserted = await cur.fetchone()
        phrase_ids[phrase] = phrase_id
        if inserted:
            new_phrases.append(phrase)

    fact_ids: dict[tuple[str, str, str], str] = {}
    for fact in facts:
        subject, relation, obj = fact
        await cur.execute(
            """
            insert into graph_facts (subject_id, object_id, subject, relation, object, embedding)
            values (%s, %s, %s, %s, %s, %s)
            on conflict (subject, relation, object) do update set subject = excluded.subject
            returning id
            """,
            (phrase_ids[subject], phrase_ids[obj], subject, relation, obj, Vector(batch.fact_embeddings[fact])),
        )
        fact_ids[fact] = (await cur.fetchone())[0]
    for chunk_id, found in chunks.items():
        for fact in found:
            await cur.execute(
                "insert into graph_triples (fact_id, chunk_id) values (%s, %s) on conflict do nothing",
                (fact_ids[fact], chunk_id),
            )

    # Synonym edges, new phrases against the whole graph (including each
    # other). ef_search must cover the neighbours asked for, or the index
    # returns fewer.
    await cur.execute(f"set local hnsw.ef_search = {SYNONYM_MAX_NEIGHBOURS + 40}")
    for phrase in new_phrases:
        # Very short phrases ("x", "22") match too freely to be synonyms.
        if len(re.sub(r"\W", "", phrase)) <= 2:
            continue
        vector = Vector(batch.phrase_embeddings[phrase])
        await cur.execute(
            """
            select id, 1 - (embedding <=> %(v)s) as similarity
            from graph_phrases
            where id <> %(id)s
            order by embedding <=> %(v)s
            limit %(k)s
            """,
            {"v": vector, "id": phrase_ids[phrase], "k": SYNONYM_MAX_NEIGHBOURS},
        )
        for other_id, similarity in await cur.fetchall():
            # `not >=` rather than `<`: a NaN similarity (a zero vector on
            # either side) must stop the loop too, not become an edge.
            if not similarity >= SYNONYM_THRESHOLD:
                break
            await cur.execute(
                """
                insert into graph_edges (phrase_a, phrase_b, weight)
                values (least(%(a)s, %(b)s), greatest(%(a)s, %(b)s), %(w)s)
                on conflict do nothing
                """,
                {"a": phrase_ids[phrase], "b": other_id, "w": float(similarity)},
            )
            stats["synonym_edges"] += cur.rowcount
    stats["facts"] = len(facts)
    stats["new_phrases"] = len(new_phrases)
    return stats


async def remove_orphans(cur) -> None:
    """After a document's chunks are deleted (its graph_triples rows go
    with them): drops facts no chunk states any more, then phrases no fact
    uses; their synonym edges cascade."""
    await cur.execute(
        "delete from graph_facts f where not exists (select 1 from graph_triples t where t.fact_id = f.id)"
    )
    await cur.execute(
        """
        delete from graph_phrases p
        where not exists (select 1 from graph_facts f where f.subject_id = p.id)
          and not exists (select 1 from graph_facts f where f.object_id = p.id)
        """
    )


# ----------------------------------------------------------------------
# The graph in memory


def _min_max(x: np.ndarray) -> np.ndarray:
    low, high = float(x.min()), float(x.max())
    if high == low:
        return np.ones_like(x)
    return (x - low) / (high - low)


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _matrix(vectors: list, dim: int) -> np.ndarray:
    """Unit rows from pgvector values (one per row), or an empty matrix."""
    if not vectors:
        return np.zeros((0, dim), dtype=np.float32)
    return _unit(np.array([np.asarray(v.to_numpy(), dtype=np.float32) for v in vectors]))


@dataclass
class KnowledgeGraph:
    """Phrase vertices come first (0 .. phrase_count-1), passage vertices
    after, in `passage_ids` order. Edge weights follow HippoRAG 2, where
    every phrase-phrase relation is recorded in both directions and ends
    up as two parallel edges: a fact edge weighs twice the number of
    chunk-level triples between the two phrases, a synonym edge twice the
    similarity, and a passage-phrase edge 1."""

    stamp: tuple
    phrase_index: dict[str, int]  # phrase text -> vertex
    chunk_counts: np.ndarray  # per phrase vertex: how many chunks state a fact with it
    passage_ids: list
    passage_matrix: np.ndarray  # unit embeddings, one row per passage
    fact_sample: np.ndarray  # unit embeddings of a sample of facts
    graph: igraph.Graph

    @property
    def phrase_count(self) -> int:
        return len(self.phrase_index)

    @classmethod
    def build(cls, stamp, phrases, passages, fact_edges, passage_edges, synonyms, fact_sample, dim: int):
        phrase_vertex = {phrase_id: i for i, (phrase_id, _) in enumerate(phrases)}
        phrase_index = {phrase: i for i, (_, phrase) in enumerate(phrases)}
        offset = len(phrases)
        passage_ids = [passage_id for passage_id, _ in passages]
        passage_vertex = {passage_id: offset + i for i, passage_id in enumerate(passage_ids)}
        matrix = _matrix([e for _, e in passages], dim)
        sample = _matrix([e for (e,) in fact_sample], dim)

        edges: dict[tuple[int, int], float] = {}
        for a, b, count in fact_edges:
            key = (min(phrase_vertex[a], phrase_vertex[b]), max(phrase_vertex[a], phrase_vertex[b]))
            edges[key] = edges.get(key, 0.0) + 2.0 * count
        chunk_counts = np.zeros(offset, dtype=np.float32)
        for chunk_id, phrase_id in passage_edges:
            if chunk_id in passage_vertex:
                edges[(phrase_vertex[phrase_id], passage_vertex[chunk_id])] = 1.0
                chunk_counts[phrase_vertex[phrase_id]] += 1
        skipped = 0
        for a, b, weight in synonyms:
            if not math.isfinite(weight):
                skipped += 1  # one NaN weight makes every PageRank score NaN
                continue
            key = (min(phrase_vertex[a], phrase_vertex[b]), max(phrase_vertex[a], phrase_vertex[b]))
            edges[key] = edges.get(key, 0.0) + 2.0 * weight
        if skipped:
            logger.warning("knowledge graph: skipped %d synonym edges with a non-finite weight", skipped)

        graph = igraph.Graph(
            n=offset + len(passage_ids), edges=list(edges), edge_attrs={"weight": list(edges.values())},
        )
        return cls(stamp, phrase_index, chunk_counts, passage_ids, matrix, sample, graph)

    def pagerank(self, reset: np.ndarray) -> np.ndarray:
        """Personalized PageRank from `reset` (one weight per vertex);
        returns the passages' scores, in passage_ids order."""
        scores = self.graph.personalized_pagerank(
            damping=DAMPING, weights="weight", reset=reset.tolist(), implementation="prpack",
        )
        return np.asarray(scores[self.phrase_count:], dtype=np.float32)


_graph: KnowledgeGraph | None = None
_graph_lock = asyncio.Lock()


async def _stamp() -> tuple:
    """Changes whenever a chunk, fact, synonym edge or phrase is added or
    removed - the signal to reload the in-memory graph."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select (select count(*) from documents), (select max(created_at) from documents),
                       (select count(*) from graph_triples), (select max(created_at) from graph_triples),
                       (select count(*) from graph_edges), (select count(*) from graph_phrases)
                """
            )
            return tuple(await cur.fetchone())


async def _load_graph(stamp: tuple) -> KnowledgeGraph:
    start = time.time()
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("select id, phrase from graph_phrases")
            phrases = await cur.fetchall()
            await cur.execute("select id, embedding from documents order by filename, chunk_index")
            passages = await cur.fetchall()
            await cur.execute(
                """
                select f.subject_id, f.object_id, count(*)
                from graph_triples t join graph_facts f on f.id = t.fact_id
                group by f.subject_id, f.object_id
                """
            )
            fact_edges = await cur.fetchall()
            await cur.execute(
                """
                select distinct t.chunk_id, f.subject_id from graph_triples t join graph_facts f on f.id = t.fact_id
                union
                select distinct t.chunk_id, f.object_id from graph_triples t join graph_facts f on f.id = t.fact_id
                """
            )
            passage_edges = await cur.fetchall()
            await cur.execute("select phrase_a, phrase_b, weight from graph_edges")
            synonyms = await cur.fetchall()
            await cur.execute("select embedding from graph_facts order by random() limit %s", (FACT_SAMPLE,))
            fact_sample = await cur.fetchall()
    graph = await asyncio.to_thread(
        KnowledgeGraph.build, stamp, phrases, passages, fact_edges, passage_edges, synonyms, fact_sample,
        settings.embed_dim,
    )
    logger.info(
        "knowledge graph loaded: %d phrases, %d passages, %d edges, in %.1fs",
        len(phrases), len(passages), graph.graph.ecount(), time.time() - start,
    )
    return graph


async def current_graph() -> KnowledgeGraph:
    """The in-memory graph, rebuilt from Postgres when the tables have
    changed since it was built (one cheap query per call to find out)."""
    global _graph
    stamp = await _stamp()
    if _graph is not None and _graph.stamp == stamp:
        return _graph
    async with _graph_lock:
        if _graph is None or _graph.stamp != stamp:
            _graph = await _load_graph(stamp)
    return _graph


# ----------------------------------------------------------------------
# Query

# HippoRAG 2's fact filter prompt (its DSPy-tuned "recognition memory"
# step), with four of its ten worked examples.
_FILTER_SYSTEM = (
    "Your input fields are:\n"
    "1. `question` (str): Query for retrieval\n"
    "2. `fact_before_filter` (str): Candidate facts to be filtered\n\n"
    "Your output fields are:\n"
    "1. `fact_after_filter` (Fact): Filtered facts in JSON format\n\n"
    "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
    "[[ ## question ## ]]\n{question}\n\n"
    "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
    "[[ ## fact_after_filter ## ]]\n{fact_after_filter}        # note: the value you produce must be "
    'pareseable according to the following JSON schema: {"type": "object", "properties": {"fact": '
    '{"type": "array", "description": "A list of facts, each fact is a list of 3 strings: [subject, '
    'predicate, object]", "items": {"type": "array", "items": {"type": "string"}}, "title": "Fact"}}, '
    '"required": ["fact"], "title": "Fact"}\n\n'
    "[[ ## completed ## ]]\n\n"
    "In adhering to this structure, your objective is: \n"
    "        You are a critical component of a high-stakes question-answering system used by top "
    "researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to "
    "a given query, ensuring that the most crucial information is presented to these stakeholders. The "
    "query requires careful analysis and possibly multi-hop reasoning to connect different pieces of "
    "information. You must select up to 4 relevant facts from the provided candidate list that have a "
    "strong connection to the query, aiding in reasoning and providing an accurate answer. The output "
    'should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts '
    'are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it '
    "will directly impact the decisions made by these high-level stakeholders. You must only use facts "
    "from the candidate list and not generate new facts. The future of critical decision-making relies "
    "on your ability to accurately filter and present relevant information."
)
_FILTER_INPUT = (
    "[[ ## question ## ]]\n{question}\n\n"
    "[[ ## fact_before_filter ## ]]\n{facts}\n\n"
    "Respond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` "
    "(must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."
)
_FILTER_OUTPUT = "[[ ## fact_after_filter ## ]]\n{facts}\n\n[[ ## completed ## ]]"
_FILTER_EXAMPLES = [
    (
        "Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?",
        '{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", '
        '"united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", '
        '"ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}',
        '{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united states"],'
        '["amaradia","flows through","ro ia de amaradia"]]}',
    ),
    (
        "When is the director of film The Ancestor 's birthday?",
        '{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", '
        '"15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", '
        '"guido brignone"], ["benh zeitlin", "born on", "october 14  1982"]]}',
        '{"fact":[["the ancestor","directed by","guido brignone"]]}',
    ),
    (
        "How many households were in the city where Angelical Tears located?",
        '{"fact": [["dow city", "had", "219 households"], ["tucson", "had", "229 762 households"], '
        '["atlantic city", "has", "15 504 households"], ["angelical tears", "located in", "oklahoma city"], '
        '["atlantic city", "had", "15 848 households"]]}',
        '{"fact": [["angelical tears", "located in", "oklahoma city"]]}',
    ),
    (
        "Do both films: Gloria (1980 Film) and A New Life (Film) have the directors from the same country?",
        '{"fact": [["sebasti n lelio watt", "received acclaim for directing", "gloria"], ["gloria", "is", '
        '"1980 american thriller crime drama film"], ["a brand new life", "is directed by", "ounie lecomte"], '
        '["gloria", "written and directed by", "john cassavetes"], ["a new life", "directed by", "alan alda"]]}',
        '{"fact":[["gloria","is","1980 american thriller crime drama film"],["gloria","written and directed by",'
        '"john cassavetes"],["a new life","directed by","alan alda"]]}',
    ),
]


def _parse_filter_reply(reply: str) -> list[list[str]]:
    section = reply.split("[[ ## fact_after_filter ## ]]", 1)[-1].split("[[ ## completed ## ]]", 1)[0]
    data = _parse_json_reply(section)
    if isinstance(data, dict):
        data = data.get("fact", [])
    if not isinstance(data, list):
        return []
    return [[str(p) for p in item] for item in data if isinstance(item, (list, tuple)) and len(item) == 3]


async def _filter_facts(question: str, candidates: list[dict]) -> list[dict]:
    """One thinking-off call that keeps the candidate facts bearing on the
    question, in the order the model lists them. A reply naming a fact
    loosely is matched to the closest candidate; a failed call keeps
    nothing, which makes graph_search fall back to embedding similarity."""
    messages = [{"role": "system", "content": _FILTER_SYSTEM}]
    for question_example, before, after in _FILTER_EXAMPLES:
        messages.append({"role": "user", "content": _FILTER_INPUT.format(question=question_example, facts=before)})
        messages.append({"role": "assistant", "content": _FILTER_OUTPUT.format(facts=after)})
    before = json.dumps({"fact": [[c["subject"], c["relation"], c["object"]] for c in candidates]}, ensure_ascii=False)
    messages.append({"role": "user", "content": _FILTER_INPUT.format(question=question, facts=before)})
    try:
        reply = await generate_from_messages(messages, max_tokens=FILTER_MAX_TOKENS, temperature=0.0)
    except Exception:
        logger.warning("graph fact filter failed", exc_info=True)
        return []
    labels = [str([c["subject"], c["relation"], c["object"]]) for c in candidates]
    kept: list[dict] = []
    for fact in _parse_filter_reply(reply):
        closest = difflib.get_close_matches(str(fact), labels, n=1, cutoff=0.0)
        if closest:
            candidate = candidates[labels.index(closest[0])]
            if candidate not in kept:
                kept.append(candidate)
    return kept[:LINKING_TOP_K]


async def _closest_facts(query_vector: list[float], k: int) -> list[dict]:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select subject, relation, object, 1 - (embedding <=> %(v)s) as similarity
                from graph_facts
                order by embedding <=> %(v)s
                limit %(k)s
                """,
                {"v": Vector(query_vector), "k": k},
            )
            rows = await cur.fetchall()
    return [{"subject": s, "relation": r, "object": o, "similarity": float(sim)} for s, r, o, sim in rows]


async def _fetch_rows(ids: list) -> dict:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select id, content, source_type, source_format, filename, page_number, source_image_path, chunk_index
                from documents
                where id = any(%s)
                """,
                (ids,),
            )
            rows = await cur.fetchall()
    return {
        row_id: {
            "content": content,
            "source_type": source_type,
            "source_format": source_format,
            "filename": filename,
            "page_number": page_number,
            "source_image_path": source_image_path,
            "chunk_index": chunk_index,
        }
        for row_id, content, source_type, source_format, filename, page_number, source_image_path, chunk_index in rows
    }


async def graph_search(query_vector: list[float], query_text: str, limit: int) -> list[dict]:
    """The best `limit` chunks for the question by the graph (see the
    module docstring), as bare rows in the shape document_routed_search
    returns, for callers to fuse and expand_context. Empty when the graph
    has nothing in it."""
    graph = await current_graph()
    if not graph.passage_ids:
        return []
    query = np.asarray(query_vector, dtype=np.float32)
    query /= np.linalg.norm(query) or 1.0
    similarity = graph.passage_matrix @ query

    candidates = await _closest_facts(query_vector, LINKING_TOP_K) if graph.phrase_count else []
    kept = await _filter_facts(query_text, candidates) if candidates else []
    if kept:
        # Fact scores as HippoRAG normalises them: 1.0 for the best fact
        # in the graph, 0.0 for the least similar one.
        floor = float((graph.fact_sample @ query).min()) if len(graph.fact_sample) else 0.0
        top = max(c["similarity"] for c in candidates)
        phrase_weights = np.zeros(graph.phrase_count, dtype=np.float32)
        linking: dict[str, list[float]] = {}
        for fact in kept:
            score = (fact["similarity"] - floor) / (top - floor) if top > floor else 1.0
            for phrase in (fact["subject"], fact["object"]):
                vertex = graph.phrase_index.get(phrase)
                if vertex is not None:
                    # A phrase stated by many chunks is a weaker pointer to any one of them.
                    phrase_weights[vertex] = score / max(graph.chunk_counts[vertex], 1.0)
                linking.setdefault(phrase, []).append(score)
        mean_score = {phrase: sum(scores) / len(scores) for phrase, scores in linking.items()}
        seeds = {
            graph.phrase_index[p] for p in sorted(mean_score, key=mean_score.get, reverse=True)[:LINKING_TOP_K]
            if p in graph.phrase_index
        }
        mask = np.zeros(graph.phrase_count, dtype=bool)
        mask[list(seeds)] = True
        phrase_weights[~mask] = 0.0
        reset = np.concatenate([phrase_weights, _min_max(similarity) * PASSAGE_NODE_WEIGHT])
        scores = await asyncio.to_thread(graph.pagerank, reset) if reset.sum() > 0 else similarity
    else:
        scores = similarity

    order = np.argsort(-scores)[:limit]
    ids = [graph.passage_ids[i] for i in order]
    rows = await _fetch_rows(ids)
    return [rows[i] for i in ids if i in rows]
