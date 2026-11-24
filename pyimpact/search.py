"""Code search over functions, methods and classes.

Two engines, one interface:

- **Lexical (default, offline):** BM25 over code-aware tokens — identifiers are
  split on snake_case and camelCase and lightly stemmed, so "hash password"
  finds ``hash_password``/``PasswordHasher``. Names weigh more than docstrings,
  which weigh more than the identifiers used in the body.
- **Semantic (``--semantic``, needs a Gemini key):** embeddings of each symbol's
  signature, docstring and source, cached on disk by content hash so only new or
  changed code is ever re-embedded. Results are fused with BM25 using reciprocal
  rank fusion, which keeps exact-name hits on top while adding meaning-based ones.

Brute-force cosine similarity is exact and fast enough at repository scale
(tens of thousands of vectors), so no vector database is needed.
"""

from __future__ import annotations

import array
import ast
import hashlib
import math
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

import networkx as nx

from .call_graph import _FUNC_TYPES, _toplevel

SEARCHABLE = frozenset({"function", "method", "class"})
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = frozenset("""
a an and are as at be by for from has have if in into is it its not of on or so the
this that to was were will with self cls none true false return def class import
get set new do does use used using via all any
""".split())
_SUFFIXES = ("ization", "ations", "ation", "ings", "ing", "ies", "es", "ed", "ly", "s")
FIELD_WEIGHTS = {"name": 4, "doc": 2, "module": 1, "params": 1, "body": 1}


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Split identifiers/prose into lowercase, stemmed search terms."""
    tokens = []
    for word in _WORD.findall(text or ""):
        word = word.lower()
        if len(word) < 2 or word in _STOP:
            continue
        tokens.append(_stem(word))
    return tokens


@dataclass
class Hit:
    node: str
    kind: str
    score: float
    file: str | None
    line: int | None
    signature: str = ""
    doc: str = ""


def _def_index(graph: nx.DiGraph) -> dict:
    """(module, lineno) -> def/class AST for every indexed definition."""
    index = graph.graph.get("index")
    found = {}
    if index is None:
        return found
    for sf in index.parsed():
        for stmt in _toplevel(sf.tree.body):
            if isinstance(stmt, (*_FUNC_TYPES, ast.ClassDef)):
                found[(sf.module, stmt.lineno)] = stmt
            if isinstance(stmt, ast.ClassDef):
                for item in _toplevel(stmt.body):
                    if isinstance(item, _FUNC_TYPES):
                        found[(sf.module, item.lineno)] = item
    return found


def _body_terms(node) -> list[str]:
    words = []
    body = node.body if not isinstance(node, ast.ClassDef) else \
        [s for s in node.body if not isinstance(s, _FUNC_TYPES)]
    for sub in (n for stmt in body for n in ast.walk(stmt)):
        if isinstance(sub, ast.Name):
            words.append(sub.id)
        elif isinstance(sub, ast.Attribute):
            words.append(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and len(sub.value) < 80:
            words.append(sub.value)
        if len(words) > 400:
            break
    return words


class SearchIndex:
    """BM25 index over the searchable nodes of a unified graph."""

    k1 = 1.5
    b = 0.75

    def __init__(self, graph: nx.DiGraph, include_tests: bool = False):
        self.graph = graph
        defs = _def_index(graph)
        self.docs: dict[str, Counter] = {}
        self.texts: dict[str, str] = {}
        for node, data in graph.nodes(data=True):
            if data.get("kind") not in SEARCHABLE or (data.get("is_test") and not include_tests):
                continue
            tree = defs.get((data["module"], data.get("lineno")))
            doc = (ast.get_docstring(tree) if tree is not None else None) or data.get("doc", "")
            fields = {
                "name": data["qualname"],
                "doc": doc or "",
                "module": data["module"],
                "params": data.get("signature", ""),
                "body": " ".join(_body_terms(tree)) if tree is not None else "",
            }
            counts: Counter = Counter()
            for name, text in fields.items():
                for token in tokenize(text):
                    counts[token] += FIELD_WEIGHTS[name]
            if counts:
                self.docs[node] = counts
                self.texts[node] = (f"{data['kind']} {data['qualname']} in {data['module']}"
                                    f"{data.get('signature', '')}\n{doc or ''}")
        self.lengths = {n: sum(c.values()) for n, c in self.docs.items()}
        self.avg_len = (sum(self.lengths.values()) / len(self.lengths)) if self.lengths else 1.0
        df: Counter = Counter()
        for counts in self.docs.values():
            df.update(counts.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self):
        return len(self.docs)

    def _hit(self, node: str, score: float) -> Hit:
        data = self.graph.nodes[node]
        return Hit(node, data["kind"], round(score, 4), data.get("rel_path"),
                   data.get("lineno"), data.get("signature", ""), data.get("doc", ""))

    def search(self, query: str, k: int = 10) -> list[Hit]:
        terms = tokenize(query)
        if not terms:
            return []
        scores: dict[str, float] = {}
        for node, counts in self.docs.items():
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[node] / self.avg_len)
            for t in terms:
                tf = counts.get(t)
                if tf:
                    score += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            if score:
                name_terms = set(tokenize(self.graph.nodes[node]["qualname"]))
                if set(terms) <= name_terms:
                    score *= 1.5                      # every query word is in the name
                fan_in = self.graph.in_degree(node)
                scores[node] = score * (1 + 0.05 * math.log1p(fan_in))
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [self._hit(node, score) for node, score in ranked]


# -- semantic search -----------------------------------------------------------------

def default_cache_path() -> str:
    base = os.environ.get("PYIMPACT_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME") \
        or os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "pyimpact", "embeddings.sqlite3")


class EmbeddingCache:
    """content-hash -> vector, in SQLite (stdlib), so re-runs are free."""

    def __init__(self, path: str | None = None):
        self.path = path or default_cache_path()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for key, blob in self.db.execute(
                    f"SELECT key, vec FROM vectors WHERE key IN ({marks})", chunk):
                found[key] = array.array("f", blob).tolist()
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                            [(k, array.array("f", v).tobytes()) for k, v in items.items()])
        self.db.commit()

    def close(self):
        self.db.close()


def _cosine_rank(query: list[float], vectors: dict[str, list[float]]) -> list[tuple[str, float]]:
    names = list(vectors)
    try:
        import numpy as np

        matrix = np.asarray([vectors[n] for n in names], dtype="float32")
        q = np.asarray(query, dtype="float32")
        sims = matrix @ q / ((np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)) + 1e-9)
        scores = sims.tolist()
    except ImportError:
        qn = math.sqrt(sum(x * x for x in query)) or 1.0
        scores = []
        for n in names:
            v = vectors[n]
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            scores.append(sum(a * b for a, b in zip(query, v)) / (qn * vn))
    return sorted(zip(names, scores), key=lambda kv: -kv[1])


def _source(graph, node, limit=1500) -> str:
    data = graph.nodes[node]
    try:
        with open(data["path"], encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, KeyError):
        return ""
    start, end = data.get("start_lineno") or data.get("lineno") or 1, data.get("end_lineno")
    return "".join(lines[start - 1:end])[:limit]


def semantic_search(index: SearchIndex, query: str, k: int = 10, *, embed_fn=None,
                    model: str | None = None, cache: EmbeddingCache | None = None,
                    hybrid: bool = True) -> list[Hit]:
    """Embedding search (optionally fused with BM25). ``embed_fn(texts,
    task_type) -> vectors`` is injectable; default uses Gemini."""
    from . import llm

    model_name = llm.embed_model(model)
    if embed_fn is None:
        def embed_fn(texts, task_type):
            return llm.embed(texts, model=model_name, task_type=task_type)

    texts = {n: f"{index.texts[n]}\n{_source(index.graph, n)}" for n in index.docs}
    keys = {n: hashlib.sha256(f"{model_name}\n{t}".encode()).hexdigest() for n, t in texts.items()}
    own_cache = cache is None
    cache = cache or EmbeddingCache()
    try:
        stored = cache.get_many(list(keys.values()))
        missing = [n for n in texts if keys[n] not in stored]
        if missing:
            vectors = embed_fn([texts[n] for n in missing], "RETRIEVAL_DOCUMENT")
            fresh = {keys[n]: v for n, v in zip(missing, vectors)}
            cache.put_many(fresh)
            stored.update(fresh)
        doc_vectors = {n: stored[keys[n]] for n in texts if keys[n] in stored}
    finally:
        if own_cache:
            cache.close()
    query_vec = embed_fn([query], "RETRIEVAL_QUERY")[0]
    semantic = _cosine_rank(query_vec, doc_vectors)
    if not hybrid:
        return [index._hit(n, s) for n, s in semantic[:k]]

    fused: dict[str, float] = {}
    for rank, (node, _) in enumerate(semantic[:100]):
        fused[node] = fused.get(node, 0.0) + 1 / (60 + rank)
    for rank, hit in enumerate(index.search(query, k=100)):
        fused[hit.node] = fused.get(hit.node, 0.0) + 1 / (60 + rank)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [index._hit(n, s * 100) for n, s in ranked]
