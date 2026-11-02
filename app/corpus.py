"""Facts about the indexed collection that are cached until the set of documents changes."""
from __future__ import annotations

import threading

from app.knowledge_base import KnowledgeBase
from app.snlp import textproc
from app.topics import TopicModel, build_topics, document_profile


class CorpusInsights:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.kb = knowledge_base
        self._key: tuple = ()
        self._topics: TopicModel | None = None
        self._profile: dict | None = None
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        key = tuple(sorted(d["file_hash"] for d in self.kb.list_documents()))
        if key != self._key:
            self._key, self._topics, self._profile = key, None, None

    def topics(self) -> TopicModel | None:
        with self._lock:
            self._refresh()
            if self._topics is None and self._key:
                self._topics = build_topics(self.kb.all_chunks())
            return self._topics

    def profile(self) -> dict:
        topic_model = self.topics()
        with self._lock:
            self._refresh()
            if self._profile is None:
                documents = []
                for doc in self.kb.list_documents():
                    chunks = self.kb.all_chunks([doc["file_hash"]])
                    documents.append({"document_name": doc["document_name"], **document_profile(chunks, topic_model)})
                everything = [c.text for c in self.kb.all_chunks()]
                self._profile = {
                    "collection": textproc.first_look(everything) if everything else {},
                    "documents": documents,
                    "topics": topic_model.topics if topic_model else [],
                    "topic_silhouette": topic_model.silhouette if topic_model else None,
                }
            return self._profile
