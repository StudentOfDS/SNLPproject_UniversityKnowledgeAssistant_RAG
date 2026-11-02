# API reference

Base URL: `http://127.0.0.1:8000/api/v1`. Interactive docs are at `/docs` while the server runs.

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health` | Status, number of documents, whether an LLM key, embedding search and an OCR engine are available |
| GET | `/documents` | Indexed documents with page count, chunk count and which pages were read with OCR |
| GET | `/corpus` | Statistics of the whole collection, keywords and a summary for each document, and the topics found |
| POST | `/reindex` | Reads any new file in `documents/` without a restart |
| POST | `/index` | Upload one or more PDFs or images (`files`, multipart) |
| POST | `/search` | Retrieval only. Form fields: `question`, `mode`, `top_k` |
| POST | `/ask` | Answer a question. Form fields: `question`, `mode`, `history`, optional `files` |

`mode` is `hybrid` (default), `tfidf`, `lm` or `semantic`. Hybrid merges all rankings that are switched on.
`lm` is the smoothed language model on its own. `semantic` needs the embedding model.
`history` is a JSON list of earlier `{"question": ..., "answer": ...}` turns. Only a short follow-up such as
"and refunds?" uses it.

Supported files: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`.

## `/ask` response

```json
{
  "query": "What attendance do I need?",
  "answer": "From SRB.pdf, page 10:\n\nStudents must ... [1]",
  "retrieval_score": 0.72,
  "confidence": "MEDIUM",
  "evidence_sufficient": true,
  "reason": "The retrieved passages cover the question.",
  "signals": {"known_term_ratio": 1.0, "term_coverage": 1.0, "semantic_similarity": null, "tfidf_similarity": 0.22},
  "analysis": {"question": "...", "question_type": "quantity", "terms": ["attend"]},
  "likely_documents": [{"document": "SRB.pdf", "probability": 0.82}],
  "suggestions": [],
  "citations": [
    {"document_name": "SRB.pdf", "page": 10, "section": "3.2.1 Eligibility", "score": 0.22,
     "text": "...", "highlights": ["Students must keep ..."]}
  ],
  "processed_documents": ["SRB.pdf"]
}
```

- `analysis` is how the question was read: its type and the stemmed words that were searched for.
- `likely_documents` is the Naive Bayes guess of which document the question is about. The probabilities
  add up to 1 over the indexed documents.
- `highlights` are the sentences of that passage the answer is based on.
- `suggestions` is only filled when the question is refused: topics the files do cover.

When the documents do not contain the answer, you get `evidence_sufficient: false`,
`confidence: "LOW"`, an empty `citations` list and this answer text:

> Sorry, I couldn't find this in the documents you've provided. Try rephrasing the question, or add the document that covers it.

`reason` says why (for example that most of the question's words never appear in the documents).
`retrieval_score` is the strength of the evidence on a 0 to 1 scale. It is not a probability that the answer is right.

## `/corpus` response

```json
{
  "collection": {"sentences": 2286, "tokens": 48675, "types": 4350, "type_token_ratio": 0.0894,
                 "zipf_slope": -0.979, "top_words": [["student", 371]], "top_bigrams": [["academic year", 99]]},
  "documents": [{"document_name": "SRB.pdf", "keywords": ["student", "course"], "summary": ["..."]}],
  "topics": [{"label": "examination, student, re, course", "words": ["..."], "passages": 96,
              "documents": {"SRB.pdf": 91}}],
  "topic_silhouette": 0.13
}
```

It returns 400 when nothing is indexed. The result is cached until the set of documents changes.

## Errors

400 for an empty question, a bad `mode`, an unsupported or broken file, or when nothing is indexed.
500 for anything unexpected (see the server log).
