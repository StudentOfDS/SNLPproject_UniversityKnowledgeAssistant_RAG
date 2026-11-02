# Evaluation

Run it with `python -m evaluation.evaluate` (add `--save` to write `evaluation/latest_results.json`).

The benchmark is `evaluation/benchmark.json`: 48 questions whose answers are in the three PDFs in
`documents/`, and 18 questions whose answers are not (general knowledge, plus a few that sound like
university questions but have no answer in the files). A retrieved chunk counts as relevant when it
contains one of the listed phrases from the source text, so the benchmark still works after you change the
chunk size. 12 of the answerable questions are yes/no questions with an expected Yes or No.

## Read this first

- All numbers below were measured with **keyword and language-model search only**. The machine used for
  the final checks could not download the embedding model, so embedding search was **not measured**, and no
  Gemini key was available, so the answers are the sentence-based ones.
- The benchmark was written by the team that built the system and was used while tuning it (merging tiny
  sections, adding the document name to the index text, the router). The numbers are therefore optimistic.
  They are a sanity check, not a general accuracy figure.
- Embedding thresholds (`SEMANTIC_STRONG`, `SEMANTIC_FLOOR`) were never calibrated on real embeddings.

## Retrieval: what each component adds

Share of answerable questions with a relevant chunk in the top 5 (recall@5) and mean reciprocal rank (MRR).

| Setup | recall@5 | MRR |
| --- | --- | --- |
| TF-IDF only | 0.854 | 0.628 |
| Language model only | 0.854 | 0.679 |
| TF-IDF + language model (rank fusion) | 0.833 | 0.639 |
| + Naive Bayes document router | 0.875 | 0.750 |

Fusing the two rankings alone did not help on this benchmark. The router is what moves the first relevant
passage up (MRR 0.639 to 0.750). Changing `ROUTER_BOOST` did not change the results much.

### Stemmers (TF-IDF only)

| Stemmer | recall@5 | MRR |
| --- | --- | --- |
| Snowball (default) | 0.854 | 0.628 |
| Porter | 0.854 | 0.628 |
| Lancaster | 0.833 | 0.609 |
| none | 0.833 | 0.614 |

### Smoothing of the language model (language model only)

| Smoothing | recall@5 | MRR |
| --- | --- | --- |
| Dirichlet (default) | 0.854 | 0.679 |
| Jelinek-Mercer | 0.854 | 0.688 |
| Laplace | 0.729 | 0.516 |

Laplace smoothing is clearly worse: adding one to every count gives the whole vocabulary too much
probability mass compared with short passages. Dirichlet and Jelinek-Mercer are within noise of each other.

## Whole pipeline (sentence answers, no Gemini)

| | Result |
| --- | --- |
| Answerable questions that got an answer | 41 of 48 |
| Answers that contain the expected keyword | 33 of 41 |
| Answers whose citations include a relevant passage | 38 of 41 |
| Unanswerable questions refused | 17 of 18 |
| Yes/No questions among the answered ones: correct verdict / wrong verdict / no verdict | 5 / 1 / 5 |

Questions that were refused although the answer is in the files:

- "What is the minimum attendance needed to sit the term-end exam?" and "Can a student with 75% attendance
  still appear for the exam?": the document says "appear for the examination", which keyword matching does
  not link to "sit the exam".
- "What happens if a student repeats exam malpractice?", "Is plagiarism treated as misconduct?": reworded.
- "When is the Diwali vacation?" and "When does the winter vacation start?": the scanned calendar was read
  as "Diwalivacation" and "December26", so the words never match. This is an OCR problem.
- "What is the date of Gudhi Padwa?": the spelling in the holiday list differs.

The one unanswerable question that got an answer is "What is the syllabus of the data structures course?".
The files contain the words "course" and "syllabus" many times, so the check cannot tell it apart from a real
question. A Gemini key helps here, because the model is allowed to reply that the passages do not hold the answer (not tested here).

Embedding search should fix most of the wording problems above. That is the reason it is part of the
hybrid search, but it was not measured here.

## Measured and removed

Each of these was connected to the pipeline, compared on the same benchmark and dropped because it did not
help (or made things slightly worse): part-of-speech weighting of query words, a dependency head boost,
noun phrases from a PCFG, WordNet and PPMI-vector query expansion, and n-gram perplexity as an off-topic
signal for the refusal check. For reference, when those modules were still stand-alone demos they measured:
HMM tagger 93.6% on held-out Penn Treebank sentences, dependency parser about 67% unlabelled attachment,
Naive Bayes sentiment 81% on movie reviews. None of that is relevant to the final system's accuracy.

## Tuning the check

`python scripts/check_question.py "your question"` prints the numbers behind the decision for each
question. If valid questions are refused, lower `MIN_STRENGTH_*` or `MIN_TERM_COVERAGE` in `.env`.
If wrong questions get answers, raise them.
