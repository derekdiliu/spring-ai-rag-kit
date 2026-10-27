# Spring AI RAG Kit

Hybrid retrieval for [Spring AI](https://github.com/spring-projects/spring-ai): vector similarity **plus** PostgreSQL full-text search, merged with Reciprocal Rank Fusion, on the table `PgVectorStore` already writes. No second datastore, no Elasticsearch, no data duplication.

```java
@Bean
RetrievalAugmentationAdvisor ragAdvisor(HybridDocumentRetriever hybrid) {   // auto-configured
    return RetrievalAugmentationAdvisor.builder().documentRetriever(hybrid).build();
}
```

## Why

Spring AI 2.0's modular RAG defines the right extension points, but ships few implementations for them:

| Extension point | What Spring AI 2.0.2 ships | What this kit adds |
|---|---|---|
| `DocumentRetriever` | `VectorStoreDocumentRetriever` only | `PgFullTextDocumentRetriever`, `HybridDocumentRetriever` |
| `DocumentJoiner` | `ConcatenationDocumentJoiner` (sorts by raw score) | `RankFusionDocumentJoiner` (RRF) |
| `RetrievalAugmentationAdvisor` | accepts **one** retriever | `HybridDocumentRetriever` *is* one retriever wrapping many |

Vector search misses exact tokens — method names, error codes, SKUs, acronyms. Full-text search misses paraphrase. Hybrid retrieval can cover both, and RRF merges them using ranks only, so cosine similarities and `ts_rank` scores never need to be put on the same scale. Whether it helps depends on your data — see the [benchmark](#benchmark), and measure your own with `rag-kit-eval`.

## What's in it

**`rag-kit-core`** — store-agnostic
- `RankFusion.rrf()` — Reciprocal Rank Fusion (default `k = 60`), with per-source weights
- `RankFusion.weightedScore()` — weighted sum of min-max normalised scores
- `HybridDocumentRetriever` — runs delegates in parallel (on virtual threads on Java 21+), fuses, applies `topK`
- `RankFusionDocumentJoiner` — drop-in replacement for `ConcatenationDocumentJoiner` when using query expansion

Every fused document carries `ragkit_fusion_score` and `ragkit_source_ranks` (e.g. `{vector=3, fulltext=1}`) in its metadata, so you can see *why* it ranked where it did.

**`rag-kit-pgvector`** — `PgFullTextDocumentRetriever`
- Reads `PgVectorStore`'s table (`public.vector_store` by default)
- `initializeSchema()` adds a generated, stored `tsvector` column and GIN index — documents ingested through `PgVectorStore.add(...)` are indexed automatically
- **Identifier splitting**: `RecipeService.createRecipe` is indexed as the whole token *and* as `recipe service create recipe`, so a question mentioning `createRecipe` finds it. PostgreSQL's default parser treats the dotted name as one token and misses it.
- `ANY_TERMS` mode (default) for natural-language questions; `WEB_SEARCH` mode for `"phrases"`, `or`, `-exclusions`
- **BM25 ranking** with `ranking(TextRanking.BM25)` (`ragkit.fulltext.ranking: bm25`), through the [`pg_textsearch`](https://github.com/timescale/pg_textsearch) extension: `initializeSchema()` then creates a `bm25` index on the content instead of the column — no table rewrite. Identifier splitting and filters work the same; `ANY_TERMS` mode only
- Metadata filters via Spring AI's own `PgVectorFilterExpressionConverter`, read from the same `FILTER_EXPRESSION` query-context key as `VectorStoreDocumentRetriever` — one filter applies to both halves of a hybrid search
- Identifiers validated, user text always bound as a parameter

**`rag-kit-spring-boot-starter`** — auto-configures both beans next to Spring AI's pgvector auto-configuration.

- Searches the table set by `spring.ai.vectorstore.pgvector.schema-name` / `table-name` unless `ragkit.fulltext.*` overrides it
- Needs a single (or `@Primary`) `JdbcTemplate`; the hybrid bean also needs a single `PgVectorStore`. If you define the store yourself, declare the `@Bean` method's return type as `PgVectorStore`, not `VectorStore`

**`rag-kit-eval`** — retrieval quality as a test, for any `DocumentRetriever`
- `Metric.RECALL`, `MRR`, `NDCG`, `HIT_RATE` at any cut-off, defined as in trec_eval / BEIR so numbers are comparable with published results
- `RetrievalEvaluator` runs a list of `EvalCase`s and returns an `EvalReport`; map documents to your ids with `.documentId(...)` when the store assigns its own or splits documents into chunks
- `@EvalDataset` injects cases from a JSON Lines file into a JUnit 5 test; a failed threshold lists the worst queries

```java
@Test
void retrievalQuality(@EvalDataset("classpath:eval/questions.jsonl") List<EvalCase> cases) {
    EvalReport report = RetrievalEvaluator.of(hybridRetriever).evaluate(cases);
    report.assertThat(Metric.RECALL, 5).isAtLeast(0.80);
    report.assertThat(Metric.MRR, 10).isAtLeast(0.60);
}
```

```json
{"id": "q1", "query": "How do I reset my password?", "relevant": ["doc-12", "doc-40"]}
{"id": "q2", "query": "Refund after 30 days", "relevant": {"doc-7": 2, "doc-9": 1}}
```

## Quick start

```xml
<dependency>
    <groupId>io.github.derekdiliu</groupId>
    <artifactId>rag-kit-spring-boot-starter</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

```yaml
ragkit:
  fulltext:
    initialize-schema: true      # adds content_tsv + GIN index on startup (off by default)
    text-search-config: english  # or 'simple' for code-heavy / mixed-language content
    ranking: bm25                # needs pg_textsearch; default ts-rank works on any PostgreSQL
  hybrid:
    top-k: 5
    vector-top-k: 20
    rrf-k: 60
```

Without Spring Boot:

```java
var fullText = PgFullTextDocumentRetriever.builder(jdbcTemplate).topK(20).build();
fullText.initializeSchema();   // once; requires PgVectorStore's table to exist

var hybrid = HybridDocumentRetriever.builder()
    .retriever("vector", VectorStoreDocumentRetriever.builder().vectorStore(pgVectorStore).topK(20).build())
    .retriever("fulltext", fullText)
    .topK(5)
    .build();
```

## Benchmark

BEIR test splits, all retrievers on the same `PgVectorStore` table. Every retriever returns its top 100; hybrid is equal-weight RRF (`k = 60`) over vector and full-text.

| Dataset | Retriever | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| SciFact | Full-text (`ts_rank`) | 0.597 | 0.742 | 0.862 | 0.556 |
| SciFact | Full-text (BM25) | 0.691 | 0.814 | 0.926 | 0.658 |
| SciFact | Vector (all-MiniLM-L6-v2) | 0.624 | 0.774 | 0.923 | 0.578 |
| SciFact | Hybrid (all-MiniLM-L6-v2 + `ts_rank`) | 0.678 | 0.797 | 0.939 | 0.644 |
| SciFact | Hybrid (all-MiniLM-L6-v2 + BM25) | 0.704 | 0.855 | 0.955 | 0.657 |
| SciFact | Vector (text-embedding-3-small) | 0.723 | 0.852 | 0.970 | 0.690 |
| SciFact | Hybrid (text-embedding-3-small + `ts_rank`) | 0.716 | 0.816 | 0.967 | 0.697 |
| FiQA | Full-text (`ts_rank`) | 0.188 | 0.242 | 0.473 | 0.238 |
| FiQA | Full-text (BM25) | 0.254 | 0.314 | 0.547 | 0.313 |
| FiQA | Vector (all-MiniLM-L6-v2) | 0.366 | 0.427 | 0.694 | 0.443 |
| FiQA | Hybrid (all-MiniLM-L6-v2 + `ts_rank`) | 0.330 | 0.416 | 0.680 | 0.395 |
| FiQA | Hybrid (all-MiniLM-L6-v2 + BM25) | 0.366 | 0.445 | 0.700 | 0.439 |
| FiQA | Vector (text-embedding-3-small) | 0.449 | 0.524 | 0.775 | 0.520 |
| FiQA | Hybrid (text-embedding-3-small + `ts_rank`) | 0.348 | 0.432 | 0.733 | 0.414 |

`text-embedding-3-small` with BM25 has not been run yet.

What it shows:

- **Hybrid helps when the two retrievers are comparably strong.** On SciFact, full-text and MiniLM are close, and fusing them beats both: 0.678 with `ts_rank`, 0.704 with BM25.
- **It hurts when one is much weaker.** On FiQA, `ts_rank` reaches 0.188 against 0.449 for `text-embedding-3-small`, and equal-weight fusion drags the result down to 0.348.
- **BM25 closes most of that gap.** It scores 0.691 on SciFact and 0.254 on FiQA (BEIR reports 0.665 and 0.236 for its BM25 baseline). With it, MiniLM hybrid on FiQA no longer loses to vector alone (0.366 either way, with higher recall), and on SciFact it is the best MiniLM result.
- **With `ts_rank`, a lower full-text weight recovers FiQA, at a cost.** Sweeping `ragkit.hybrid.fulltext-weight` with MiniLM (nDCG@10):

  | fulltext-weight | 1.0 (default) | 0.7 | 0.5 | 0.3 | 0.1 | vector only |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | SciFact | 0.678 | **0.683** | 0.678 | 0.676 | 0.661 | 0.624 |
  | FiQA | 0.330 | 0.343 | 0.353 | 0.363 | **0.368** | 0.366 |

  The cost: a document found only by full-text scores at most `w / 61`, so it can only enter the top `n` if `w > 61 / (60 + n)` — above 0.38 for the top 100 here, and above 0.94 for the starter's default `top-k: 5`. Below that, hybrid just re-orders the vector results (Recall@100 equals vector-only at 0.3 and below) and gives up the exact-token matches vector search misses, which BEIR's natural-language queries don't test. The default stays 1.0; check a lower weight against your own queries with `rag-kit-eval`.
- **Without the extension, `ts_rank` still gets most of the way to BM25** (0.597 vs 0.691, 0.188 vs 0.254). The cover-density `ts_rank_cd` reached only 0.33 and 0.03 on the same queries, which is why the retriever uses `ts_rank`.

Setup: vector search is exact (no ANN index), so numbers reflect the model, not index recall. MiniLM runs locally through `spring-ai-transformers` with Spring AI's default 128-token truncation. Reproduce with:

```bash
mvn -Pbenchmark -pl rag-kit-benchmark -am verify -DskipTests \
    -Dbenchmark.datasets=scifact,fiqa -Dbenchmark.models=minilm,openai   # openai needs OPENAI_API_KEY
```

## Requirements

Java 17+ (virtual threads on 21+) · Spring AI 2.0.x · Spring Boot 4.1.x · PostgreSQL 12+ (generated columns); BM25 ranking needs PostgreSQL 17+ with `pg_textsearch` in `shared_preload_libraries`

## Notes and limits

- `initializeSchema()` adds a **stored** generated column, which rewrites the table once. On large tables run it in a maintenance window. Set `useStoredTsvector(false)` to try the retriever with no schema change (sequential scan).
- The query `tsquery` is built in a fenced one-row subquery (`OFFSET 0`). Without the fence PostgreSQL inlines it and ignores the GIN index — measured at 45 ms vs 0.5 ms on 20k rows.
- `ts_rank` is not BM25: it has no inverse document frequency, so common query words weigh as much as rare ones. Use BM25 ranking where you can install `pg_textsearch`.
- Chinese text needs a segmentation extension (e.g. `zhparser`) and a matching text search configuration.

## Build

```bash
mvn verify          # unit tests + Testcontainers integration tests (pgvector/pgvector:pg17)
```

The BM25 integration tests build an image from `pgvector/pgvector:pg17` and the `pg_textsearch` release package, downloaded from GitHub and checked against a pinned SHA-256.

Integration tests are skipped automatically when Docker is not available.

## Roadmap

- [x] RRF / weighted fusion, hybrid retriever, fusion joiner
- [x] PostgreSQL full-text retriever with identifier splitting and shared filters
- [x] Spring Boot auto-configuration
- [x] `rag-kit-eval`: Recall@k, MRR, nDCG, hit rate; JUnit 5 support for RAG regression tests in CI
- [ ] Rerankers as `DocumentPostProcessor` (Spring AI ships none)
- [x] BM25 ranking via `pg_textsearch`
- [x] Benchmark on BEIR (SciFact, FiQA): vector vs full-text vs hybrid
- [ ] Publish to Maven Central

## License

Apache 2.0
