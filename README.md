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

Vector search misses exact tokens — method names, error codes, SKUs, acronyms. Full-text search misses paraphrase. Hybrid retrieval gets both, and RRF merges them using ranks only, so cosine similarities and `ts_rank` scores never need to be put on the same scale.

## What's in it

**`rag-kit-core`** — store-agnostic
- `RankFusion.rrf()` — Reciprocal Rank Fusion (default `k = 60`), with per-source weights
- `RankFusion.weightedScore()` — weighted sum of min-max normalised scores
- `HybridDocumentRetriever` — runs delegates in parallel on virtual threads, fuses, applies `topK`
- `RankFusionDocumentJoiner` — drop-in replacement for `ConcatenationDocumentJoiner` when using query expansion

Every fused document carries `ragkit_fusion_score` and `ragkit_source_ranks` (e.g. `{vector=3, fulltext=1}`) in its metadata, so you can see *why* it ranked where it did.

**`rag-kit-pgvector`** — `PgFullTextDocumentRetriever`
- Reads `PgVectorStore`'s table (`public.vector_store` by default)
- `initializeSchema()` adds a generated, stored `tsvector` column and GIN index — documents ingested through `PgVectorStore.add(...)` are indexed automatically
- **Identifier splitting**: `RecipeService.createRecipe` is indexed as the whole token *and* as `recipe service create recipe`, so a question mentioning `createRecipe` finds it. PostgreSQL's default parser treats the dotted name as one token and misses it.
- `ANY_TERMS` mode (default) for natural-language questions; `WEB_SEARCH` mode for `"phrases"`, `or`, `-exclusions`
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

## Requirements

Java 17+ (virtual threads on 21+) · Spring AI 2.0.x · Spring Boot 4.1.x · PostgreSQL 12+ (generated columns)

## Notes and limits

- `initializeSchema()` adds a **stored** generated column, which rewrites the table once. On large tables run it in a maintenance window. Set `useStoredTsvector(false)` to try the retriever with no schema change (sequential scan).
- The query `tsquery` is built in a fenced one-row subquery (`OFFSET 0`). Without the fence PostgreSQL inlines it and ignores the GIN index — measured at 45 ms vs 0.5 ms on 20k rows.
- `ts_rank` is not BM25: it has no inverse document frequency, so common query words weigh as much as rare ones. For true BM25 ranking, the `pg_textsearch` / ParadeDB extensions are on the roadmap as an alternative backend.
- Chinese text needs a segmentation extension (e.g. `zhparser`) and a matching text search configuration.

## Build

```bash
mvn verify          # unit tests + Testcontainers integration tests (pgvector/pgvector:pg17)
```

Integration tests are skipped automatically when Docker is not available.

## Roadmap

- [x] RRF / weighted fusion, hybrid retriever, fusion joiner
- [x] PostgreSQL full-text retriever with identifier splitting and shared filters
- [x] Spring Boot auto-configuration
- [x] `rag-kit-eval`: Recall@k, MRR, nDCG, hit rate; JUnit 5 support for RAG regression tests in CI
- [ ] Rerankers as `DocumentPostProcessor` (Spring AI ships none)
- [ ] BM25 backend via `pg_textsearch`
- [ ] Benchmark on BEIR (SciFact, FiQA): vector vs full-text vs hybrid
- [ ] Publish to Maven Central

## License

Apache 2.0
