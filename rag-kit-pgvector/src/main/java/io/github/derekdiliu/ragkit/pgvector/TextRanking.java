package io.github.derekdiliu.ragkit.pgvector;

/**
 * How matching documents are scored and ordered.
 */
public enum TextRanking {

	/**
	 * PostgreSQL's built-in {@code ts_rank} with log-length normalisation, over the
	 * generated {@code tsvector} column and its GIN index. Works on any PostgreSQL 12+.
	 * Has no inverse document frequency, so common query terms weigh as much as rare ones.
	 */
	TS_RANK,

	/**
	 * Okapi BM25 from the
	 * <a href="https://github.com/timescale/pg_textsearch">pg_textsearch</a> extension,
	 * over a {@code bm25} index on the content; no extra column. Needs PostgreSQL 17+ with
	 * {@code shared_preload_libraries = 'pg_textsearch'}. Scores are the (positive) BM25
	 * score. Supports {@link TextQueryMode#ANY_TERMS} only.
	 */
	BM25

}
