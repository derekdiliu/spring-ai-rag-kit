package io.github.derekdiliu.ragkit.pgvector;

import org.junit.jupiter.api.Test;

import org.springframework.jdbc.core.JdbcTemplate;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

/**
 * SQL generation and input validation, without a database. Behaviour against a real
 * PostgreSQL is covered by {@link PgFullTextDocumentRetrieverIT}.
 */
class PgFullTextDocumentRetrieverTests {

	private final JdbcTemplate jdbcTemplate = new JdbcTemplate();

	@Test
	void defaultsTargetPgVectorStoreTable() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate).build().buildSearchSql("");

		assertThat(sql).contains(", \"public\".\"vector_store\" WHERE")
			.contains("\"content_tsv\" @@ t.q")
			.contains("ts_rank(\"content_tsv\", t.q, 1|32)")
			.contains("'english'::regconfig")
			.endsWith("ORDER BY ragkit_score DESC LIMIT ?");
	}

	@Test
	void anyTermsModeOrsQuotedLexemesAndSplitsIdentifiers() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate).build().buildSearchSql("");

		assertThat(sql).contains("quote_literal(lexeme)")
			.contains("' | '")
			.contains("AS tsquery)")
			.contains("regexp_replace(input.qt, '([a-z0-9])([A-Z])', '\\1 \\2', 'g')");
	}

	@Test
	void buildsTsqueryOnceBehindAnOptimisationFence() {
		// Without OFFSET 0 PostgreSQL inlines the subquery, rebuilds the tsquery per row
		// and falls back to a sequential scan instead of the GIN index.
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate).build().buildSearchSql("");

		assertThat(sql).contains("FROM (SELECT CAST(? AS text) AS qt) AS input OFFSET 0) AS t, ");
	}

	@Test
	void webSearchModeUsesWebsearchToTsquery() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate)
			.queryMode(TextQueryMode.WEB_SEARCH)
			.build()
			.buildSearchSql("");

		assertThat(sql).contains("websearch_to_tsquery('english'::regconfig, input.qt)").doesNotContain("quote_literal");
	}

	@Test
	void appendsFilterAndMinScoreInParameterOrder() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate)
			.minScore(0.1)
			.build()
			.buildSearchSql("\"metadata\"::jsonb @@ '$.lang == \"java\"'::jsonpath");

		assertThat(sql).contains(" AND \"metadata\"::jsonb @@ '$.lang == \"java\"'::jsonpath AND ts_rank(");
		// Bind order: query text (FROM), min score (WHERE), limit (LIMIT).
		assertThat(sql.indexOf("CAST(? AS text)")).isLessThan(sql.indexOf(">= ?"));
		assertThat(sql.indexOf(">= ?")).isLessThan(sql.indexOf("LIMIT ?"));
	}

	@Test
	void computesTsvectorInlineWhenNotStored() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate)
			.useStoredTsvector(false)
			.splitIdentifiers(false)
			.build()
			.buildSearchSql("");

		assertThat(sql).doesNotContain("content_tsv")
			.contains("to_tsvector('english'::regconfig, coalesce(\"content\", '')) @@ t.q");
	}

	@Test
	void rejectsIdentifiersThatCouldInjectSql() {
		var builder = PgFullTextDocumentRetriever.builder(this.jdbcTemplate);

		assertThatIllegalArgumentException().isThrownBy(() -> builder.tableName("vector_store; DROP TABLE x"));
		assertThatIllegalArgumentException().isThrownBy(() -> builder.schemaName("public\""));
		assertThatIllegalArgumentException().isThrownBy(() -> builder.tsvectorColumn("1col"));
		assertThatIllegalArgumentException().isThrownBy(() -> builder.textSearchConfig("english'); --"));
	}

	@Test
	void bm25RanksThroughTheIndexAndDropsNonMatchingRows() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate)
			.ranking(TextRanking.BM25)
			.splitIdentifiers(false)
			.build()
			.buildSearchSql("");

		assertThat(sql).contains("coalesce(\"content\", '') <@> to_bm25query(CAST(? AS text), 'public.vector_store_bm25_idx')")
			.contains("ORDER BY ragkit_bm25 LIMIT ?) AS ranked WHERE ragkit_bm25 < 0")
			.contains("-ragkit_bm25 AS ragkit_score")
			.doesNotContain("content_tsv");
	}

	@Test
	void bm25SplitsIdentifiersOnBothSides() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate)
			.ranking(TextRanking.BM25)
			.build()
			.buildSearchSql("");

		assertThat(sql).contains("regexp_replace(coalesce(\"content\", ''), '([a-z0-9])([A-Z])'")
			.contains("to_bm25query(CAST(? AS text), ");
		// Bound values: the query text once, then the limit.
		assertThat(sql.chars().filter(c -> c == '?')).hasSize(2);
		// The query side is split in Java, the same way SQL splits documents.
		assertThat(PgFullTextDocumentRetriever.splitIdentifiers("RecipeService.createRecipe user_id a/b-c"))
			.isEqualTo("RecipeService.createRecipe user_id a/b-c Recipe Service create Recipe user id a b c");
	}

	@Test
	void bm25AppliesFilterBeforeTheLimitAndMinScoreAfter() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate)
			.ranking(TextRanking.BM25)
			.minScore(2.5)
			.build()
			.buildSearchSql("\"metadata\"::jsonb @@ '$.lang == \"java\"'::jsonpath");

		assertThat(sql).contains("FROM \"public\".\"vector_store\" WHERE \"metadata\"::jsonb @@");
		// Bind order: query text, limit (inner query), min score (outer query).
		assertThat(sql.indexOf("CAST(? AS text)")).isLessThan(sql.indexOf("LIMIT ?"));
		assertThat(sql.indexOf("LIMIT ?")).isLessThan(sql.indexOf("-ragkit_bm25 >= ?"));
	}

	@Test
	void bm25SupportsAnyTermsOnly() {
		var builder = PgFullTextDocumentRetriever.builder(this.jdbcTemplate)
			.ranking(TextRanking.BM25)
			.queryMode(TextQueryMode.WEB_SEARCH);

		assertThatIllegalArgumentException().isThrownBy(builder::build).withMessageContaining("ANY_TERMS");
	}

	@Test
	void minScoreRangeDependsOnRanking() {
		assertThatIllegalArgumentException()
			.isThrownBy(() -> PgFullTextDocumentRetriever.builder(this.jdbcTemplate).minScore(2.5).build());
		PgFullTextDocumentRetriever.builder(this.jdbcTemplate).ranking(TextRanking.BM25).minScore(2.5).build();
	}

	@Test
	void acceptsSchemaQualifiedTextSearchConfig() {
		String sql = PgFullTextDocumentRetriever.builder(this.jdbcTemplate)
			.textSearchConfig("public.chinese_zh")
			.build()
			.buildSearchSql("");

		assertThat(sql).contains("'public.chinese_zh'::regconfig");
	}

}
