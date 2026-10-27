package io.github.derekdiliu.ragkit.pgvector;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.json.JsonMapper;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionConverter;
import org.springframework.ai.vectorstore.filter.FilterExpressionTextParser;
import org.springframework.ai.vectorstore.pgvector.PgVectorFilterExpressionConverter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

/**
 * A {@link DocumentRetriever} backed by PostgreSQL full-text search, reading the same
 * table that Spring AI's {@code PgVectorStore} writes. No second copy of the data and no
 * extra search engine: ingest once through the vector store, then retrieve by keyword,
 * identifier or error code alongside vector similarity (see
 * {@code HybridDocumentRetriever}).
 * <p>
 * Call {@link #initializeSchema()} once to add a stored, generated {@code tsvector}
 * column and its GIN index to the vector store table. Without it, set
 * {@link Builder#useStoredTsvector(boolean) useStoredTsvector(false)} to compute the
 * {@code tsvector} on the fly (fine for trying it out; a sequential scan on large tables).
 * <p>
 * Scores are {@code ts_rank(..., 1|32)}: term frequency divided by {@code 1 + log} of
 * the document length, mapped into {@code [0, 1)}. With
 * {@link Builder#ranking(TextRanking) ranking(BM25)} they are BM25 scores from the
 * {@code pg_textsearch} extension, and {@link #initializeSchema()} creates a {@code bm25}
 * index instead of the column. Neither is comparable with cosine similarity; fuse with
 * rank-based methods such as RRF.
 * <p>
 * Metadata filters use {@link PgVectorFilterExpressionConverter}, so they behave exactly
 * like {@code PgVectorStore} filters, and can be supplied per query under
 * {@link VectorStoreDocumentRetriever#FILTER_EXPRESSION} in the query context (as a
 * {@link Filter.Expression} or its text form).
 */
public final class PgFullTextDocumentRetriever implements DocumentRetriever {

	/** Metadata key holding the full-text score of a returned document. */
	public static final String FULLTEXT_SCORE = "ragkit_fulltext_score";

	private static final Log logger = LogFactory.getLog(PgFullTextDocumentRetriever.class);

	private static final Pattern IDENTIFIER = Pattern.compile("[A-Za-z_][A-Za-z0-9_]{0,62}");

	private static final Pattern TEXT_SEARCH_CONFIG = Pattern
		.compile("([A-Za-z_][A-Za-z0-9_]{0,62}\\.)?[A-Za-z_][A-Za-z0-9_]{0,62}");

	private static final Pattern CAMEL_CASE_BOUNDARY = Pattern.compile("([a-z0-9])([A-Z])");

	private static final Pattern IDENTIFIER_SEPARATORS = Pattern.compile("[._/:#-]+");

	private static final String SCORE_COLUMN = "ragkit_score";

	private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {
	};

	private final JdbcTemplate jdbcTemplate;

	private final String schemaName;

	private final String tableName;

	private final String idColumn;

	private final String contentColumn;

	private final String metadataColumn;

	private final String tsvectorColumn;

	private final String textSearchConfig;

	private final TextQueryMode queryMode;

	private final TextRanking ranking;

	private final boolean splitIdentifiers;

	private final boolean useStoredTsvector;

	private final int topK;

	private final double minScore;

	private final Filter.Expression defaultFilterExpression;

	private final FilterExpressionConverter filterConverter;

	private final JsonMapper jsonMapper;

	private final RowMapper<Document> rowMapper = this::mapRow;

	private PgFullTextDocumentRetriever(Builder b) {
		this.jdbcTemplate = b.jdbcTemplate;
		this.schemaName = b.schemaName;
		this.tableName = b.tableName;
		this.idColumn = b.idColumn;
		this.contentColumn = b.contentColumn;
		this.metadataColumn = b.metadataColumn;
		this.tsvectorColumn = b.tsvectorColumn;
		this.textSearchConfig = b.textSearchConfig;
		this.queryMode = b.queryMode;
		this.ranking = b.ranking;
		this.splitIdentifiers = b.splitIdentifiers;
		this.useStoredTsvector = b.useStoredTsvector;
		this.topK = b.topK;
		this.minScore = b.minScore;
		this.defaultFilterExpression = b.filterExpression;
		this.filterConverter = new PgVectorFilterExpressionConverter(this.metadataColumn);
		this.jsonMapper = b.jsonMapper != null ? b.jsonMapper : JsonMapper.builder().build();
	}

	public static Builder builder(JdbcTemplate jdbcTemplate) {
		return new Builder(jdbcTemplate);
	}

	@Override
	public List<Document> retrieve(Query query) {
		Assert.notNull(query, "query must not be null");
		if (!StringUtils.hasText(query.text())) {
			return List.of();
		}

		Filter.Expression filter = resolveFilter(query);
		String filterSql = filter != null ? this.filterConverter.convertExpression(filter) : "";

		String sql = buildSearchSql(filterSql);
		List<Object> args = new ArrayList<>(3);
		if (this.ranking == TextRanking.BM25) {
			args.add(this.splitIdentifiers ? splitIdentifiers(query.text()) : query.text());
			args.add(this.topK);
		}
		else {
			args.add(query.text());
		}
		if (this.minScore > 0) {
			args.add(this.minScore);
		}
		if (this.ranking == TextRanking.TS_RANK) {
			args.add(this.topK);
		}

		if (logger.isTraceEnabled()) {
			logger.trace("Full-text search SQL: " + sql);
		}
		return this.jdbcTemplate.query(sql, this.rowMapper, args.toArray());
	}

	/**
	 * Adds a generated, stored {@code tsvector} column and a GIN index to the table, or
	 * for {@link TextRanking#BM25} the {@code pg_textsearch} extension and a {@code bm25}
	 * index. Idempotent. Requires the table to exist already (create the
	 * {@code PgVectorStore} first).
	 * <p>
	 * Adding a stored generated column rewrites the table; on a large table run this in a
	 * maintenance window. The BM25 index needs no new column.
	 */
	public void initializeSchema() {
		String table = qualifiedTable();
		if (this.ranking == TextRanking.BM25) {
			this.jdbcTemplate.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch");
			this.jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS " + quote(bm25IndexName()) + " ON " + table
					+ " USING bm25 ((" + documentTextExpression() + ")) WITH (text_config = '" + this.textSearchConfig
					+ "')");
			logger.info("BM25 index " + bm25IndexName() + " ready on " + table);
			return;
		}
		this.jdbcTemplate.execute("ALTER TABLE " + table + " ADD COLUMN IF NOT EXISTS " + quote(this.tsvectorColumn)
				+ " tsvector GENERATED ALWAYS AS (" + documentTsvectorExpression() + ") STORED");
		this.jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS " + quote(indexName()) + " ON " + table
				+ " USING gin (" + quote(this.tsvectorColumn) + ")");
		logger.info("Full-text column " + this.tsvectorColumn + " and GIN index ready on " + table);
	}

	/** The search SQL, exposed for tests and for logging. */
	String buildSearchSql(String filterSql) {
		if (this.ranking == TextRanking.BM25) {
			return buildBm25SearchSql(filterSql);
		}
		String tsvector = this.useStoredTsvector ? quote(this.tsvectorColumn) : documentTsvectorExpression();
		// ts_rank with log-length normalisation. On BEIR (nDCG@10) it scores 0.60 on SciFact
		// and 0.19 on FiQA, where ts_rank_cd scores 0.33 and 0.03: cover density degrades
		// badly on the OR queries ANY_TERMS builds, and long documents win without length
		// normalisation.
		String rank = "ts_rank(" + tsvector + ", t.q, 1|32)";

		StringBuilder sql = new StringBuilder(512);
		sql.append("SELECT ")
			.append(quote(this.idColumn))
			.append(" AS ragkit_id, ")
			.append(quote(this.contentColumn))
			.append(" AS ragkit_content, ")
			.append(quote(this.metadataColumn))
			.append(" AS ragkit_metadata, ")
			.append(rank)
			.append(" AS ")
			.append(SCORE_COLUMN);
		// Build the tsquery once in a one-row subquery. OFFSET 0 stops the planner from
		// inlining it, which would rebuild the tsquery per row and rule out the GIN index
		// (measured: sequential scan at 45 ms vs index scan at 0.5 ms on 20k rows).
		sql.append(" FROM (SELECT ")
			.append(queryTsqueryExpression())
			.append(" AS q FROM (SELECT CAST(? AS text) AS qt) AS input OFFSET 0) AS t, ")
			.append(qualifiedTable());
		sql.append(" WHERE ").append(tsvector).append(" @@ t.q");
		if (StringUtils.hasText(filterSql)) {
			sql.append(" AND ").append(filterSql);
		}
		if (this.minScore > 0) {
			sql.append(" AND ").append(rank).append(" >= ?");
		}
		sql.append(" ORDER BY ").append(SCORE_COLUMN).append(" DESC LIMIT ?");
		return sql.toString();
	}

	/**
	 * BM25 via pg_textsearch. Its {@code <@>} operator returns the negated score, so the
	 * index scan yields the best matches first in ascending order. The outer query drops
	 * rows that match no query term (score 0), which a non-index plan would return.
	 */
	private String buildBm25SearchSql(String filterSql) {
		// The query text is bound as one value, identifier split in Java: an expression
		// over another relation's column would keep the planner from ordering by the index.
		String bm25 = documentTextExpression() + " <@> to_bm25query(CAST(? AS text), '" + this.schemaName + "."
				+ bm25IndexName() + "')";
		StringBuilder sql = new StringBuilder(512);
		sql.append("SELECT ragkit_id, ragkit_content, ragkit_metadata, -ragkit_bm25 AS ")
			.append(SCORE_COLUMN)
			.append(" FROM (SELECT ")
			.append(quote(this.idColumn))
			.append(" AS ragkit_id, ")
			.append(quote(this.contentColumn))
			.append(" AS ragkit_content, ")
			.append(quote(this.metadataColumn))
			.append(" AS ragkit_metadata, ")
			.append(bm25)
			.append(" AS ragkit_bm25 FROM ")
			.append(qualifiedTable());
		if (StringUtils.hasText(filterSql)) {
			sql.append(" WHERE ").append(filterSql);
		}
		sql.append(" ORDER BY ragkit_bm25 LIMIT ?) AS ranked WHERE ragkit_bm25 < 0");
		if (this.minScore > 0) {
			sql.append(" AND -ragkit_bm25 >= ?");
		}
		sql.append(" ORDER BY ragkit_bm25");
		return sql.toString();
	}

	private String documentTsvectorExpression() {
		return "to_tsvector(" + regconfig() + ", " + documentTextExpression() + ")";
	}

	/** The document text as indexed, identifier split when enabled. */
	private String documentTextExpression() {
		return maybeSplit("coalesce(" + quote(this.contentColumn) + ", '')");
	}

	private String queryTsqueryExpression() {
		return switch (this.queryMode) {
			case WEB_SEARCH -> "websearch_to_tsquery(" + regconfig() + ", input.qt)";
			// Lexemes of the normalised query, quoted and OR-ed. Casting (not to_tsquery)
			// avoids stemming the already-stemmed lexemes a second time.
			case ANY_TERMS -> "CAST(array_to_string(ARRAY(SELECT quote_literal(lexeme) FROM unnest(tsvector_to_array("
					+ "to_tsvector(" + regconfig() + ", " + maybeSplit("input.qt") + "))) AS lexeme), ' | ') AS tsquery)";
		};
	}

	/**
	 * Appends a copy of the text with camelCase boundaries and {@code . _ / : # -}
	 * separators turned into spaces, so {@code RecipeService.createRecipe} is indexed as
	 * the whole token and as {@code recipe service create recipe}. Applied identically to
	 * documents and queries.
	 */
	private String maybeSplit(String textExpression) {
		if (!this.splitIdentifiers) {
			return textExpression;
		}
		return "(" + textExpression + " || ' ' || regexp_replace(regexp_replace(" + textExpression
				+ ", '([a-z0-9])([A-Z])', '\\1 \\2', 'g'), '[._/:#-]+', ' ', 'g'))";
	}

	/** {@link #maybeSplit} for a query bound as a single value. */
	static String splitIdentifiers(String text) {
		String words = CAMEL_CASE_BOUNDARY.matcher(text).replaceAll("$1 $2");
		return text + " " + IDENTIFIER_SEPARATORS.matcher(words).replaceAll(" ");
	}

	private String regconfig() {
		return "'" + this.textSearchConfig + "'::regconfig";
	}

	private String qualifiedTable() {
		return quote(this.schemaName) + "." + quote(this.tableName);
	}

	private String indexName() {
		return truncate(this.tableName + "_" + this.tsvectorColumn + "_gin_idx");
	}

	private String bm25IndexName() {
		return truncate(this.tableName + "_bm25_idx");
	}

	private static String truncate(String identifier) {
		return identifier.length() > 63 ? identifier.substring(0, 63) : identifier;
	}

	private static String quote(String identifier) {
		return "\"" + identifier + "\"";
	}

	private Filter.Expression resolveFilter(Query query) {
		Object fromContext = query.context().get(VectorStoreDocumentRetriever.FILTER_EXPRESSION);
		if (fromContext instanceof Filter.Expression expression) {
			return expression;
		}
		if (fromContext != null && StringUtils.hasText(fromContext.toString())) {
			return new FilterExpressionTextParser().parse(fromContext.toString());
		}
		return this.defaultFilterExpression;
	}

	private Document mapRow(ResultSet rs, int rowNum) throws SQLException {
		String metadataJson = rs.getString("ragkit_metadata");
		Map<String, Object> metadata = StringUtils.hasText(metadataJson)
				? new HashMap<>(this.jsonMapper.readValue(metadataJson, MAP_TYPE)) : new HashMap<>();
		// Document rejects null metadata values; JSON metadata may contain them.
		metadata.values().removeIf(Objects::isNull);
		double score = rs.getDouble(SCORE_COLUMN);
		metadata.put(FULLTEXT_SCORE, score);
		String content = rs.getString("ragkit_content");
		return Document.builder()
			.id(rs.getString("ragkit_id"))
			.text(content != null ? content : "")
			.metadata(metadata)
			.score(score)
			.build();
	}

	public static final class Builder {

		private final JdbcTemplate jdbcTemplate;

		private String schemaName = "public";

		private String tableName = "vector_store";

		private String idColumn = "id";

		private String contentColumn = "content";

		private String metadataColumn = "metadata";

		private String tsvectorColumn = "content_tsv";

		private String textSearchConfig = "english";

		private TextQueryMode queryMode = TextQueryMode.ANY_TERMS;

		private TextRanking ranking = TextRanking.TS_RANK;

		private boolean splitIdentifiers = true;

		private boolean useStoredTsvector = true;

		private int topK = 4;

		private double minScore = 0.0;

		private Filter.Expression filterExpression;

		private JsonMapper jsonMapper;

		private Builder(JdbcTemplate jdbcTemplate) {
			Assert.notNull(jdbcTemplate, "jdbcTemplate must not be null");
			this.jdbcTemplate = jdbcTemplate;
		}

		/** Defaults to {@code public}, matching {@code PgVectorStore}. */
		public Builder schemaName(String schemaName) {
			this.schemaName = identifier(schemaName, "schemaName");
			return this;
		}

		/** Defaults to {@code vector_store}, matching {@code PgVectorStore}. */
		public Builder tableName(String tableName) {
			this.tableName = identifier(tableName, "tableName");
			return this;
		}

		public Builder idColumn(String idColumn) {
			this.idColumn = identifier(idColumn, "idColumn");
			return this;
		}

		public Builder contentColumn(String contentColumn) {
			this.contentColumn = identifier(contentColumn, "contentColumn");
			return this;
		}

		public Builder metadataColumn(String metadataColumn) {
			this.metadataColumn = identifier(metadataColumn, "metadataColumn");
			return this;
		}

		/** Name of the generated {@code tsvector} column. Defaults to {@code content_tsv}. */
		public Builder tsvectorColumn(String tsvectorColumn) {
			this.tsvectorColumn = identifier(tsvectorColumn, "tsvectorColumn");
			return this;
		}

		/**
		 * PostgreSQL text search configuration, e.g. {@code english}, {@code simple} (no
		 * stemming or stop words; a reasonable choice for mixed-language or code-heavy
		 * content) or a custom one such as {@code public.chinese_zh}. Changing it after
		 * {@link PgFullTextDocumentRetriever#initializeSchema()} requires dropping the
		 * generated column first.
		 */
		public Builder textSearchConfig(String textSearchConfig) {
			Assert.hasText(textSearchConfig, "textSearchConfig must not be empty");
			Assert.isTrue(TEXT_SEARCH_CONFIG.matcher(textSearchConfig).matches(),
					() -> "invalid text search configuration name: " + textSearchConfig);
			this.textSearchConfig = textSearchConfig;
			return this;
		}

		public Builder queryMode(TextQueryMode queryMode) {
			Assert.notNull(queryMode, "queryMode must not be null");
			this.queryMode = queryMode;
			return this;
		}

		/**
		 * How matches are scored. Defaults to {@link TextRanking#TS_RANK}, which works on
		 * any PostgreSQL; {@link TextRanking#BM25} needs the {@code pg_textsearch}
		 * extension.
		 */
		public Builder ranking(TextRanking ranking) {
			Assert.notNull(ranking, "ranking must not be null");
			this.ranking = ranking;
			return this;
		}

		/**
		 * Whether to split camelCase and dotted/underscored identifiers into words, on
		 * both the document and query side. Defaults to {@code true}. Must match the
		 * setting used when the column was created.
		 */
		public Builder splitIdentifiers(boolean splitIdentifiers) {
			this.splitIdentifiers = splitIdentifiers;
			return this;
		}

		/**
		 * Whether to read the stored generated column (fast, needs
		 * {@link PgFullTextDocumentRetriever#initializeSchema()}) or compute the
		 * {@code tsvector} per query (no schema change, sequential scan). Defaults to
		 * {@code true}.
		 */
		public Builder useStoredTsvector(boolean useStoredTsvector) {
			this.useStoredTsvector = useStoredTsvector;
			return this;
		}

		/** Maximum documents returned. Defaults to 4, like {@code SearchRequest}. */
		public Builder topK(int topK) {
			Assert.isTrue(topK > 0, "topK must be positive");
			this.topK = topK;
			return this;
		}

		/**
		 * Drop matches scoring below this. Defaults to 0. {@code ts_rank} scores lie in
		 * {@code [0, 1)}; BM25 scores are unbounded.
		 */
		public Builder minScore(double minScore) {
			Assert.isTrue(minScore >= 0 && Double.isFinite(minScore), "minScore must be a non-negative number");
			this.minScore = minScore;
			return this;
		}

		/** Filter applied when the query context does not supply one. */
		public Builder filterExpression(Filter.Expression filterExpression) {
			this.filterExpression = filterExpression;
			return this;
		}

		public Builder jsonMapper(JsonMapper jsonMapper) {
			this.jsonMapper = jsonMapper;
			return this;
		}

		public PgFullTextDocumentRetriever build() {
			if (this.ranking == TextRanking.TS_RANK) {
				Assert.isTrue(this.minScore < 1, "minScore must be below 1 for ts_rank, whose scores lie in [0, 1)");
			}
			else {
				Assert.isTrue(this.queryMode == TextQueryMode.ANY_TERMS, "BM25 ranking supports the ANY_TERMS query mode only");
			}
			return new PgFullTextDocumentRetriever(this);
		}

		private static String identifier(String value, String name) {
			Assert.hasText(value, () -> name + " must not be empty");
			Assert.isTrue(IDENTIFIER.matcher(value).matches(),
					() -> name + " must be a plain SQL identifier (letters, digits, underscore): " + value);
			return value;
		}

	}

}
