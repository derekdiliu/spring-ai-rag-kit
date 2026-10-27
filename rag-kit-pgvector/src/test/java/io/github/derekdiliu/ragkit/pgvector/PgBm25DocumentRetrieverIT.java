package io.github.derekdiliu.ragkit.pgvector;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.github.derekdiliu.ragkit.retrieval.HybridDocumentRetriever;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.postgresql.ds.PGSimpleDataSource;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.postgresql.PostgreSQLContainer;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.jdbc.core.ConnectionCallback;
import org.springframework.jdbc.core.JdbcTemplate;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * BM25 ranking through pg_textsearch, on the table Spring AI's {@link PgVectorStore}
 * creates. Skipped when Docker is unavailable.
 */
@Testcontainers(disabledWithoutDocker = true)
class PgBm25DocumentRetrieverIT {

	@Container
	static final PostgreSQLContainer postgres = PgTextSearchContainers.create();

	static JdbcTemplate jdbcTemplate;

	static PgVectorStore vectorStore;

	static PgFullTextDocumentRetriever bm25;

	@BeforeAll
	static void ingest() {
		var dataSource = new PGSimpleDataSource();
		dataSource.setUrl(postgres.getJdbcUrl());
		dataSource.setUser(postgres.getUsername());
		dataSource.setPassword(postgres.getPassword());
		jdbcTemplate = new JdbcTemplate(dataSource);

		vectorStore = PgVectorStore.builder(jdbcTemplate, new HashingEmbeddingModel())
			.dimensions(HashingEmbeddingModel.DIMENSIONS)
			.initializeSchema(true)
			.removeExistingVectorStoreTable(true)
			.build();
		vectorStore.afterPropertiesSet();

		vectorStore.add(List.of(
				doc("RecipeService.createRecipe validates the slug and persists the recipe", "java", "RecipeService.java"),
				doc("The user login flow issues a JWT after password verification", "python", "auth.py"),
				doc("Error E2041: connection refused when the database is unreachable", "docs", "troubleshooting.md"),
				doc("Shopping lists can be generated from a meal plan", "docs", "features.md"),
				doc("A recipe can be scaled to any number of servings", "docs", "recipes.md")));

		bm25 = PgFullTextDocumentRetriever.builder(jdbcTemplate).ranking(TextRanking.BM25).topK(10).build();
		bm25.initializeSchema();
		bm25.initializeSchema(); // idempotent
	}

	private static Document doc(String text, String lang, String path) {
		return Document.builder().text(text).metadata(Map.of("lang", lang, "path", path)).build();
	}

	private static List<String> paths(List<Document> documents) {
		return documents.stream().map(d -> (String) d.getMetadata().get("path")).toList();
	}

	@Test
	void findsMethodNameInsideDottedIdentifier() {
		List<Document> result = bm25.retrieve(new Query("where is createRecipe implemented?"));

		assertThat(paths(result)).first().isEqualTo("RecipeService.java");
		assertThat(result.get(0).getScore()).isPositive();
		assertThat(result.get(0).getMetadata()).containsKey(PgFullTextDocumentRetriever.FULLTEXT_SCORE);
	}

	@Test
	void returnsOnlyDocumentsMatchingAQueryTerm() {
		assertThat(paths(bm25.retrieve(new Query("what does E2041 mean")))).containsExactly("troubleshooting.md");
	}

	@Test
	void naturalLanguageQueryMatchesOnAnyTerm() {
		List<String> result = paths(bm25.retrieve(new Query("how do I scale a recipe for more servings")));

		assertThat(result).first().isEqualTo("recipes.md");
		assertThat(result).contains("RecipeService.java");
	}

	@Test
	void stopWordsOnlyQueryReturnsNothing() {
		assertThat(bm25.retrieve(new Query("what is the"))).isEmpty();
	}

	@Test
	void hostileInputIsJustText() {
		List<Document> result = bm25.retrieve(new Query("it's a 'test' & | ! : (recipe) \\ ; DROP TABLE vector_store"));

		assertThat(paths(result)).contains("recipes.md");
		assertThat(jdbcTemplate.queryForObject("SELECT count(*) FROM vector_store", Integer.class)).isEqualTo(5);
	}

	@Test
	void appliesFilterFromQueryContext() {
		var filter = new FilterExpressionBuilder().eq("lang", "docs").build();
		Query query = Query.builder()
			.text("recipe")
			.context(Map.<String, Object>of(VectorStoreDocumentRetriever.FILTER_EXPRESSION, filter))
			.build();

		assertThat(paths(bm25.retrieve(query))).containsExactly("recipes.md");
	}

	@Test
	void addsNoColumnToTheTable() {
		assertThat(jdbcTemplate.queryForList(
				"SELECT column_name FROM information_schema.columns WHERE table_name = 'vector_store'", String.class))
			.containsExactlyInAnyOrder("id", "content", "metadata", "embedding");
	}

	@Test
	void searchUsesTheBm25Index() {
		String sql = bm25.buildSearchSql("");
		List<String> plan = jdbcTemplate.execute((ConnectionCallback<List<String>>) connection -> {
			// Five rows are cheaper to sort than to scan by index; make the planner show
			// whether the index is usable at all.
			connection.createStatement().execute("SET enable_seqscan = off");
			try (PreparedStatement explain = connection.prepareStatement("EXPLAIN " + sql)) {
				explain.setString(1, "createRecipe");
				explain.setInt(2, 10);
				List<String> lines = new ArrayList<>();
				try (ResultSet rs = explain.executeQuery()) {
					while (rs.next()) {
						lines.add(rs.getString(1));
					}
				}
				return lines;
			}
		});

		assertThat(String.join("\n", plan)).contains("Index Scan using vector_store_bm25_idx");
	}

	@Test
	void hybridRetrieverFusesVectorAndBm25() {
		var hybrid = HybridDocumentRetriever.builder()
			.retriever("vector", VectorStoreDocumentRetriever.builder().vectorStore(vectorStore).topK(5).build())
			.retriever("fulltext", bm25)
			.topK(3)
			.build();

		assertThat(hybrid.retrieve(new Query("E2041")).get(0).getMetadata()).containsEntry("path", "troubleshooting.md");
	}

}
