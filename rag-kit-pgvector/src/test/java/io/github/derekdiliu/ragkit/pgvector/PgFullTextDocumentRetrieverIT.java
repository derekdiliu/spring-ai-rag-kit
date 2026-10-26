package io.github.derekdiliu.ragkit.pgvector;

import java.util.List;
import java.util.Map;

import io.github.derekdiliu.ragkit.fusion.FusionMetadata;
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
import org.springframework.jdbc.core.JdbcTemplate;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Runs against the real {@code pgvector/pgvector} image, writing through Spring AI's own
 * {@link PgVectorStore} so the full-text retriever is proven to work on the exact table
 * layout the vector store creates. Skipped when Docker is unavailable.
 */
@Testcontainers(disabledWithoutDocker = true)
class PgFullTextDocumentRetrieverIT {

	@Container
	static final PostgreSQLContainer postgres = new PostgreSQLContainer("pgvector/pgvector:pg17");

	static JdbcTemplate jdbcTemplate;

	static PgVectorStore vectorStore;

	static PgFullTextDocumentRetriever fullText;

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

		fullText = PgFullTextDocumentRetriever.builder(jdbcTemplate).topK(10).build();
		fullText.initializeSchema();
		fullText.initializeSchema(); // idempotent
	}

	private static Document doc(String text, String lang, String path) {
		return Document.builder().text(text).metadata(Map.of("lang", lang, "path", path)).build();
	}

	private static List<String> paths(List<Document> documents) {
		return documents.stream().map(d -> (String) d.getMetadata().get("path")).toList();
	}

	@Test
	void findsMethodNameInsideDottedIdentifier() {
		List<Document> result = fullText.retrieve(new Query("where is createRecipe implemented?"));

		assertThat(paths(result)).first().isEqualTo("RecipeService.java");
		assertThat(result.get(0).getScore()).isBetween(0.0, 1.0);
		assertThat(result.get(0).getMetadata()).containsKey(PgFullTextDocumentRetriever.FULLTEXT_SCORE);
	}

	@Test
	void findsExactErrorCode() {
		assertThat(paths(fullText.retrieve(new Query("what does E2041 mean")))).containsExactly("troubleshooting.md");
	}

	@Test
	void naturalLanguageQueryMatchesOnAnyTerm() {
		// No document contains every word of the question.
		List<String> result = paths(fullText.retrieve(new Query("how do I scale a recipe for more servings")));

		assertThat(result).first().isEqualTo("recipes.md");
		assertThat(result).contains("RecipeService.java");
	}

	@Test
	void stopWordsOnlyQueryReturnsNothing() {
		assertThat(fullText.retrieve(new Query("what is the"))).isEmpty();
	}

	@Test
	void hostileInputIsJustText() {
		List<Document> result = fullText.retrieve(new Query("it's a 'test' & | ! : (recipe) \\ ; DROP TABLE vector_store"));

		assertThat(paths(result)).contains("recipes.md");
		assertThat(jdbcTemplate.queryForObject("SELECT count(*) FROM vector_store", Integer.class)).isEqualTo(5);
	}

	@Test
	void appliesFilterFromQueryContextLikeVectorStoreDocumentRetriever() {
		var filter = new FilterExpressionBuilder().eq("lang", "docs").build();
		Query query = Query.builder()
			.text("recipe")
			.context(Map.<String, Object>of(VectorStoreDocumentRetriever.FILTER_EXPRESSION, filter))
			.build();

		assertThat(paths(fullText.retrieve(query))).containsExactly("recipes.md");
	}

	@Test
	void acceptsTextFilterFromQueryContext() {
		Query query = Query.builder()
			.text("recipe")
			.context(Map.<String, Object>of(VectorStoreDocumentRetriever.FILTER_EXPRESSION, "lang == 'java'"))
			.build();

		assertThat(paths(fullText.retrieve(query))).containsExactly("RecipeService.java");
	}

	@Test
	void webSearchModeSupportsPhrasesAndExclusion() {
		var webSearch = PgFullTextDocumentRetriever.builder(jdbcTemplate).queryMode(TextQueryMode.WEB_SEARCH).build();

		assertThat(paths(webSearch.retrieve(new Query("\"meal plan\"")))).containsExactly("features.md");
		assertThat(webSearch.retrieve(new Query("login -password"))).isEmpty();
	}

	@Test
	void worksWithoutStoredColumn() {
		var inline = PgFullTextDocumentRetriever.builder(jdbcTemplate).useStoredTsvector(false).build();

		assertThat(paths(inline.retrieve(new Query("createRecipe")))).first().isEqualTo("RecipeService.java");
	}

	@Test
	void hybridRetrieverFusesVectorAndFullText() {
		var hybrid = HybridDocumentRetriever.builder()
			.retriever("vector", VectorStoreDocumentRetriever.builder().vectorStore(vectorStore).topK(5).build())
			.retriever("fulltext", fullText)
			.topK(3)
			.build();

		List<Document> result = hybrid.retrieve(new Query("E2041"));

		assertThat(result).hasSizeLessThanOrEqualTo(3);
		Document top = result.get(0);
		assertThat(top.getMetadata()).containsEntry("path", "troubleshooting.md");
		@SuppressWarnings("unchecked")
		Map<String, Integer> ranks = (Map<String, Integer>) top.getMetadata().get(FusionMetadata.SOURCE_RANKS);
		assertThat(ranks).containsEntry("fulltext", 1);
	}

}
