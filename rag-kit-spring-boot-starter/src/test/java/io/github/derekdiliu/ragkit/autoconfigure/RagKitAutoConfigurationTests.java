package io.github.derekdiliu.ragkit.autoconfigure;

import javax.sql.DataSource;

import io.github.derekdiliu.ragkit.fusion.ReciprocalRankFusion;
import io.github.derekdiliu.ragkit.pgvector.PgFullTextDocumentRetriever;
import io.github.derekdiliu.ragkit.retrieval.HybridDocumentRetriever;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.ArgumentMatchers;

import org.springframework.ai.rag.Query;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.assertj.AssertableApplicationContext;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.startsWith;
import static org.mockito.BDDMockito.then;
import static org.mockito.Mockito.mock;

/**
 * Bean wiring only. No database is touched because schema initialisation is off by
 * default and the JdbcTemplate's DataSource is a mock.
 */
class RagKitAutoConfigurationTests {

	private final ApplicationContextRunner runner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(RagKitAutoConfiguration.class));

	@Test
	void registersFullTextAndHybridRetrieversNextToAVectorStore() {
		this.runner.withBean(JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean(PgVectorStore.class, () -> mock(PgVectorStore.class))
			.run(context -> {
				assertThat(context).hasSingleBean(PgFullTextDocumentRetriever.class);
				assertThat(context).hasSingleBean(HybridDocumentRetriever.class);

				HybridDocumentRetriever hybrid = context.getBean(HybridDocumentRetriever.class);
				assertThat(hybrid.getRetrievers()).extracting(HybridDocumentRetriever.NamedRetriever::name)
					.containsExactly("vector", "fulltext");
				assertThat(hybrid.getTopK()).isEqualTo(5);
				assertThat(hybrid.getFusion()).isInstanceOfSatisfying(ReciprocalRankFusion.class,
						rrf -> assertThat(rrf.getK()).isEqualTo(60));
			});
	}

	@Test
	void bindsHybridProperties() {
		this.runner.withBean(JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean(PgVectorStore.class, () -> mock(PgVectorStore.class))
			.withPropertyValues("ragkit.hybrid.top-k=8", "ragkit.hybrid.rrf-k=10", "ragkit.hybrid.fulltext-weight=2.5")
			.run(context -> {
				HybridDocumentRetriever hybrid = context.getBean(HybridDocumentRetriever.class);
				assertThat(hybrid.getTopK()).isEqualTo(8);
				assertThat(((ReciprocalRankFusion) hybrid.getFusion()).getK()).isEqualTo(10);
				assertThat(hybrid.getRetrievers().get(1).weight()).isEqualTo(2.5);
			});
	}

	@Test
	void fullTextOnlyWithoutAVectorStore() {
		this.runner.withBean(JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class))).run(context -> {
			assertThat(context).hasSingleBean(PgFullTextDocumentRetriever.class);
			assertThat(context).doesNotHaveBean(HybridDocumentRetriever.class);
		});
	}

	@Test
	void canBeSwitchedOff() {
		this.runner.withBean(JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean(PgVectorStore.class, () -> mock(PgVectorStore.class))
			.withPropertyValues("ragkit.fulltext.enabled=false")
			.run(context -> {
				assertThat(context).doesNotHaveBean(PgFullTextDocumentRetriever.class);
				assertThat(context).doesNotHaveBean(HybridDocumentRetriever.class);
			});

		this.runner.withBean(JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean(PgVectorStore.class, () -> mock(PgVectorStore.class))
			.withPropertyValues("ragkit.hybrid.enabled=false")
			.run(context -> {
				assertThat(context).hasSingleBean(PgFullTextDocumentRetriever.class);
				assertThat(context).doesNotHaveBean(HybridDocumentRetriever.class);
			});
	}

	@Test
	void backsOffWithoutJdbcTemplate() {
		this.runner.run(context -> assertThat(context).doesNotHaveBean(PgFullTextDocumentRetriever.class));
	}

	@Test
	void backsOffWhenUserDefinesTheirOwnRetriever() {
		this.runner.withBean(JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean("custom", PgFullTextDocumentRetriever.class,
					() -> PgFullTextDocumentRetriever.builder(new JdbcTemplate()).tableName("my_docs").build())
			.run(context -> {
				assertThat(context).hasSingleBean(PgFullTextDocumentRetriever.class);
				assertThat(context).hasBean("custom");
			});
	}

	@Test
	void backsOffWithSeveralJdbcTemplatesAndNoPrimary() {
		this.runner.withBean("first", JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean("second", JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean(PgVectorStore.class, () -> mock(PgVectorStore.class))
			.run(context -> {
				assertThat(context).hasNotFailed();
				assertThat(context).doesNotHaveBean(PgFullTextDocumentRetriever.class);
				assertThat(context).doesNotHaveBean(HybridDocumentRetriever.class);
			});
	}

	@Test
	void usesThePrimaryJdbcTemplate() {
		JdbcTemplate primary = mock(JdbcTemplate.class);
		this.runner.withBean("primary", JdbcTemplate.class, () -> primary, bd -> bd.setPrimary(true))
			.withBean("other", JdbcTemplate.class, () -> mock(JdbcTemplate.class))
			.run(context -> {
				assertThat(searchSql(context, primary)).isNotEmpty();
			});
	}

	@Test
	void initializesSchemaWithSeveralVectorStores() {
		JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
		this.runner.withBean(JdbcTemplate.class, () -> jdbcTemplate)
			.withBean(PgVectorStore.class, () -> mock(PgVectorStore.class))
			.withBean("other", VectorStore.class, () -> mock(VectorStore.class))
			.withPropertyValues("ragkit.fulltext.initialize-schema=true")
			.run(context -> {
				assertThat(context).hasNotFailed();
				then(jdbcTemplate).should().execute(startsWith("ALTER TABLE \"public\".\"vector_store\""));
			});
	}

	@Test
	void followsTheTableConfiguredForPgVectorStore() {
		JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
		this.runner.withBean(JdbcTemplate.class, () -> jdbcTemplate)
			.withPropertyValues("spring.ai.vectorstore.pgvector.schema-name=rag",
					"spring.ai.vectorstore.pgvector.table-name=docs")
			.run(context -> assertThat(searchSql(context, jdbcTemplate)).contains("\"rag\".\"docs\""));
	}

	@Test
	void explicitTableOverridesPgVectorStoreProperties() {
		JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
		this.runner.withBean(JdbcTemplate.class, () -> jdbcTemplate)
			.withPropertyValues("spring.ai.vectorstore.pgvector.table-name=docs", "ragkit.fulltext.schema-name=kb",
					"ragkit.fulltext.table-name=chunks")
			.run(context -> assertThat(searchSql(context, jdbcTemplate)).contains("\"kb\".\"chunks\"")
				.doesNotContain("docs"));
	}

	@Test
	void bm25RankingCanBeSelected() {
		JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
		this.runner.withBean(JdbcTemplate.class, () -> jdbcTemplate)
			.withPropertyValues("ragkit.fulltext.ranking=bm25")
			.run(context -> assertThat(searchSql(context, jdbcTemplate)).contains("to_bm25query("));
	}

	@Test
	void noHybridForAVectorStoreThatIsNotPgVectorStore() {
		this.runner.withBean(JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean(VectorStore.class, () -> mock(VectorStore.class))
			.run(context -> {
				assertThat(context).hasSingleBean(PgFullTextDocumentRetriever.class);
				assertThat(context).doesNotHaveBean(HybridDocumentRetriever.class);
			});
	}

	@Test
	void hybridUsesPgVectorStoreNextToAnotherVectorStore() {
		this.runner.withBean(JdbcTemplate.class, () -> new JdbcTemplate(mock(DataSource.class)))
			.withBean(PgVectorStore.class, () -> mock(PgVectorStore.class))
			.withBean("other", VectorStore.class, () -> mock(VectorStore.class))
			.run(context -> assertThat(context).hasSingleBean(HybridDocumentRetriever.class));
	}

	/** Runs a full-text search against a mock JdbcTemplate and returns the SQL it sent. */
	private static String searchSql(AssertableApplicationContext context, JdbcTemplate jdbcTemplate) {
		context.getBean(PgFullTextDocumentRetriever.class).retrieve(new Query("q"));
		ArgumentCaptor<String> sql = ArgumentCaptor.forClass(String.class);
		then(jdbcTemplate).should().query(sql.capture(), ArgumentMatchers.<RowMapper<Object>>any(), any(Object[].class));
		return sql.getValue();
	}

}
