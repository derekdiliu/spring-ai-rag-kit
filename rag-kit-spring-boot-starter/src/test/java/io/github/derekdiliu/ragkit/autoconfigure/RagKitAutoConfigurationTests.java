package io.github.derekdiliu.ragkit.autoconfigure;

import io.github.derekdiliu.ragkit.fusion.ReciprocalRankFusion;
import io.github.derekdiliu.ragkit.pgvector.PgFullTextDocumentRetriever;
import io.github.derekdiliu.ragkit.retrieval.HybridDocumentRetriever;
import org.junit.jupiter.api.Test;

import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.jdbc.core.JdbcTemplate;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

/**
 * Bean wiring only. No database is touched because schema initialisation is off by
 * default and the JdbcTemplate has no DataSource.
 */
class RagKitAutoConfigurationTests {

	private final ApplicationContextRunner runner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(RagKitAutoConfiguration.class));

	@Test
	void registersFullTextAndHybridRetrieversNextToAVectorStore() {
		this.runner.withBean(JdbcTemplate.class, JdbcTemplate::new)
			.withBean(VectorStore.class, () -> mock(VectorStore.class))
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
		this.runner.withBean(JdbcTemplate.class, JdbcTemplate::new)
			.withBean(VectorStore.class, () -> mock(VectorStore.class))
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
		this.runner.withBean(JdbcTemplate.class, JdbcTemplate::new).run(context -> {
			assertThat(context).hasSingleBean(PgFullTextDocumentRetriever.class);
			assertThat(context).doesNotHaveBean(HybridDocumentRetriever.class);
		});
	}

	@Test
	void canBeSwitchedOff() {
		this.runner.withBean(JdbcTemplate.class, JdbcTemplate::new)
			.withBean(VectorStore.class, () -> mock(VectorStore.class))
			.withPropertyValues("ragkit.fulltext.enabled=false")
			.run(context -> {
				assertThat(context).doesNotHaveBean(PgFullTextDocumentRetriever.class);
				assertThat(context).doesNotHaveBean(HybridDocumentRetriever.class);
			});

		this.runner.withBean(JdbcTemplate.class, JdbcTemplate::new)
			.withBean(VectorStore.class, () -> mock(VectorStore.class))
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
		this.runner.withBean(JdbcTemplate.class, JdbcTemplate::new)
			.withBean("custom", PgFullTextDocumentRetriever.class,
					() -> PgFullTextDocumentRetriever.builder(new JdbcTemplate()).tableName("my_docs").build())
			.run(context -> {
				assertThat(context).hasSingleBean(PgFullTextDocumentRetriever.class);
				assertThat(context).hasBean("custom");
			});
	}

}
