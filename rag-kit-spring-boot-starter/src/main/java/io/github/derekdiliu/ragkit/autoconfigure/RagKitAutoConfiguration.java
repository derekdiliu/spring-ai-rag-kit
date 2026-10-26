package io.github.derekdiliu.ragkit.autoconfigure;

import io.github.derekdiliu.ragkit.fusion.RankFusion;
import io.github.derekdiliu.ragkit.pgvector.PgFullTextDocumentRetriever;
import io.github.derekdiliu.ragkit.retrieval.HybridDocumentRetriever;

import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.autoconfigure.condition.ConditionalOnSingleCandidate;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;
import org.springframework.jdbc.core.JdbcTemplate;

/**
 * Registers a {@link PgFullTextDocumentRetriever} over Spring AI's pgvector table and,
 * when a {@link PgVectorStore} is present, a {@link HybridDocumentRetriever} that fuses
 * both with Reciprocal Rank Fusion. Inject the hybrid bean into
 * {@code RetrievalAugmentationAdvisor.builder().documentRetriever(...)}.
 */
@AutoConfiguration(afterName = {
		"org.springframework.ai.vectorstore.pgvector.autoconfigure.PgVectorStoreAutoConfiguration",
		"org.springframework.boot.jdbc.autoconfigure.JdbcTemplateAutoConfiguration" })
@ConditionalOnClass({ PgFullTextDocumentRetriever.class, JdbcTemplate.class })
@EnableConfigurationProperties(RagKitProperties.class)
public class RagKitAutoConfiguration {

	private static final String PGVECTOR_PREFIX = "spring.ai.vectorstore.pgvector";

	@Bean
	@ConditionalOnMissingBean
	@ConditionalOnSingleCandidate(JdbcTemplate.class)
	@ConditionalOnProperty(prefix = RagKitProperties.PREFIX + ".fulltext", name = "enabled", havingValue = "true",
			matchIfMissing = true)
	PgFullTextDocumentRetriever ragKitFullTextDocumentRetriever(JdbcTemplate jdbcTemplate,
			RagKitProperties properties, Environment environment, ObjectProvider<VectorStore> vectorStores) {
		RagKitProperties.FullText p = properties.getFulltext();
		// Unless set explicitly, follow the table PgVectorStore was configured with.
		String schemaName = (p.getSchemaName() != null) ? p.getSchemaName()
				: environment.getProperty(PGVECTOR_PREFIX + ".schema-name", PgVectorStore.DEFAULT_SCHEMA_NAME);
		String tableName = (p.getTableName() != null) ? p.getTableName()
				: environment.getProperty(PGVECTOR_PREFIX + ".table-name", PgVectorStore.DEFAULT_TABLE_NAME);
		PgFullTextDocumentRetriever retriever = PgFullTextDocumentRetriever.builder(jdbcTemplate)
			.schemaName(schemaName)
			.tableName(tableName)
			.tsvectorColumn(p.getTsvectorColumn())
			.textSearchConfig(p.getTextSearchConfig())
			.queryMode(p.getQueryMode())
			.splitIdentifiers(p.isSplitIdentifiers())
			.useStoredTsvector(p.isUseStoredTsvector())
			.topK(p.getTopK())
			.build();
		if (p.isInitializeSchema()) {
			// Resolving the stores first makes sure PgVectorStore has created its table.
			vectorStores.forEach(store -> {
			});
			retriever.initializeSchema();
		}
		return retriever;
	}

	@Bean
	@ConditionalOnMissingBean
	@ConditionalOnSingleCandidate(PgVectorStore.class)
	@ConditionalOnBean(PgFullTextDocumentRetriever.class)
	@ConditionalOnProperty(prefix = RagKitProperties.PREFIX + ".hybrid", name = "enabled", havingValue = "true",
			matchIfMissing = true)
	HybridDocumentRetriever ragKitHybridDocumentRetriever(PgVectorStore vectorStore,
			PgFullTextDocumentRetriever fullTextRetriever, RagKitProperties properties) {
		RagKitProperties.Hybrid p = properties.getHybrid();
		VectorStoreDocumentRetriever vectorRetriever = VectorStoreDocumentRetriever.builder()
			.vectorStore(vectorStore)
			.topK(p.getVectorTopK())
			.similarityThreshold(p.getSimilarityThreshold())
			.build();
		return HybridDocumentRetriever.builder()
			.retriever("vector", p.getVectorWeight(), vectorRetriever)
			.retriever("fulltext", p.getFulltextWeight(), fullTextRetriever)
			.fusion(RankFusion.rrf(p.getRrfK()))
			.topK(p.getTopK())
			.build();
	}

}
