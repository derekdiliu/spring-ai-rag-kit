package io.github.derekdiliu.ragkit.autoconfigure;

import io.github.derekdiliu.ragkit.pgvector.TextQueryMode;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Configuration for rag-kit, under the {@code ragkit} prefix.
 */
@ConfigurationProperties(prefix = RagKitProperties.PREFIX)
public class RagKitProperties {

	public static final String PREFIX = "ragkit";

	private final FullText fulltext = new FullText();

	private final Hybrid hybrid = new Hybrid();

	public FullText getFulltext() {
		return this.fulltext;
	}

	public Hybrid getHybrid() {
		return this.hybrid;
	}

	public static class FullText {

		/** Whether to create the full-text DocumentRetriever bean. */
		private boolean enabled = true;

		/**
		 * Whether to add the generated tsvector column and GIN index on startup. Off by
		 * default because it alters (and on first run rewrites) the vector store table.
		 */
		private boolean initializeSchema = false;

		/** Schema of the vector store table. Defaults to PgVectorStore's. */
		private String schemaName = "public";

		/** Vector store table. Defaults to PgVectorStore's. */
		private String tableName = "vector_store";

		/** Name of the generated tsvector column. */
		private String tsvectorColumn = "content_tsv";

		/** PostgreSQL text search configuration, e.g. english or simple. */
		private String textSearchConfig = "english";

		/** How query text becomes a tsquery. */
		private TextQueryMode queryMode = TextQueryMode.ANY_TERMS;

		/** Split camelCase and dotted identifiers into words. */
		private boolean splitIdentifiers = true;

		/** Read the stored tsvector column instead of computing it per query. */
		private boolean useStoredTsvector = true;

		/** Maximum documents returned by the full-text retriever. */
		private int topK = 20;

		public boolean isEnabled() {
			return this.enabled;
		}

		public void setEnabled(boolean enabled) {
			this.enabled = enabled;
		}

		public boolean isInitializeSchema() {
			return this.initializeSchema;
		}

		public void setInitializeSchema(boolean initializeSchema) {
			this.initializeSchema = initializeSchema;
		}

		public String getSchemaName() {
			return this.schemaName;
		}

		public void setSchemaName(String schemaName) {
			this.schemaName = schemaName;
		}

		public String getTableName() {
			return this.tableName;
		}

		public void setTableName(String tableName) {
			this.tableName = tableName;
		}

		public String getTsvectorColumn() {
			return this.tsvectorColumn;
		}

		public void setTsvectorColumn(String tsvectorColumn) {
			this.tsvectorColumn = tsvectorColumn;
		}

		public String getTextSearchConfig() {
			return this.textSearchConfig;
		}

		public void setTextSearchConfig(String textSearchConfig) {
			this.textSearchConfig = textSearchConfig;
		}

		public TextQueryMode getQueryMode() {
			return this.queryMode;
		}

		public void setQueryMode(TextQueryMode queryMode) {
			this.queryMode = queryMode;
		}

		public boolean isSplitIdentifiers() {
			return this.splitIdentifiers;
		}

		public void setSplitIdentifiers(boolean splitIdentifiers) {
			this.splitIdentifiers = splitIdentifiers;
		}

		public boolean isUseStoredTsvector() {
			return this.useStoredTsvector;
		}

		public void setUseStoredTsvector(boolean useStoredTsvector) {
			this.useStoredTsvector = useStoredTsvector;
		}

		public int getTopK() {
			return this.topK;
		}

		public void setTopK(int topK) {
			this.topK = topK;
		}

	}

	public static class Hybrid {

		/** Whether to create the hybrid (vector + full-text) DocumentRetriever bean. */
		private boolean enabled = true;

		/** Maximum documents returned by the vector retriever before fusion. */
		private int vectorTopK = 20;

		/** Minimum similarity for the vector retriever. */
		private double similarityThreshold = 0.0;

		/** Maximum fused documents returned. */
		private int topK = 5;

		/** RRF constant k; larger values flatten the advantage of top ranks. */
		private int rrfK = 60;

		/** Weight of the vector retriever's ranks. */
		private double vectorWeight = 1.0;

		/** Weight of the full-text retriever's ranks. */
		private double fulltextWeight = 1.0;

		public boolean isEnabled() {
			return this.enabled;
		}

		public void setEnabled(boolean enabled) {
			this.enabled = enabled;
		}

		public int getVectorTopK() {
			return this.vectorTopK;
		}

		public void setVectorTopK(int vectorTopK) {
			this.vectorTopK = vectorTopK;
		}

		public double getSimilarityThreshold() {
			return this.similarityThreshold;
		}

		public void setSimilarityThreshold(double similarityThreshold) {
			this.similarityThreshold = similarityThreshold;
		}

		public int getTopK() {
			return this.topK;
		}

		public void setTopK(int topK) {
			this.topK = topK;
		}

		public int getRrfK() {
			return this.rrfK;
		}

		public void setRrfK(int rrfK) {
			this.rrfK = rrfK;
		}

		public double getVectorWeight() {
			return this.vectorWeight;
		}

		public void setVectorWeight(double vectorWeight) {
			this.vectorWeight = vectorWeight;
		}

		public double getFulltextWeight() {
			return this.fulltextWeight;
		}

		public void setFulltextWeight(double fulltextWeight) {
			this.fulltextWeight = fulltextWeight;
		}

	}

}
