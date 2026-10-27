package io.github.derekdiliu.ragkit.benchmark;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import com.zaxxer.hikari.HikariDataSource;
import io.github.derekdiliu.ragkit.eval.EvalReport;
import io.github.derekdiliu.ragkit.eval.Metric;
import io.github.derekdiliu.ragkit.eval.RetrievalEvaluator;
import io.github.derekdiliu.ragkit.pgvector.PgFullTextDocumentRetriever;
import io.github.derekdiliu.ragkit.pgvector.PgTextSearchContainers;
import io.github.derekdiliu.ragkit.pgvector.TextRanking;
import io.github.derekdiliu.ragkit.retrieval.HybridDocumentRetriever;

import org.springframework.ai.document.Document;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiEmbeddingModel;
import org.springframework.ai.openai.OpenAiEmbeddingOptions;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.transformers.TransformersEmbeddingModel;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.jdbc.core.JdbcTemplate;

/**
 * Compares full-text ({@code ts_rank} and BM25), vector and hybrid retrieval on BEIR test
 * splits, all on the same PgVectorStore table in a throwaway container with pgvector and
 * pg_textsearch.
 * <p>
 * System properties: {@code benchmark.datasets} (default {@code scifact,fiqa}),
 * {@code benchmark.models} ({@code minilm} and/or {@code openai}; default {@code minilm};
 * {@code openai} reads {@code OPENAI_API_KEY}), {@code benchmark.cache} (dataset cache
 * directory) and {@code benchmark.output} (Markdown results file).
 * <p>
 * Every retriever returns its top {@value #DEPTH}. The vector table has no ANN index, so
 * vector search is exact and the numbers reflect the embedding model, not index recall.
 */
public final class BeirBenchmark {

	private static final int DEPTH = 100;

	private BeirBenchmark() {
	}

	public static void main(String[] args) throws Exception {
		List<String> datasets = list(System.getProperty("benchmark.datasets", "scifact,fiqa"));
		List<String> models = list(System.getProperty("benchmark.models", "minilm"));
		Path cache = Path.of(System.getProperty("benchmark.cache",
				System.getProperty("user.home") + "/.cache/ragkit/beir"));
		Path output = Path.of(System.getProperty("benchmark.output", "beir-results.md"));

		StringBuilder results = new StringBuilder()
			.append("| Dataset | Retriever | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |\n")
			.append("| --- | --- | ---: | ---: | ---: | ---: |\n");

		try (var postgres = PgTextSearchContainers.create()) {
			postgres.start();
			try (var dataSource = new HikariDataSource()) {
				dataSource.setJdbcUrl(postgres.getJdbcUrl());
				dataSource.setUsername(postgres.getUsername());
				dataSource.setPassword(postgres.getPassword());
				var jdbcTemplate = new JdbcTemplate(dataSource);

				for (String datasetName : datasets) {
					BeirDataset dataset = BeirDataset.load(datasetName, cache);
					System.out.printf("%s: %,d documents, %,d test queries%n", datasetName, dataset.corpus().size(),
							dataset.testCases().size());
					boolean fullTextReported = false;
					for (String model : models) {
						EmbeddingModel embeddingModel = embeddingModel(model);
						String table = datasetName + "_" + model;
						PgVectorStore vectorStore = PgVectorStore.builder(jdbcTemplate, embeddingModel)
							.vectorTableName(table)
							.dimensions(embeddingModel.dimensions())
							.indexType(PgVectorStore.PgIndexType.NONE)
							.initializeSchema(true)
							.build();
						vectorStore.afterPropertiesSet();
						ingest(vectorStore, dataset.corpus(), table);

						var fullText = PgFullTextDocumentRetriever.builder(jdbcTemplate)
							.tableName(table)
							.topK(DEPTH)
							.build();
						fullText.initializeSchema();
						var bm25 = PgFullTextDocumentRetriever.builder(jdbcTemplate)
							.tableName(table)
							.ranking(TextRanking.BM25)
							.topK(DEPTH)
							.build();
						bm25.initializeSchema();
						var vector = VectorStoreDocumentRetriever.builder().vectorStore(vectorStore).topK(DEPTH).build();
						var hybrid = HybridDocumentRetriever.builder()
							.retriever("vector", vector)
							.retriever("fulltext", fullText)
							.topK(DEPTH)
							.build();
						var hybridBm25 = HybridDocumentRetriever.builder()
							.retriever("vector", vector)
							.retriever("fulltext", bm25)
							.topK(DEPTH)
							.build();

						if (!fullTextReported) {
							results.append(row(dataset, "Full-text (`ts_rank`)", fullText));
							results.append(row(dataset, "Full-text (BM25)", bm25));
							fullTextReported = true;
						}
						results.append(row(dataset, "Vector (" + label(model) + ")", vector));
						results.append(row(dataset, "Hybrid RRF (" + label(model) + " + `ts_rank`)", hybrid));
						results.append(row(dataset, "Hybrid RRF (" + label(model) + " + BM25)", hybridBm25));
					}
				}
			}
		}

		Files.createDirectories(output.toAbsolutePath().getParent());
		Files.writeString(output, results);
		System.out.println();
		System.out.println(results);
		System.out.println("Written to " + output.toAbsolutePath());
	}

	private static EmbeddingModel embeddingModel(String model) throws Exception {
		if (model.equals("minilm")) {
			var minilm = new TransformersEmbeddingModel(MetadataMode.NONE);
			minilm.afterPropertiesSet();
			return minilm;
		}
		if (model.equals("openai")) {
			String apiKey = System.getenv("OPENAI_API_KEY");
			if (apiKey == null || apiKey.isBlank()) {
				throw new IllegalStateException("benchmark.models includes openai but OPENAI_API_KEY is not set");
			}
			return OpenAiEmbeddingModel.builder()
				.options(OpenAiEmbeddingOptions.builder().apiKey(apiKey).model("text-embedding-3-small").build())
				.metadataMode(MetadataMode.NONE)
				.build();
		}
		throw new IllegalArgumentException("Unknown model " + model + "; use minilm or openai");
	}

	private static String label(String model) {
		return model.equals("openai") ? "text-embedding-3-small" : "all-MiniLM-L6-v2";
	}

	private static void ingest(PgVectorStore vectorStore, List<Document> corpus, String table) {
		long start = System.nanoTime();
		int batch = 1000;
		for (int from = 0; from < corpus.size(); from += batch) {
			vectorStore.add(corpus.subList(from, Math.min(from + batch, corpus.size())));
			System.out.printf("  %s: embedded %,d / %,d%n", table, Math.min(from + batch, corpus.size()),
					corpus.size());
		}
		System.out.printf("  %s: ingested in %s%n", table, Duration.ofNanos(System.nanoTime() - start));
	}

	private static String row(BeirDataset dataset, String retrieverName, DocumentRetriever retriever) {
		long start = System.nanoTime();
		EvalReport report = RetrievalEvaluator.of(retriever)
			.documentId(document -> (String) document.getMetadata().get(BeirDataset.ID_KEY))
			.evaluate(dataset.testCases());
		System.out.printf("  %s / %s evaluated in %s%n", dataset.name(), retrieverName,
				Duration.ofNanos(System.nanoTime() - start));
		return String.format(Locale.ROOT, "| %s | %s | %.3f | %.3f | %.3f | %.3f |%n", dataset.name(), retrieverName,
				report.mean(Metric.NDCG, 10), report.mean(Metric.RECALL, 10), report.mean(Metric.RECALL, DEPTH),
				report.mean(Metric.MRR, 10));
	}

	private static List<String> list(String commaSeparated) {
		return Arrays.stream(commaSeparated.split(",")).map(String::trim).filter(s -> !s.isEmpty()).toList();
	}

}
