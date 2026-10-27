package io.github.derekdiliu.ragkit.rerank;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.postretrieval.document.DocumentPostProcessor;
import org.springframework.util.Assert;

/**
 * Re-orders retrieved documents by a {@link RelevanceScorer} and keeps the best
 * {@code topN}. Retrieve generously (e.g. 50 candidates), then let the reranker pick what
 * reaches the prompt:
 *
 * <pre>{@code
 * var advisor = RetrievalAugmentationAdvisor.builder()
 *     .documentRetriever(hybridRetriever)              // topK(50)
 *     .documentPostProcessors(RerankingDocumentPostProcessor.builder(crossEncoder).topN(5).build())
 *     .build();
 * }</pre>
 *
 * Each returned document's score is the reranker's, also stored under
 * {@link #RERANK_SCORE}; other metadata, such as fusion ranks, is kept.
 */
public final class RerankingDocumentPostProcessor implements DocumentPostProcessor {

	/** Metadata key holding the reranker's score. */
	public static final String RERANK_SCORE = "ragkit_rerank_score";

	private final RelevanceScorer scorer;

	private final int topN;

	private RerankingDocumentPostProcessor(Builder builder) {
		this.scorer = builder.scorer;
		this.topN = builder.topN;
	}

	public static Builder builder(RelevanceScorer scorer) {
		return new Builder(scorer);
	}

	@Override
	public List<Document> process(Query query, List<Document> documents) {
		Assert.notNull(query, "query must not be null");
		Assert.notNull(documents, "documents must not be null");
		if (documents.isEmpty()) {
			return List.of();
		}
		List<String> texts = documents.stream().map(d -> d.getText() != null ? d.getText() : "").toList();
		List<Double> scores = this.scorer.score(query.text(), texts);
		Assert.state(scores != null && scores.size() == documents.size(),
				() -> "scorer returned " + (scores == null ? "null" : scores.size() + " scores") + " for "
						+ documents.size() + " documents");

		List<Document> rescored = new ArrayList<>(documents.size());
		for (int i = 0; i < documents.size(); i++) {
			Document document = documents.get(i);
			double score = scores.get(i);
			Map<String, Object> metadata = new HashMap<>(document.getMetadata());
			metadata.put(RERANK_SCORE, score);
			rescored.add(document.mutate().metadata(metadata).score(score).build());
		}
		// Stable sort: equal scores keep the retriever's order.
		rescored.sort(Comparator.comparingDouble(Document::getScore).reversed());
		return rescored.size() > this.topN ? List.copyOf(rescored.subList(0, this.topN)) : rescored;
	}

	public int getTopN() {
		return this.topN;
	}

	public static final class Builder {

		public static final int DEFAULT_TOP_N = 5;

		private final RelevanceScorer scorer;

		private int topN = DEFAULT_TOP_N;

		private Builder(RelevanceScorer scorer) {
			Assert.notNull(scorer, "scorer must not be null");
			this.scorer = scorer;
		}

		/** Maximum documents kept after reranking. Defaults to 5. */
		public Builder topN(int topN) {
			Assert.isTrue(topN > 0, "topN must be positive");
			this.topN = topN;
			return this;
		}

		public RerankingDocumentPostProcessor build() {
			return new RerankingDocumentPostProcessor(this);
		}

	}

}
