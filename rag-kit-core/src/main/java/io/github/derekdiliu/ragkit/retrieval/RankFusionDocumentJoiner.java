package io.github.derekdiliu.ragkit.retrieval;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.github.derekdiliu.ragkit.fusion.RankFusion;
import io.github.derekdiliu.ragkit.fusion.RankedList;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.join.DocumentJoiner;
import org.springframework.util.Assert;

/**
 * A {@link DocumentJoiner} that merges the per-query result lists produced by query
 * expansion (e.g. {@code MultiQueryExpander}) using rank fusion.
 * <p>
 * Spring AI's default {@code ConcatenationDocumentJoiner} sorts the union by raw score,
 * so a document that ranks well for several expanded queries gains nothing over one that
 * appears once. Rank fusion rewards agreement across queries, which is the point of
 * expanding them.
 *
 * <pre>{@code
 * RetrievalAugmentationAdvisor.builder()
 *     .queryExpander(MultiQueryExpander.builder().chatClientBuilder(builder).build())
 *     .documentRetriever(retriever)
 *     .documentJoiner(new RankFusionDocumentJoiner())
 *     .build();
 * }</pre>
 */
public final class RankFusionDocumentJoiner implements DocumentJoiner {

	private final RankFusion fusion;

	private final int topK;

	/** RRF with {@code k = 60} and no result limit. */
	public RankFusionDocumentJoiner() {
		this(RankFusion.rrf(), Integer.MAX_VALUE);
	}

	public RankFusionDocumentJoiner(RankFusion fusion, int topK) {
		Assert.notNull(fusion, "fusion must not be null");
		Assert.isTrue(topK > 0, "topK must be positive");
		this.fusion = fusion;
		this.topK = topK;
	}

	@Override
	public List<Document> join(Map<Query, List<List<Document>>> documentsForQuery) {
		Assert.notNull(documentsForQuery, "documentsForQuery must not be null");
		Assert.noNullElements(documentsForQuery.keySet(), "documentsForQuery must not contain null keys");
		Assert.noNullElements(documentsForQuery.values(), "documentsForQuery must not contain null values");

		List<RankedList> rankedLists = new ArrayList<>();
		int queryIndex = 0;
		for (Map.Entry<Query, List<List<Document>>> entry : documentsForQuery.entrySet()) {
			int listIndex = 0;
			for (List<Document> documents : entry.getValue()) {
				rankedLists.add(RankedList.of("query-" + queryIndex + "-" + listIndex, documents));
				listIndex++;
			}
			queryIndex++;
		}

		List<Document> fused = this.fusion.fuse(rankedLists);
		return fused.size() > this.topK ? List.copyOf(fused.subList(0, this.topK)) : fused;
	}

}
