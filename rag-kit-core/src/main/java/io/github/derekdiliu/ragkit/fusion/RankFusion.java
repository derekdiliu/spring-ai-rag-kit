package io.github.derekdiliu.ragkit.fusion;

import java.util.List;

import org.springframework.ai.document.Document;

/**
 * Merges several ranked lists of documents into one.
 * <p>
 * Implementations deduplicate by {@link Document#getId()}, set {@link Document#getScore()}
 * to the fused score, and record per-source ranks under
 * {@link FusionMetadata#SOURCE_RANKS}. Input documents are never mutated.
 */
@FunctionalInterface
public interface RankFusion {

	/**
	 * @param rankedLists the lists to merge, each ordered best match first
	 * @return the fused documents, best match first, without a size limit
	 */
	List<Document> fuse(List<RankedList> rankedLists);

	/** Reciprocal Rank Fusion with the conventional {@code k = 60}. */
	static RankFusion rrf() {
		return new ReciprocalRankFusion(ReciprocalRankFusion.DEFAULT_K);
	}

	/** Reciprocal Rank Fusion with a custom {@code k}. */
	static RankFusion rrf(int k) {
		return new ReciprocalRankFusion(k);
	}

	/** Weighted sum of min-max normalised scores. */
	static RankFusion weightedScore() {
		return new WeightedScoreFusion();
	}

}
