package io.github.derekdiliu.ragkit.rerank;

import java.util.List;

/**
 * Scores how relevant each text is to a query, reading both together: a cross-encoder
 * model or a reranking API. Higher is more relevant.
 */
@FunctionalInterface
public interface RelevanceScorer {

	/**
	 * @param query the query text
	 * @param texts candidate texts
	 * @return one score per text, in the same order
	 */
	List<Double> score(String query, List<String> texts);

}
