package io.github.derekdiliu.ragkit.eval;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.util.Assert;

/**
 * One evaluation query and the documents that count as relevant for it.
 *
 * @param id identifies the query in reports
 * @param query the text sent to the retriever
 * @param relevance relevant document ids mapped to a positive grade (1 = relevant, higher
 * = more relevant); the grade is the gain in nDCG, other metrics only check presence
 */
public record EvalCase(String id, String query, Map<String, Integer> relevance) {

	public EvalCase {
		Assert.hasText(id, "id must not be empty");
		Assert.hasText(query, "query must not be empty");
		Assert.notEmpty(relevance, () -> "case " + id + " needs at least one relevant document");
		Assert.isTrue(relevance.values().stream().allMatch(grade -> grade != null && grade > 0),
				() -> "case " + id + " has a relevance grade that is not positive");
		relevance = Map.copyOf(relevance);
	}

	/** A case whose relevant documents all have grade 1. */
	public static EvalCase of(String id, String query, String... relevantIds) {
		return new EvalCase(id, query,
				Arrays.stream(relevantIds).distinct().collect(Collectors.toMap(Function.identity(), rid -> 1)));
	}

}
