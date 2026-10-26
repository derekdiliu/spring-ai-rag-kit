package io.github.derekdiliu.ragkit.eval;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.util.Assert;

/**
 * Ranking metrics for a single query, computed over the top {@code k} retrieved ids. Each
 * id counts once, at its first position. Definitions follow trec_eval and BEIR, so
 * numbers are comparable with published results.
 */
public enum Metric {

	/** Fraction of the relevant documents that appear in the top k. */
	RECALL("recall") {
		@Override
		double compute(List<String> topK, Map<String, Integer> relevance, int k) {
			return topK.stream().filter(relevance::containsKey).count() / (double) relevance.size();
		}
	},

	/** Reciprocal rank of the first relevant document in the top k, or 0. */
	MRR("mrr") {
		@Override
		double compute(List<String> topK, Map<String, Integer> relevance, int k) {
			for (int i = 0; i < topK.size(); i++) {
				if (relevance.containsKey(topK.get(i))) {
					return 1.0 / (i + 1);
				}
			}
			return 0;
		}
	},

	/**
	 * Normalised discounted cumulative gain with the relevance grade as a linear gain,
	 * divided by the gain of the ideal ranking.
	 */
	NDCG("ndcg") {
		@Override
		double compute(List<String> topK, Map<String, Integer> relevance, int k) {
			double dcg = 0;
			for (int i = 0; i < topK.size(); i++) {
				dcg += relevance.getOrDefault(topK.get(i), 0) / log2(i + 2);
			}
			List<Integer> ideal = relevance.values().stream().sorted((a, b) -> b - a).limit(k).toList();
			double idcg = 0;
			for (int i = 0; i < ideal.size(); i++) {
				idcg += ideal.get(i) / log2(i + 2);
			}
			return dcg / idcg;
		}
	},

	/** 1 if any relevant document appears in the top k, else 0. */
	HIT_RATE("hit_rate") {
		@Override
		double compute(List<String> topK, Map<String, Integer> relevance, int k) {
			return topK.stream().anyMatch(relevance::containsKey) ? 1 : 0;
		}
	};

	private final String name;

	Metric(String name) {
		this.name = name;
	}

	/**
	 * Scores one query.
	 * @param rankedIds retrieved document ids, best first
	 * @param relevance relevant ids mapped to positive grades, as in {@link EvalCase}
	 * @param k cut-off; only the first {@code k} distinct ids are considered
	 */
	public double score(List<String> rankedIds, Map<String, Integer> relevance, int k) {
		Assert.isTrue(k > 0, "k must be positive");
		Assert.notEmpty(relevance, "relevance must not be empty");
		Set<String> seen = new HashSet<>();
		List<String> topK = rankedIds.stream().filter(seen::add).limit(k).toList();
		return compute(topK, relevance, k);
	}

	abstract double compute(List<String> topK, Map<String, Integer> relevance, int k);

	/** The conventional label, e.g. {@code ndcg@10}. */
	public String label(int k) {
		return this.name + "@" + k;
	}

	private static double log2(int x) {
		return Math.log(x) / Math.log(2);
	}

}
