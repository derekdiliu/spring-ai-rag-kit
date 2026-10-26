package io.github.derekdiliu.ragkit.eval;

import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import org.springframework.util.Assert;

/**
 * The ranked ids retrieved for each {@link EvalCase}. Metrics are computed on demand, so
 * one evaluation run can be reported at several cut-offs.
 */
public final class EvalReport {

	private static final int WORST_QUERIES_SHOWN = 5;

	private final List<QueryResult> results;

	public EvalReport(List<QueryResult> results) {
		Assert.notEmpty(results, "results must not be empty");
		this.results = List.copyOf(results);
	}

	public List<QueryResult> getResults() {
		return this.results;
	}

	/** The metric averaged over all queries. */
	public double mean(Metric metric, int k) {
		return this.results.stream().mapToDouble(result -> result.score(metric, k)).average().orElseThrow();
	}

	/**
	 * Starts an assertion on the mean of {@code metric} at {@code k}. A failure lists the
	 * lowest-scoring queries.
	 */
	public MetricAssert assertThat(Metric metric, int k) {
		return new MetricAssert(metric, k);
	}

	/**
	 * What the retriever returned for one case.
	 *
	 * @param evalCase the case
	 * @param rankedIds distinct document ids, best first
	 */
	public record QueryResult(EvalCase evalCase, List<String> rankedIds) {

		public QueryResult {
			Assert.notNull(evalCase, "evalCase must not be null");
			rankedIds = List.copyOf(rankedIds);
		}

		public double score(Metric metric, int k) {
			return metric.score(this.rankedIds, this.evalCase.relevance(), k);
		}

	}

	public final class MetricAssert {

		private final Metric metric;

		private final int k;

		private MetricAssert(Metric metric, int k) {
			this.metric = metric;
			this.k = k;
		}

		/** Fails with an {@link AssertionError} if the mean is below {@code minimum}. */
		public MetricAssert isAtLeast(double minimum) {
			double actual = mean(this.metric, this.k);
			if (actual < minimum) {
				String worst = EvalReport.this.results.stream()
					.sorted(Comparator.comparingDouble(result -> result.score(this.metric, this.k)))
					.limit(WORST_QUERIES_SHOWN)
					.map(result -> format("%s (%.2f)", result.evalCase().id(), result.score(this.metric, this.k)))
					.collect(Collectors.joining(", "));
				throw new AssertionError(format("%s was %.4f, expected at least %.4f; lowest-scoring queries: %s",
						this.metric.label(this.k), actual, minimum, worst));
			}
			return this;
		}

	}

	private static String format(String pattern, Object... args) {
		return String.format(Locale.ROOT, pattern, args);
	}

}
