package io.github.derekdiliu.ragkit.eval;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.within;

class MetricTests {

	private static final Map<String, Integer> ABC = Map.of("a", 1, "b", 1, "c", 1);

	private static double log2(int x) {
		return Math.log(x) / Math.log(2);
	}

	@Test
	void recallIsTheShareOfRelevantDocumentsInTheTopK() {
		List<String> ranked = List.of("a", "x", "b", "y");

		assertThat(Metric.RECALL.score(ranked, ABC, 2)).isCloseTo(1.0 / 3, within(1e-12));
		assertThat(Metric.RECALL.score(ranked, ABC, 4)).isCloseTo(2.0 / 3, within(1e-12));
	}

	@Test
	void mrrUsesTheFirstRelevantRankWithinK() {
		List<String> ranked = List.of("x", "y", "b");

		assertThat(Metric.MRR.score(ranked, ABC, 10)).isCloseTo(1.0 / 3, within(1e-12));
		assertThat(Metric.MRR.score(ranked, ABC, 2)).isZero();
	}

	@Test
	void hitRateIsOneWhenAnyRelevantDocumentIsInTheTopK() {
		List<String> ranked = List.of("x", "a");

		assertThat(Metric.HIT_RATE.score(ranked, ABC, 1)).isZero();
		assertThat(Metric.HIT_RATE.score(ranked, ABC, 2)).isEqualTo(1.0);
	}

	@Test
	void ndcgUsesGradesAsLinearGainAgainstTheIdealRanking() {
		Map<String, Integer> graded = Map.of("a", 2, "b", 1);

		double dcg = 0 / log2(2) + 2 / log2(3) + 1 / log2(4);
		double idcg = 2 / log2(2) + 1 / log2(3);
		assertThat(Metric.NDCG.score(List.of("x", "a", "b"), graded, 10)).isCloseTo(dcg / idcg, within(1e-12));
		assertThat(Metric.NDCG.score(List.of("a", "b"), graded, 10)).isCloseTo(1.0, within(1e-12));
	}

	@Test
	void ndcgIdealRankingIsCutAtK() {
		// Only one slot, filled by a relevant document: as good as it can be at k=1.
		assertThat(Metric.NDCG.score(List.of("a", "x"), ABC, 1)).isCloseTo(1.0, within(1e-12));
	}

	@Test
	void eachIdCountsOnceAtItsFirstPosition() {
		List<String> ranked = List.of("a", "a", "a", "b");

		assertThat(Metric.RECALL.score(ranked, ABC, 2)).isCloseTo(2.0 / 3, within(1e-12));
		assertThat(Metric.NDCG.score(ranked, Map.of("a", 1), 3)).isCloseTo(1.0, within(1e-12));
	}

	@Test
	void labelsFollowTheUsualNotation() {
		assertThat(Metric.NDCG.label(10)).isEqualTo("ndcg@10");
		assertThat(Metric.HIT_RATE.label(5)).isEqualTo("hit_rate@5");
	}

	@Test
	void rejectsNonPositiveK() {
		assertThatIllegalArgumentException().isThrownBy(() -> Metric.RECALL.score(List.of("a"), ABC, 0));
	}

}
