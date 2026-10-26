package io.github.derekdiliu.ragkit.fusion;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.ai.document.Document;

/**
 * Weighted sum of per-list min-max normalised scores.
 * <p>
 * Within each list, scores are rescaled to {@code [0, 1]} before being multiplied by the
 * list's weight. A list whose scores are all equal (or missing) falls back to a linear
 * rank-based score so that its ordering still counts. A document absent from a list
 * contributes 0 for that list.
 * <p>
 * Prefer {@link ReciprocalRankFusion} unless you have evaluated that score magnitudes
 * carry signal for your data; normalised scores are sensitive to outliers.
 */
public final class WeightedScoreFusion extends AbstractRankFusion {

	@Override
	protected void contribute(RankedList list, Map<String, Candidate> candidates) {
		List<Document> documents = list.documents();
		if (documents.isEmpty()) {
			return;
		}

		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		boolean allScored = true;
		for (Document document : documents) {
			Double score = document.getScore();
			if (score == null || !Double.isFinite(score)) {
				allScored = false;
				break;
			}
			min = Math.min(min, score);
			max = Math.max(max, score);
		}
		boolean useScores = allScored && max > min;

		Set<String> seen = new HashSet<>();
		int size = documents.size();
		for (int i = 0; i < size; i++) {
			Document document = documents.get(i);
			int rank = i + 1;
			Candidate candidate = candidate(candidates, document, list, rank);
			if (!seen.add(document.getId())) {
				continue;
			}
			double normalised = useScores ? (document.getScore() - min) / (max - min)
					: (size == 1 ? 1.0 : 1.0 - (double) i / (size - 1));
			candidate.add(list.weight() * normalised);
		}
	}

	@Override
	public String toString() {
		return "WeightedScoreFusion";
	}

}
