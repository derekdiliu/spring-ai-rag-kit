package io.github.derekdiliu.ragkit.fusion;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.ai.document.Document;
import org.springframework.util.Assert;

/**
 * Reciprocal Rank Fusion (Cormack, Clarke and Büttcher, SIGIR 2009).
 * <p>
 * Each document scores {@code sum(weight_i / (k + rank_i))} over the lists that contain
 * it, with 1-based ranks. Only ranks are used, so lists whose raw scores live on
 * different scales (cosine similarity vs {@code ts_rank_cd}) can be merged without any
 * tuning. Larger {@code k} flattens the advantage of top positions.
 */
public final class ReciprocalRankFusion extends AbstractRankFusion {

	public static final int DEFAULT_K = 60;

	private final int k;

	public ReciprocalRankFusion(int k) {
		Assert.isTrue(k >= 0, "k must not be negative");
		this.k = k;
	}

	public int getK() {
		return this.k;
	}

	@Override
	protected void contribute(RankedList list, Map<String, Candidate> candidates) {
		List<Document> documents = list.documents();
		Set<String> seen = new HashSet<>();
		for (int i = 0; i < documents.size(); i++) {
			Document document = documents.get(i);
			int rank = i + 1;
			Candidate candidate = candidate(candidates, document, list, rank);
			// A duplicate within the same list only counts at its best position.
			if (seen.add(document.getId())) {
				candidate.add(list.weight() / (this.k + rank));
			}
		}
	}

	@Override
	public String toString() {
		return "ReciprocalRankFusion[k=" + this.k + "]";
	}

}
