package io.github.derekdiliu.ragkit.fusion;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.ai.document.Document;
import org.springframework.util.Assert;

/**
 * Shared bookkeeping for fusion strategies: deduplication by id, per-source rank tracking,
 * stable ordering and building the output documents.
 */
abstract class AbstractRankFusion implements RankFusion {

	@Override
	public final List<Document> fuse(List<RankedList> rankedLists) {
		Assert.notNull(rankedLists, "rankedLists must not be null");
		Assert.noNullElements(rankedLists, "rankedLists must not contain null elements");

		Map<String, Candidate> candidates = new LinkedHashMap<>();
		for (RankedList list : rankedLists) {
			contribute(list, candidates);
		}

		List<Candidate> ordered = new ArrayList<>(candidates.values());
		// Higher score first; ties keep first-seen order (LinkedHashMap + stable sort).
		ordered.sort(Comparator.comparingDouble(Candidate::score).reversed());

		List<Document> result = new ArrayList<>(ordered.size());
		for (Candidate candidate : ordered) {
			result.add(candidate.toDocument());
		}
		return result;
	}

	/**
	 * Adds the contribution of one ranked list to the candidates. Implementations call
	 * {@link #candidate(Map, Document, RankedList, int)} for every document they score.
	 */
	protected abstract void contribute(RankedList list, Map<String, Candidate> candidates);

	protected static Candidate candidate(Map<String, Candidate> candidates, Document document, RankedList list,
			int rank) {
		Candidate candidate = candidates.computeIfAbsent(document.getId(), id -> new Candidate(document));
		// If the same source returns a document twice, keep its best (lowest) rank.
		candidate.sourceRanks.merge(list.source(), rank, Math::min);
		return candidate;
	}

	protected static final class Candidate {

		private final Document document;

		private final Map<String, Integer> sourceRanks = new LinkedHashMap<>();

		private double score;

		private Candidate(Document document) {
			this.document = document;
		}

		void add(double contribution) {
			this.score += contribution;
		}

		double score() {
			return this.score;
		}

		private Document toDocument() {
			Map<String, Object> metadata = new HashMap<>(this.document.getMetadata());
			metadata.put(FusionMetadata.FUSION_SCORE, this.score);
			metadata.put(FusionMetadata.SOURCE_RANKS, Map.copyOf(this.sourceRanks));
			return this.document.mutate().metadata(metadata).score(this.score).build();
		}

	}

}
