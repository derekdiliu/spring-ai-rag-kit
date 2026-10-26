package io.github.derekdiliu.ragkit.fusion;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.within;

class ReciprocalRankFusionTests {

	private static Document doc(String id, double score) {
		return Document.builder().id(id).text("text " + id).score(score).build();
	}

	@Test
	void documentFoundByBothListsOutranksDocumentsFoundByOne() {
		// "b" is second in both lists; "a" and "c" each top only one list.
		var vector = RankedList.of("vector", List.of(doc("a", 0.91), doc("b", 0.90), doc("d", 0.50)));
		var fulltext = RankedList.of("fulltext", List.of(doc("c", 0.08), doc("b", 0.07)));

		List<Document> fused = RankFusion.rrf().fuse(List.of(vector, fulltext));

		assertThat(fused).extracting(Document::getId).containsExactly("b", "a", "c", "d");
		assertThat(fused.get(0).getScore()).isCloseTo(2.0 / 62, within(1e-12));
		assertThat(fused.get(1).getScore()).isCloseTo(1.0 / 61, within(1e-12));
	}

	@Test
	void ignoresRawScoreScales() {
		// Full-text scores are tiny compared with cosine similarity; only ranks matter.
		var vector = RankedList.of("vector", List.of(doc("a", 0.99)));
		var fulltext = RankedList.of("fulltext", List.of(doc("b", 0.0001)));

		List<Document> fused = RankFusion.rrf().fuse(List.of(vector, fulltext));

		assertThat(fused.get(0).getScore()).isEqualTo(fused.get(1).getScore());
	}

	@Test
	void weightScalesAListsContribution() {
		var vector = new RankedList("vector", 1.0, List.of(doc("a", 0.9)));
		var fulltext = new RankedList("fulltext", 3.0, List.of(doc("b", 0.1)));

		List<Document> fused = RankFusion.rrf(0).fuse(List.of(vector, fulltext));

		assertThat(fused).extracting(Document::getId).containsExactly("b", "a");
		assertThat(fused.get(0).getScore()).isCloseTo(3.0, within(1e-12));
	}

	@Test
	void recordsSourceRanksAndFusionScoreInMetadata() {
		var vector = RankedList.of("vector", List.of(doc("a", 0.9), doc("b", 0.8)));
		var fulltext = RankedList.of("fulltext", List.of(doc("b", 0.2)));

		Document b = RankFusion.rrf().fuse(List.of(vector, fulltext)).get(0);

		assertThat(b.getId()).isEqualTo("b");
		assertThat(b.getMetadata()).containsEntry(FusionMetadata.SOURCE_RANKS, Map.of("vector", 2, "fulltext", 1));
		assertThat(b.getMetadata().get(FusionMetadata.FUSION_SCORE)).isEqualTo(b.getScore());
	}

	@Test
	void duplicateWithinOneListCountsOnceAtBestRank() {
		var list = RankedList.of("vector", List.of(doc("a", 0.9), doc("b", 0.8), doc("a", 0.7)));

		List<Document> fused = RankFusion.rrf().fuse(List.of(list));

		assertThat(fused).extracting(Document::getId).containsExactly("a", "b");
		assertThat(fused.get(0).getScore()).isCloseTo(1.0 / 61, within(1e-12));
		assertThat(fused.get(0).getMetadata()).containsEntry(FusionMetadata.SOURCE_RANKS, Map.of("vector", 1));
	}

	@Test
	void doesNotMutateInputDocuments() {
		Document original = doc("a", 0.9);
		original.getMetadata().put("path", "A.java");

		Document fused = RankFusion.rrf().fuse(List.of(RankedList.of("vector", List.of(original)))).get(0);

		assertThat(original.getScore()).isEqualTo(0.9);
		assertThat(original.getMetadata()).containsOnlyKeys("path");
		assertThat(fused.getMetadata()).containsKeys("path", FusionMetadata.FUSION_SCORE, FusionMetadata.SOURCE_RANKS);
		assertThat(fused.getText()).isEqualTo(original.getText());
	}

	@Test
	void tiesKeepFirstSeenOrder() {
		var first = RankedList.of("one", List.of(doc("x", 0.1)));
		var second = RankedList.of("two", List.of(doc("y", 0.9)));

		assertThat(RankFusion.rrf().fuse(List.of(first, second))).extracting(Document::getId).containsExactly("x", "y");
	}

	@Test
	void emptyInputGivesEmptyOutput() {
		assertThat(RankFusion.rrf().fuse(List.of())).isEmpty();
		assertThat(RankFusion.rrf().fuse(List.of(RankedList.of("vector", List.of())))).isEmpty();
	}

	@Test
	void rejectsNegativeK() {
		assertThatIllegalArgumentException().isThrownBy(() -> RankFusion.rrf(-1));
	}

}
