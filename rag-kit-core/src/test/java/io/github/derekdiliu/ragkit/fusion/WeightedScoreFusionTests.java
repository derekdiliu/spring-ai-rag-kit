package io.github.derekdiliu.ragkit.fusion;

import java.util.List;

import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class WeightedScoreFusionTests {

	private static Document doc(String id, Double score) {
		return Document.builder().id(id).text("text " + id).score(score).build();
	}

	@Test
	void normalisesEachListBeforeWeighting() {
		// Vector scores 0.80..0.90 and full-text scores 0.01..0.05 both map to [0, 1].
		var vector = new RankedList("vector", 0.7, List.of(doc("a", 0.90), doc("b", 0.85), doc("c", 0.80)));
		var fulltext = new RankedList("fulltext", 0.3, List.of(doc("c", 0.05), doc("b", 0.01)));

		List<Document> fused = RankFusion.weightedScore().fuse(List.of(vector, fulltext));

		assertThat(fused).extracting(Document::getId).containsExactly("a", "b", "c");
		assertThat(fused.get(0).getScore()).isCloseTo(0.7, within(1e-9));
		assertThat(fused.get(1).getScore()).isCloseTo(0.35, within(1e-9));
		assertThat(fused.get(2).getScore()).isCloseTo(0.3, within(1e-9));
	}

	@Test
	void fallsBackToRankWhenScoresAreMissingOrEqual() {
		var noScores = RankedList.of("x", List.of(doc("a", null), doc("b", null), doc("c", null)));

		List<Document> fused = RankFusion.weightedScore().fuse(List.of(noScores));

		assertThat(fused).extracting(Document::getId).containsExactly("a", "b", "c");
		assertThat(fused).extracting(Document::getScore).containsExactly(1.0, 0.5, 0.0);
	}

	@Test
	void singleDocumentListScoresOne() {
		List<Document> fused = RankFusion.weightedScore().fuse(List.of(RankedList.of("x", List.of(doc("a", 0.3)))));

		assertThat(fused.get(0).getScore()).isEqualTo(1.0);
	}

}
