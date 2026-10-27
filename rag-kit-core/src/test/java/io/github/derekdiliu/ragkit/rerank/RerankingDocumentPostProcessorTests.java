package io.github.derekdiliu.ragkit.rerank;

import java.util.List;
import java.util.Map;

import io.github.derekdiliu.ragkit.fusion.FusionMetadata;
import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;

class RerankingDocumentPostProcessorTests {

	private static Document doc(String id, String text) {
		return Document.builder().id(id).text(text).score(0.5).build();
	}

	/** Scores a text by how many times it contains the query. */
	private static final RelevanceScorer COUNTING = (query, texts) -> texts.stream()
		.map(text -> (double) text.split(query, -1).length - 1)
		.toList();

	private final List<Document> retrieved = List.of(doc("a", "recipe"), doc("b", "nothing here"),
			doc("c", "recipe recipe recipe"), doc("d", "recipe recipe"));

	@Test
	void ordersByScorerAndKeepsTopN() {
		var reranker = RerankingDocumentPostProcessor.builder(COUNTING).topN(2).build();

		List<Document> result = reranker.process(new Query("recipe"), this.retrieved);

		assertThat(result).extracting(Document::getId).containsExactly("c", "d");
		assertThat(result).extracting(Document::getScore).containsExactly(3.0, 2.0);
		assertThat(result.get(0).getMetadata()).containsEntry(RerankingDocumentPostProcessor.RERANK_SCORE, 3.0);
	}

	@Test
	void keepsExistingMetadataAndRetrieverOrderOnTies() {
		var fused = Document.builder()
			.id("x")
			.text("same")
			.metadata(FusionMetadata.SOURCE_RANKS, Map.of("vector", 1))
			.build();
		var reranker = RerankingDocumentPostProcessor.builder((query, texts) -> texts.stream().map(t -> 1.0).toList())
			.topN(10)
			.build();

		List<Document> result = reranker.process(new Query("q"), List.of(fused, doc("y", "same"), doc("z", "same")));

		assertThat(result).extracting(Document::getId).containsExactly("x", "y", "z");
		assertThat(result.get(0).getMetadata()).containsEntry(FusionMetadata.SOURCE_RANKS, Map.of("vector", 1));
	}

	@Test
	void sendsTheQueryAndDocumentTextsToTheScorer() {
		var reranker = RerankingDocumentPostProcessor.builder((query, texts) -> {
			assertThat(query).isEqualTo("recipe");
			assertThat(texts).containsExactly("recipe", "nothing here", "recipe recipe recipe", "recipe recipe");
			return List.of(0.0, 0.0, 0.0, 0.0);
		}).build();

		reranker.process(new Query("recipe"), this.retrieved);
	}

	@Test
	void returnsEmptyWithoutCallingTheScorer() {
		var reranker = RerankingDocumentPostProcessor.builder((query, texts) -> {
			throw new AssertionError("scorer called");
		}).build();

		assertThat(reranker.process(new Query("q"), List.of())).isEmpty();
	}

	@Test
	void rejectsAScorerReturningTheWrongNumberOfScores() {
		var reranker = RerankingDocumentPostProcessor.builder((query, texts) -> List.of(1.0)).build();

		assertThatIllegalStateException().isThrownBy(() -> reranker.process(new Query("q"), this.retrieved))
			.withMessageContaining("1 scores for 4 documents");
	}

	@Test
	void defaultsToFiveAndRejectsNonPositiveTopN() {
		assertThat(RerankingDocumentPostProcessor.builder(COUNTING).build().getTopN()).isEqualTo(5);
		assertThatIllegalArgumentException().isThrownBy(() -> RerankingDocumentPostProcessor.builder(COUNTING).topN(0));
	}

}
