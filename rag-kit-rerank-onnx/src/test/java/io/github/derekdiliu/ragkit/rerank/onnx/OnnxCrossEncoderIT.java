package io.github.derekdiliu.ragkit.rerank.onnx;

import java.util.List;

import io.github.derekdiliu.ragkit.rerank.RerankingDocumentPostProcessor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Runs the default model, downloaded on first use (~90 MB) and cached.
 */
class OnnxCrossEncoderIT {

	static OnnxCrossEncoder crossEncoder;

	@BeforeAll
	static void load() {
		crossEncoder = OnnxCrossEncoder.create();
	}

	@AfterAll
	static void close() throws Exception {
		crossEncoder.close();
	}

	private static final String QUERY = "How many people live in Berlin?";

	private static final List<String> PASSAGES = List.of(
			"Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
			"New York City is famous for the Metropolitan Museum of Art.",
			"Berlin is well known for its museums.",
			"The capital of Germany has about three and a half million residents.",
			"Pasta is boiled in salted water.");

	@Test
	void scoresTheAnsweringPassageAboveUnrelatedOnes() {
		List<Double> scores = crossEncoder.score(QUERY, PASSAGES);

		assertThat(scores).hasSize(5).allSatisfy(score -> assertThat(score).isBetween(0.0, 1.0));
		assertThat(scores.get(0)).isGreaterThan(0.9);
		assertThat(scores.get(0)).isGreaterThan(scores.get(2)).isGreaterThan(scores.get(1));
		assertThat(scores.get(4)).isLessThan(0.01);
	}

	@Test
	void batchingDoesNotChangeScores() {
		try (var smallBatches = OnnxCrossEncoder.builder().batchSize(2).build()) {
			List<Double> batched = smallBatches.score(QUERY, PASSAGES);
			List<Double> whole = crossEncoder.score(QUERY, PASSAGES);

			for (int i = 0; i < PASSAGES.size(); i++) {
				assertThat(batched.get(i)).isCloseTo(whole.get(i), within(1e-4));
			}
		}
		catch (Exception ex) {
			throw new IllegalStateException(ex);
		}
	}

	@Test
	void truncatesLongTexts() {
		String longText = "Berlin has about 3.5 million inhabitants. " + "Unrelated filler words. ".repeat(2000);

		assertThat(crossEncoder.score(QUERY, List.of(longText)).get(0)).isGreaterThan(0.5);
	}

	@Test
	void emptyInputGivesNoScores() {
		assertThat(crossEncoder.score(QUERY, List.of())).isEmpty();
	}

	@Test
	void reranksRetrievedDocuments() {
		List<Document> retrieved = PASSAGES.stream().map(text -> Document.builder().text(text).build()).toList();
		var reranker = RerankingDocumentPostProcessor.builder(crossEncoder).topN(2).build();

		List<Document> result = reranker.process(new Query(QUERY), retrieved);

		assertThat(result).extracting(Document::getText).containsExactly(PASSAGES.get(0), PASSAGES.get(3));
	}

}
