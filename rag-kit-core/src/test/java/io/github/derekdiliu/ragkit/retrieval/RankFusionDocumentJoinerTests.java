package io.github.derekdiliu.ragkit.retrieval;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.derekdiliu.ragkit.fusion.RankFusion;
import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

import static org.assertj.core.api.Assertions.assertThat;

class RankFusionDocumentJoinerTests {

	private static Document doc(String id, double score) {
		return Document.builder().id(id).text("text " + id).score(score).build();
	}

	@Test
	void rewardsDocumentsReturnedForSeveralExpandedQueries() {
		// "shared" never has the top raw score but appears for both expanded queries;
		// concatenation sorted by raw score would put it last.
		Map<Query, List<List<Document>>> input = new LinkedHashMap<>();
		input.put(new Query("how do I log in"), List.of(List.of(doc("login", 0.95), doc("shared", 0.70))));
		input.put(new Query("authentication flow"), List.of(List.of(doc("auth", 0.93), doc("shared", 0.71))));

		List<Document> joined = new RankFusionDocumentJoiner().join(input);

		assertThat(joined).extracting(Document::getId).containsExactly("shared", "login", "auth");
	}

	@Test
	void appliesTopK() {
		Map<Query, List<List<Document>>> input = Map.of(new Query("q"),
				List.of(List.of(doc("a", 0.9), doc("b", 0.8), doc("c", 0.7))));

		assertThat(new RankFusionDocumentJoiner(RankFusion.rrf(), 2).join(input)).extracting(Document::getId)
			.containsExactly("a", "b");
	}

}
