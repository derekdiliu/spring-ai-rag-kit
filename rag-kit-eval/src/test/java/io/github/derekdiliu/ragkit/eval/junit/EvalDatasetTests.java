package io.github.derekdiliu.ragkit.eval.junit;

import java.util.List;
import java.util.Map;

import io.github.derekdiliu.ragkit.eval.EvalCase;
import io.github.derekdiliu.ragkit.eval.EvalDatasets;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class EvalDatasetTests {

	@Test
	void injectsCasesWithoutAnExplicitExtension(@EvalDataset("classpath:eval/sample.jsonl") List<EvalCase> cases) {
		assertThat(cases).containsExactly(new EvalCase("q1", "How do I reset my password?", Map.of("doc-12", 1, "doc-40", 1)),
				new EvalCase("q2", "Refund after 30 days", Map.of("doc-7", 2, "doc-9", 1)));
	}

	@Test
	void reportsTheLineOfAnInvalidCase() {
		assertThatIllegalArgumentException().isThrownBy(() -> EvalDatasets.load("classpath:eval/invalid.jsonl"))
			.withMessageStartingWith("classpath:eval/invalid.jsonl line 3: ")
			.withMessageContaining("needs at least one relevant document");
	}

	@Test
	void rejectsAMissingFile() {
		assertThatThrownBy(() -> EvalDatasets.load("classpath:eval/missing.jsonl"))
			.hasMessageContaining("classpath:eval/missing.jsonl");
	}

}
