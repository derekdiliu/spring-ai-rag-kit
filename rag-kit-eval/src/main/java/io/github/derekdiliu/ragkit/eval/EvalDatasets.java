package io.github.derekdiliu.ragkit.eval;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;

/**
 * Loads evaluation cases from JSON Lines, one case per line:
 *
 * <pre>
 * {"id": "q1", "query": "How do I reset my password?", "relevant": ["doc-12", "doc-40"]}
 * {"id": "q2", "query": "Refund after 30 days", "relevant": {"doc-7": 2, "doc-9": 1}}
 * </pre>
 *
 * {@code relevant} is either a list of ids (grade 1 each) or an object of graded ids.
 */
public final class EvalDatasets {

	private static final JsonMapper JSON = JsonMapper.builder().build();

	private EvalDatasets() {
	}

	/**
	 * Loads a dataset from a Spring resource location, e.g.
	 * {@code classpath:eval/questions.jsonl} or {@code file:eval/questions.jsonl}.
	 */
	public static List<EvalCase> load(String location) {
		Resource resource = new DefaultResourceLoader().getResource(location);
		List<EvalCase> cases = new ArrayList<>();
		try (BufferedReader reader = new BufferedReader(
				new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
			String line;
			int lineNumber = 0;
			while ((line = reader.readLine()) != null) {
				lineNumber++;
				if (!line.isBlank()) {
					cases.add(parse(line, location, lineNumber));
				}
			}
		}
		catch (IOException ex) {
			throw new UncheckedIOException("Cannot read evaluation dataset " + location, ex);
		}
		if (cases.isEmpty()) {
			throw new IllegalArgumentException("Evaluation dataset " + location + " has no cases");
		}
		return List.copyOf(cases);
	}

	private static EvalCase parse(String line, String location, int lineNumber) {
		try {
			JsonNode node = JSON.readTree(line);
			JsonNode relevant = node.path("relevant");
			Map<String, Integer> relevance = new LinkedHashMap<>();
			if (relevant.isArray()) {
				relevant.forEach(id -> relevance.put(id.asString(), 1));
			}
			else if (relevant.isObject()) {
				relevant.properties().forEach(entry -> relevance.put(entry.getKey(), entry.getValue().asInt()));
			}
			return new EvalCase(node.path("id").asString(""), node.path("query").asString(""), relevance);
		}
		catch (JacksonException | IllegalArgumentException ex) {
			throw new IllegalArgumentException(location + " line " + lineNumber + ": " + ex.getMessage(), ex);
		}
	}

}
