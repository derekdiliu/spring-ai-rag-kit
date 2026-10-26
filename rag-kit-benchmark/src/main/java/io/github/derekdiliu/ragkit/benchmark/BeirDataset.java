package io.github.derekdiliu.ragkit.benchmark;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.zip.ZipFile;

import io.github.derekdiliu.ragkit.eval.EvalCase;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

import org.springframework.ai.document.Document;

/**
 * A BEIR dataset (Thakur et al., NeurIPS 2021): the corpus as Spring AI documents and the
 * test split as evaluation cases. The zip is downloaded once into a cache directory.
 *
 * @param name the BEIR name, e.g. {@code scifact}
 * @param corpus one document per corpus entry, text = title + body
 * @param testCases test queries with their graded relevant document ids
 */
record BeirDataset(String name, List<Document> corpus, List<EvalCase> testCases) {

	/** Metadata key holding the BEIR document id; PgVectorStore needs UUID ids of its own. */
	static final String ID_KEY = "beir_id";

	private static final String BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/";

	private static final JsonMapper JSON = JsonMapper.builder().build();

	static BeirDataset load(String name, Path cacheDir) throws IOException, InterruptedException {
		Path zip = download(name, cacheDir);
		try (ZipFile zipFile = new ZipFile(zip.toFile())) {
			List<Document> corpus = new ArrayList<>();
			for (String line : lines(zipFile, name + "/corpus.jsonl")) {
				JsonNode node = JSON.readTree(line);
				String id = node.path("_id").asString();
				String text = (node.path("title").asString("") + "\n" + node.path("text").asString("")).strip();
				if (!text.isEmpty()) {
					corpus.add(Document.builder()
						.id(UUID.nameUUIDFromBytes(id.getBytes(StandardCharsets.UTF_8)).toString())
						.text(text)
						.metadata(ID_KEY, id)
						.build());
				}
			}

			Map<String, String> queries = new HashMap<>();
			for (String line : lines(zipFile, name + "/queries.jsonl")) {
				JsonNode node = JSON.readTree(line);
				queries.put(node.path("_id").asString(), node.path("text").asString());
			}

			// query-id <TAB> corpus-id <TAB> score, after a header line
			Map<String, Map<String, Integer>> qrels = new LinkedHashMap<>();
			List<String> rows = lines(zipFile, name + "/qrels/test.tsv");
			for (String row : rows.subList(1, rows.size())) {
				String[] fields = row.split("\t");
				int grade = Integer.parseInt(fields[2].trim());
				if (grade > 0) {
					qrels.computeIfAbsent(fields[0], q -> new LinkedHashMap<>()).put(fields[1], grade);
				}
			}

			List<EvalCase> testCases = qrels.entrySet()
				.stream()
				.map(entry -> new EvalCase(entry.getKey(), queries.get(entry.getKey()), entry.getValue()))
				.toList();
			return new BeirDataset(name, List.copyOf(corpus), testCases);
		}
	}

	private static Path download(String name, Path cacheDir) throws IOException, InterruptedException {
		Path zip = cacheDir.resolve(name + ".zip");
		if (Files.exists(zip)) {
			return zip;
		}
		Files.createDirectories(cacheDir);
		System.out.printf("Downloading BEIR %s into %s%n", name, cacheDir);
		Path partial = Files.createTempFile(cacheDir, name, ".part");
		HttpResponse<Path> response = HttpClient.newBuilder()
			.followRedirects(HttpClient.Redirect.NORMAL)
			.build()
			.send(HttpRequest.newBuilder(URI.create(BASE_URL + name + ".zip")).build(),
					HttpResponse.BodyHandlers.ofFile(partial));
		if (response.statusCode() != 200) {
			Files.deleteIfExists(partial);
			throw new IOException("Downloading " + name + " failed with HTTP " + response.statusCode());
		}
		return Files.move(partial, zip, StandardCopyOption.ATOMIC_MOVE);
	}

	private static List<String> lines(ZipFile zipFile, String entryName) throws IOException {
		var entry = zipFile.getEntry(entryName);
		if (entry == null) {
			throw new IOException(zipFile.getName() + " has no " + entryName);
		}
		try (var reader = new BufferedReader(
				new InputStreamReader(zipFile.getInputStream(entry), StandardCharsets.UTF_8))) {
			return reader.lines().filter(line -> !line.isBlank()).toList();
		}
	}

}
