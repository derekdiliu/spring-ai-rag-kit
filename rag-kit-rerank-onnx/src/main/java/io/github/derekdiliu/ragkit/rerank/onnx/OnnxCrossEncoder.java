package io.github.derekdiliu.ragkit.rerank.onnx;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.util.PairList;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.derekdiliu.ragkit.rerank.RelevanceScorer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.springframework.util.Assert;

/**
 * A {@link RelevanceScorer} that runs a cross-encoder locally with ONNX Runtime: the query
 * and each text are read together, which ranks far better than comparing embeddings, at
 * the cost of one model pass per text.
 * <p>
 * Defaults to {@value #DEFAULT_MODEL} (Apache 2.0, ~90 MB), downloaded once from Hugging
 * Face at a pinned revision, checked against its SHA-256 and cached under
 * {@code ~/.cache/ragkit/models}. Other BERT-style cross-encoders exported to ONNX can be
 * loaded with {@link Builder#modelFiles(Path, Path)}.
 * <p>
 * Scores are the model's logit through a sigmoid, in {@code (0, 1)}. Instances are
 * thread-safe; {@link #close()} releases the native session.
 */
public final class OnnxCrossEncoder implements RelevanceScorer, AutoCloseable {

	public static final String DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2";

	private static final String DEFAULT_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a";

	private static final String MODEL_SHA256 = "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a";

	private static final String TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66";

	private static final Log logger = LogFactory.getLog(OnnxCrossEncoder.class);

	private final OrtEnvironment environment;

	private final OrtSession session;

	private final HuggingFaceTokenizer tokenizer;

	private final int batchSize;

	private OnnxCrossEncoder(Builder builder) throws IOException, OrtException {
		Path model = builder.modelPath;
		Path tokenizerFile = builder.tokenizerPath;
		if (model == null) {
			Path dir = builder.cacheDirectory.resolve(DEFAULT_MODEL.replace('/', '_') + "-" + DEFAULT_REVISION);
			model = download("onnx/model.onnx", dir.resolve("model.onnx"), MODEL_SHA256);
			tokenizerFile = download("tokenizer.json", dir.resolve("tokenizer.json"), TOKENIZER_SHA256);
		}
		// Pairs longer than maxLength lose tokens from the longer side first, as in
		// sentence-transformers; each batch is padded to its longest pair.
		this.tokenizer = HuggingFaceTokenizer.newInstance(tokenizerFile,
				Map.of("truncation", "true", "padding", "true", "maxLength", String.valueOf(builder.maxLength)));
		this.environment = OrtEnvironment.getEnvironment();
		this.session = this.environment.createSession(model.toString(), new OrtSession.SessionOptions());
		this.batchSize = builder.batchSize;
	}

	/** The default model with default settings. */
	public static OnnxCrossEncoder create() {
		return builder().build();
	}

	public static Builder builder() {
		return new Builder();
	}

	@Override
	public List<Double> score(String query, List<String> texts) {
		Assert.notNull(query, "query must not be null");
		Assert.notNull(texts, "texts must not be null");
		List<Double> scores = new ArrayList<>(texts.size());
		for (int from = 0; from < texts.size(); from += this.batchSize) {
			scores.addAll(scoreBatch(query, texts.subList(from, Math.min(from + this.batchSize, texts.size()))));
		}
		return scores;
	}

	private List<Double> scoreBatch(String query, List<String> texts) {
		PairList<String, String> pairs = new PairList<>(texts.size());
		texts.forEach(text -> pairs.add(query, text));
		Encoding[] encodings = this.tokenizer.batchEncode(pairs);
		long[][] ids = new long[encodings.length][];
		long[][] attentionMask = new long[encodings.length][];
		long[][] typeIds = new long[encodings.length][];
		for (int i = 0; i < encodings.length; i++) {
			ids[i] = encodings[i].getIds();
			attentionMask[i] = encodings[i].getAttentionMask();
			typeIds[i] = encodings[i].getTypeIds();
		}
		Map<String, OnnxTensor> inputs = new HashMap<>();
		try {
			inputs.put("input_ids", OnnxTensor.createTensor(this.environment, ids));
			inputs.put("attention_mask", OnnxTensor.createTensor(this.environment, attentionMask));
			// Not every cross-encoder takes segment ids (e.g. RoBERTa-based ones do not).
			if (this.session.getInputNames().contains("token_type_ids")) {
				inputs.put("token_type_ids", OnnxTensor.createTensor(this.environment, typeIds));
			}
			try (OrtSession.Result result = this.session.run(inputs)) {
				float[][] logits = (float[][]) result.get(0).getValue();
				List<Double> scores = new ArrayList<>(logits.length);
				for (float[] logit : logits) {
					scores.add(1.0 / (1.0 + Math.exp(-logit[0])));
				}
				return scores;
			}
		}
		catch (OrtException ex) {
			throw new IllegalStateException("Cross-encoder inference failed", ex);
		}
		finally {
			inputs.values().forEach(OnnxTensor::close);
		}
	}

	@Override
	public void close() throws OrtException {
		this.session.close();
		this.tokenizer.close();
	}

	private static Path download(String file, Path target, String sha256) throws IOException {
		if (Files.exists(target)) {
			return target;
		}
		Files.createDirectories(target.getParent());
		URI uri = URI.create("https://huggingface.co/" + DEFAULT_MODEL + "/resolve/" + DEFAULT_REVISION + "/" + file);
		logger.info("Downloading " + uri + " into " + target.getParent());
		Path partial = Files.createTempFile(target.getParent(), target.getFileName().toString(), ".part");
		try {
			HttpResponse<Path> response = HttpClient.newBuilder()
				.followRedirects(HttpClient.Redirect.NORMAL)
				.build()
				.send(HttpRequest.newBuilder(uri).build(), HttpResponse.BodyHandlers.ofFile(partial));
			if (response.statusCode() != 200) {
				throw new IOException("Downloading " + uri + " failed with HTTP " + response.statusCode());
			}
			String actual = sha256(partial);
			if (!actual.equals(sha256)) {
				throw new IOException(uri + " has SHA-256 " + actual + ", expected " + sha256);
			}
			return Files.move(partial, target, StandardCopyOption.ATOMIC_MOVE);
		}
		catch (InterruptedException ex) {
			Thread.currentThread().interrupt();
			throw new IOException("Interrupted while downloading " + uri, ex);
		}
		finally {
			Files.deleteIfExists(partial);
		}
	}

	private static String sha256(Path file) throws IOException {
		try (var in = new DigestInputStream(Files.newInputStream(file), MessageDigest.getInstance("SHA-256"))) {
			in.transferTo(OutputStream.nullOutputStream());
			return HexFormat.of().formatHex(in.getMessageDigest().digest());
		}
		catch (NoSuchAlgorithmException ex) {
			throw new IllegalStateException(ex);
		}
	}

	public static final class Builder {

		public static final int DEFAULT_MAX_LENGTH = 512;

		public static final int DEFAULT_BATCH_SIZE = 32;

		private Path modelPath;

		private Path tokenizerPath;

		private Path cacheDirectory = Path.of(System.getProperty("user.home"), ".cache", "ragkit", "models");

		private int maxLength = DEFAULT_MAX_LENGTH;

		private int batchSize = DEFAULT_BATCH_SIZE;

		private Builder() {
		}

		/** Uses a local cross-encoder instead of downloading {@value #DEFAULT_MODEL}. */
		public Builder modelFiles(Path model, Path tokenizer) {
			Assert.notNull(model, "model must not be null");
			Assert.notNull(tokenizer, "tokenizer must not be null");
			this.modelPath = model;
			this.tokenizerPath = tokenizer;
			return this;
		}

		/** Where the default model is cached. Defaults to {@code ~/.cache/ragkit/models}. */
		public Builder cacheDirectory(Path cacheDirectory) {
			Assert.notNull(cacheDirectory, "cacheDirectory must not be null");
			this.cacheDirectory = cacheDirectory;
			return this;
		}

		/** Maximum tokens per query-text pair. Defaults to 512, the model's limit. */
		public Builder maxLength(int maxLength) {
			Assert.isTrue(maxLength > 0, "maxLength must be positive");
			this.maxLength = maxLength;
			return this;
		}

		/** Pairs per model pass. Defaults to 32. */
		public Builder batchSize(int batchSize) {
			Assert.isTrue(batchSize > 0, "batchSize must be positive");
			this.batchSize = batchSize;
			return this;
		}

		public OnnxCrossEncoder build() {
			try {
				return new OnnxCrossEncoder(this);
			}
			catch (IOException ex) {
				throw new UncheckedIOException(ex);
			}
			catch (OrtException ex) {
				throw new IllegalStateException("Cannot load the cross-encoder model", ex);
			}
		}

	}

}
