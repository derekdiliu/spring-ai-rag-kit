package io.github.derekdiliu.ragkit.pgvector;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Map;
import java.util.zip.ZipInputStream;

import org.testcontainers.DockerClientFactory;
import org.testcontainers.images.builder.ImageFromDockerfile;
import org.testcontainers.postgresql.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

/**
 * PostgreSQL with pgvector and the pg_textsearch extension (BM25), preloaded: the
 * {@code pgvector/pgvector:pg17} image plus pg_textsearch's released Debian package,
 * pinned and checksummed. Docker caches the built image.
 */
public final class PgTextSearchContainers {

	public static final String PG_TEXTSEARCH_VERSION = "1.5.1";

	/** SHA-256 of the release zips, by Docker architecture. */
	private static final Map<String, String> SHA256 = Map.of("amd64",
			"ed4410e8b6879a8971cd87fd03853f182e47b85b199c9bfcd9b34f159caba9fb", "arm64",
			"e65f953e99f47c05dde88a031884cf36ea269bcb62377f554b76ddb4d13fdc3e");

	private PgTextSearchContainers() {
	}

	public static PostgreSQLContainer create() {
		String arch = dockerArchitecture();
		Path deb = debianPackage(arch);
		var image = new ImageFromDockerfile("ragkit/pgvector-textsearch:pg17-" + PG_TEXTSEARCH_VERSION + "-" + arch,
				false)
			.withFileFromPath("pg_textsearch.deb", deb)
			.withDockerfileFromBuilder(dockerfile -> dockerfile.from("pgvector/pgvector:pg17")
				.copy("pg_textsearch.deb", "/tmp/pg_textsearch.deb")
				.run("dpkg -i /tmp/pg_textsearch.deb && rm /tmp/pg_textsearch.deb")
				.build());
		return new PostgreSQLContainer(DockerImageName.parse(image.get()).asCompatibleSubstituteFor("postgres"))
			.withCommand("postgres", "-c", "fsync=off", "-c", "shared_preload_libraries=pg_textsearch");
	}

	private static String dockerArchitecture() {
		String arch = DockerClientFactory.instance().getInfo().getArchitecture();
		return switch (arch) {
			case "x86_64", "amd64" -> "amd64";
			case "aarch64", "arm64" -> "arm64";
			default -> throw new IllegalStateException("No pg_textsearch build for Docker architecture " + arch);
		};
	}

	/** Downloads the release zip once into target/, verifies it, and extracts the .deb. */
	private static Path debianPackage(String arch) {
		String zipName = "pg-textsearch-v" + PG_TEXTSEARCH_VERSION + "-pg17-" + arch + ".zip";
		Path dir = Path.of("target", "pg_textsearch");
		Path zip = dir.resolve(zipName);
		Path deb = dir.resolve(zipName.replace(".zip", ".deb"));
		try {
			Files.createDirectories(dir);
			if (Files.notExists(zip)) {
				Path partial = Files.createTempFile(dir, zipName, ".part");
				HttpResponse<Path> response = HttpClient.newBuilder()
					.followRedirects(HttpClient.Redirect.NORMAL)
					.build()
					.send(HttpRequest.newBuilder(URI.create("https://github.com/timescale/pg_textsearch/releases/download/v"
							+ PG_TEXTSEARCH_VERSION + "/" + zipName)).build(), HttpResponse.BodyHandlers.ofFile(partial));
				if (response.statusCode() != 200) {
					throw new IOException("Downloading " + zipName + " failed with HTTP " + response.statusCode());
				}
				Files.move(partial, zip, StandardCopyOption.ATOMIC_MOVE);
			}
			String actual = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(zip)));
			if (!actual.equals(SHA256.get(arch))) {
				Files.delete(zip);
				throw new IOException(zipName + " has SHA-256 " + actual + ", expected " + SHA256.get(arch));
			}
			try (InputStream in = Files.newInputStream(zip); var entries = new ZipInputStream(in)) {
				for (var entry = entries.getNextEntry(); entry != null; entry = entries.getNextEntry()) {
					if (entry.getName().endsWith(".deb")) {
						Files.copy(entries, deb, StandardCopyOption.REPLACE_EXISTING);
						return deb;
					}
				}
			}
			throw new IOException(zipName + " contains no .deb");
		}
		catch (IOException ex) {
			throw new UncheckedIOException(ex);
		}
		catch (InterruptedException ex) {
			Thread.currentThread().interrupt();
			throw new IllegalStateException(ex);
		}
		catch (NoSuchAlgorithmException ex) {
			throw new IllegalStateException(ex);
		}
	}

}
