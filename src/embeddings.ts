import * as crypto from "node:crypto";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { homedir } from "node:os";
import {
  pipeline,
  type FeatureExtractionPipeline,
} from "@huggingface/transformers";
import type { SkillSummary } from "./skills";

const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";
const HF_MIRROR_URL =
  "https://hf-mirror.com/Xenova/all-MiniLM-L6-v2/resolve/main";
const SIMILARITY_THRESHOLD = 0.35;
const TOP_K = 5;

const MODEL_FILES = [
  "onnx/model.onnx",
  "onnx/model_int8.onnx",
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.txt",
  "special_tokens_map.json",
  "config.json",
] as const;

let model: FeatureExtractionPipeline | null = null;
let modelPromise: Promise<void> | null = null;

function getModelCacheDir(): string {
  const xdgCacheHome = process.env.XDG_CACHE_HOME;
  return xdgCacheHome
    ? path.join(
        xdgCacheHome,
        "opencode-agent-skills",
        "models",
        "Xenova--all-MiniLM-L6-v2",
      )
    : path.join(
        homedir(),
        ".cache",
        "opencode-agent-skills",
        "models",
        "Xenova--all-MiniLM-L6-v2",
      );
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function downloadFile(url: string, destPath: string): Promise<void> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `Failed to download ${url}: ${res.status} ${res.statusText}`,
    );
  }
  const arrayBuffer = await res.arrayBuffer();
  await fs.mkdir(path.dirname(destPath), { recursive: true });
  await fs.writeFile(destPath, Buffer.from(arrayBuffer));
}

async function ensureModelDownloaded(): Promise<string> {
  const cacheDir = getModelCacheDir();
  const onnxDir = path.join(cacheDir, "onnx");

  // Check if main ONNX file exists
  const mainModel = path.join(onnxDir, "model.onnx");
  if (await fileExists(mainModel)) {
    return cacheDir;
  }

  // Download all model files from hf-mirror
  console.log(
    "[embeddings] Model not found in cache, downloading from hf-mirror.com...",
  );
  for (const file of MODEL_FILES) {
    const filePath = path.join(cacheDir, file);
    if (await fileExists(filePath)) continue;
    const url = `${HF_MIRROR_URL}/${file}`;
    console.log(`[embeddings] Downloading ${file}...`);
    await downloadFile(url, filePath);
  }

  return cacheDir;
}

async function ensureModel(): Promise<void> {
  if (model) return;
  if (!modelPromise) {
    modelPromise = (async () => {
      const localPath = await ensureModelDownloaded();
      // Use int8 dtype since model_int8.onnx is available on hf-mirror
      // (model_q8.onnx does not exist on the mirror)
      model = await pipeline("feature-extraction", localPath, {
        dtype: "int8",
      });
    })();
  }
  await modelPromise;
}

function getCachePath(contentHash: string): string {
  const xdgCacheHome = process.env.XDG_CACHE_HOME;
  const baseDir = xdgCacheHome
    ? path.join(xdgCacheHome, "opencode-agent-skills")
    : path.join(homedir(), ".cache", "opencode-agent-skills");
  return path.join(baseDir, "embeddings", `${contentHash}.bin`);
}

/**
 * Generate an embedding for the given text.
 * Results are cached to disk based on content hash.
 */
export async function getEmbedding(text: string): Promise<Float32Array> {
  await ensureModel();
  if (!model) throw new Error("Model failed to load");

  const hash = crypto.createHash("sha256").update(text).digest("hex");
  const cachePath = getCachePath(hash);

  try {
    const buffer = await fs.readFile(cachePath);
    return new Float32Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
  } catch {
    // Generate new embedding
  }

  const result = await model(text, { pooling: "mean", normalize: true });

  const embedding =
    result.data instanceof Float32Array
      ? result.data
      : new Float32Array(Array.from(result.data as ArrayLike<number>));

  await fs.mkdir(path.dirname(cachePath), { recursive: true });
  await fs.writeFile(
    cachePath,
    Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength),
  );

  return embedding;
}

/**
 * Compute cosine similarity between two embedding vectors.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vectors must have the same length (got ${a.length} and ${b.length})`,
    );
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const valA = a[i]!;
    const valB = b[i]!;
    dotProduct += valA * valB;
    normA += valA * valA;
    normB += valB * valB;
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Precompute embeddings for all skills at plugin startup.
 * Embeddings are cached to disk, so this warms the cache.
 */
export async function precomputeSkillEmbeddings(
  skills: SkillSummary[],
): Promise<void> {
  await Promise.all(
    skills.map((skill) => getEmbedding(skill.description).catch(() => {})),
  );
}

/**
 * Match user message against available skills using semantic similarity.
 * Returns top matching skills above the threshold, sorted by score.
 */
export async function matchSkills(
  userMessage: string,
  availableSkills: SkillSummary[],
  topK: number = TOP_K,
  threshold: number = SIMILARITY_THRESHOLD,
): Promise<SkillSummary[]> {
  if (availableSkills.length === 0) {
    return [];
  }

  const queryEmbedding = await getEmbedding(userMessage);

  const scored = await Promise.all(
    availableSkills.map(async (skill) => ({
      skill,
      score: cosineSimilarity(
        queryEmbedding,
        await getEmbedding(skill.description),
      ),
    })),
  );

  return scored
    .filter((s) => s.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((s) => s.skill);
}
