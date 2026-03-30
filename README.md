# opencode-agent-skills (HF-Free)

> Fork of [joshuadavidthomas/opencode-agent-skills](https://github.com/joshuadavidthomas/opencode-agent-skills) with Hugging Face dependency removed for use in China.

**What changed:** The embedding model is downloaded from [hf-mirror.com](https://hf-mirror.com) instead of huggingface.co. After first-run download, the plugin works **fully offline**.

---

## AI Agent Auto-Install Instructions

### Step 1: Clone and Link

```bash
git clone https://github.com/xiaowugui/opencode-agent-skills.git ~/.config/opencode/opencode-agent-skills
mkdir -p ~/.config/opencode/plugin
ln -sf ~/.config/opencode/opencode-agent-skills/src/plugin.ts ~/.config/opencode/plugin/skills.ts
```

### Step 2: Verify Tests Pass

```bash
cd ~/.config/opencode/opencode-agent-skills
bun install
bun test        # All 25 tests pass — also warms the model cache
```

### Step 3: Restart OpenCode

After restarting, the plugin auto-discovers skills from:

1. `.opencode/skills/` (project)
2. `.claude/skills/` (project)
3. `~/.config/opencode/skills/` (user)
4. `~/.claude/skills/` (user)

---

## First-Run Model Download

On first use, the plugin automatically downloads the embedding model from `hf-mirror.com`:

- **Model:** `Xenova/all-MiniLM-L6-v2` (ONNX, ~22 MB)
- **Cache:** `~/.cache/opencode-agent-skills/models/Xenova--all-MiniLM-L6-v2/`
- **Subsequent runs:** Zero network — loads from local cache

To pre-warm the cache without triggering skill matching:

```bash
cd ~/.config/opencode/opencode-agent-skills
bun test         # runs embeddings tests → downloads + caches model
```

---

## Features

| Tool                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `use_skill`            | Load a skill's SKILL.md into context             |
| `read_skill_file`      | Read supporting files from a skill directory     |
| `run_skill_script`     | Execute scripts from a skill directory           |
| `get_available_skills` | Get available skills                             |
| **Auto-matching**      | Semantically matches messages to relevant skills |

---

## Model Details

| Item            | Value                                                       |
| --------------- | ----------------------------------------------------------- |
| Base model      | sentence-transformers/all-MiniLM-L6-v2                      |
| ONNX version    | Xenova/all-MiniLM-L6-v2                                     |
| Embedding dim   | 384                                                         |
| Download source | https://hf-mirror.com/Xenova/all-MiniLM-L6-v2/resolve/main/ |
| File loaded     | onnx/model_int8.onnx (~22 MB)                               |

---

## Key Files

| File                 | Purpose                                      |
| -------------------- | -------------------------------------------- |
| `src/embeddings.ts`  | Local embedding model via hf-mirror download |
| `src/skills.ts`      | Skill discovery and management               |
| `src/plugin.ts`      | OpenCode plugin entry point                  |
| `src/superpowers.ts` | Optional Superpowers workflow integration    |

---

## Why This Fork

The original plugin connects to `huggingface.co` and `api-inference.huggingface.co` at runtime — **both are inaccessible in China**. This fork:

1. Downloads model files from `hf-mirror.com` (China-accessible) on first run
2. Loads model from local filesystem — zero HuggingFace network dependency after install
3. Uses `dtype: "int8"` since `model_q8.onnx` is not available on hf-mirror

---

## Development

```bash
bun install          # Install dependencies
bun test             # Run all tests (25 tests)
bun run typecheck    # TypeScript check
```

---

## License

MIT — same as upstream.
