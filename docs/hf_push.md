# Pushing `llm_forge-25m` to HuggingFace Hub

The repo ships an HF-compatible directory at
`experiments/checkpoints/llm_forge-25m/` (config.json, params.safetensors,
tokenizer.json, README.md with frontmatter, etc.). Upload it with
the official `huggingface-cli` — no custom Python needed.

## One-time setup

```bash
# 1. Create a write token at https://huggingface.co/settings/tokens
# 2. Either save it via the CLI:
huggingface-cli login
# ...or set it inline:
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

## Create the repo (public)

```bash
huggingface-cli repo create adesh01/llm_forge-25m --type model --public
```

## Upload

```bash
cd experiments/checkpoints/llm_forge-25m
huggingface-cli upload adesh01/llm_forge-25m . . --commit-message="Initial 1B-token pretrain"
```

The `. .` syntax means "current directory → repo root". `huggingface-cli`
diffs against the previous commit and only re-uploads changed files, so
a 1KB README tweak costs ~1KB, not 111MB.

## Verify

```bash
huggingface-cli repo info adesh01/llm_forge-25m
# or open https://huggingface.co/adesh01/llm_forge-25m
```

## Updating a single file

```bash
# Re-push just the README (e.g. after a local edit)
huggingface-cli upload adesh01/llm_forge-25m README.md README.md \
    --commit-message="Update README"
```

## Re-pushing `params.safetensors` only

```bash
huggingface-cli upload adesh01/llm_forge-25m params.safetensors params.safetensors \
    --commit-message="Update weights"
```
