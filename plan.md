# Soup — Roadmap

**Repo:** https://github.com/MakazhanAlpamys/Soup
**PyPI:** https://pypi.org/project/soup-cli/ (`pip install soup-cli`)
**Version:** v0.10.0 | 611 tests | CI green

### How to publish

```bash
# 1. Bump version in pyproject.toml + soup_cli/__init__.py
# 2. Tag and push
git tag v0.X.0
git push --tags
# GitHub Actions auto-publishes to PyPI
```

---

## Completed (v0.1.0 – v0.10.0)

All core CLI functionality is shipped:

- **CLI:** init, train, chat, push, merge, export, eval, serve, sweep, diff, doctor, quickstart, ui, version
- **Data:** inspect, validate, convert, merge, dedup, stats, generate
- **Training:** SFT, DPO, GRPO, PPO/RLHF, Reward Model, LoRA/QLoRA, QAT, auto batch size, resume, DeepSpeed, W&B, Unsloth backend
- **Multimodal:** `modality: vision`, LLaVA/ShareGPT4V, LLaMA-Vision/Qwen2-VL/Pixtral
- **Serving:** OpenAI-compatible API, transformers + vLLM backends, SSE streaming, tensor parallelism
- **Tracking:** SQLite experiment tracker, runs list/show/compare/delete
- **Export:** GGUF (Ollama/llama.cpp), LoRA merge
- **Web UI:** Dashboard, New Training, Data Explorer, Model Chat (FastAPI + SPA)
- **UX:** friendly errors, --verbose, confirmation prompts, Rich progress bars
- **Community:** CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, examples/, FUNDING.yml, issue/PR templates, GitHub Discussions
- **Tests:** 611 tests, 40 files, ruff lint, CI on Python 3.9/3.11/3.12

---

## Next

- [ ] First post on Reddit (r/LocalLLaMA, r/MachineLearning)
- [ ] Community building, content, marketing

---

## Future (when there's demand)

- **Cloud Mode (BYOG)** — Modal/RunPod/Vast integration, cost estimator
- **Managed Platform (SaaS)** — app.soup.dev, monetization (when 300+ stars)

---

## Principles

1. **CLI-first** — everything works from the terminal; UI is a bonus
2. **Zero config by default** — `soup train` with a minimal config just works
3. **Fail fast, fail loud** — bad data or missing GPU = immediate, clear error
4. **Open source core** — CLI is always free; monetize via managed service
5. **Test-driven** — every feature has tests, written alongside the code
