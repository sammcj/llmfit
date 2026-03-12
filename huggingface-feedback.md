# Feedback for HuggingFace: How to help LLMfit (and the local-LLM ecosystem) succeed

## Context

LLMfit is an open-source tool that helps users find the right open-weight LLM for their hardware. We scrape HuggingFace's API to build a catalog of ~536 models with metadata like parameter counts, context windows, memory requirements, capabilities, and MoE configurations. The HF Hub API is our primary data source, and we deeply appreciate the platform. Below is concrete, prioritized feedback on where API gaps force us into brittle workarounds.

---

## 1. Structured model metadata is the #1 gap

**Problem:** The HF Hub API returns repo-level metadata (downloads, likes, tags, safetensors info) but almost no *model-level technical metadata*. We have to reverse-engineer critical fields by fetching `config.json` as a raw file and parsing it ourselves.

**What we need that doesn't exist as first-class API fields:**

| Field | Current workaround | Pain level |
|---|---|---|
| **Parameter count** | Parse `safetensors.parameters` dict, take `max(values)`. Fails for gated models without a token. | High |
| **Context window** | Fetch raw `config.json`, check 5+ keys (`max_position_embeddings`, `max_sequence_length`, `seq_length`, `n_positions`, `sliding_window`), then check `text_config` sub-dict for multimodal models. Falls back to 4096 for 77 models (~14%). | High |
| **MoE configuration** (num_experts, active_experts, active_params) | We maintain a hardcoded lookup table of 11 architecture patterns and 26 model-specific active parameter counts. For other MoE models we use a rough formula (5% shared, rest divided by experts). | High |
| **Capabilities** (vision, tool_use, function_calling, code, reasoning) | We regex-match on repo names and org names (`"qwen3" in rid`, `"vision" in rid`, etc.). 64% of our models have empty or missing capability arrays. | High |
| **License** | Available in model cards but not easily machine-readable in a standardized way | Medium |
| **Knowledge cutoff date** | Not available at all; we don't track it | Medium |

**Ask:** A `/models/{id}` response that includes standardized, structured fields like:

```json
{
  "technical_metadata": {
    "total_parameters": 70000000000,
    "active_parameters": 17000000000,
    "context_window": 131072,
    "is_moe": true,
    "num_experts": 16,
    "active_experts": 1,
    "capabilities": ["text-generation", "tool-use", "vision"],
    "quantization_formats_available": ["BF16", "Q4_K_M", "Q8_0"],
    "license_spdx": "llama3.1"
  }
}
```

This single change would eliminate ~400 lines of inference/estimation logic from our scraper and dramatically improve accuracy for every downstream tool.

---

## 2. Gated models are a black box

**Problem:** For gated models (Llama, some DeepSeek, etc.), the API returns HTTP 401 without a token and even with a token, `safetensors` metadata is sometimes missing. This forces us to maintain a **40-model hardcoded fallback table** with manually researched parameter counts, memory estimates, and context windows. This data goes stale and is never auto-updated.

**Ask:**

- Return basic technical metadata (parameter count, architecture, context window) for gated models **without requiring access approval**. This is non-sensitive information that's already published in model cards and blog posts.
- Alternatively, provide a dedicated "model specs" endpoint that returns technical metadata for any model regardless of gating status.

---

## 3. No capability/feature discovery API

**Problem:** There's no way to programmatically determine what a model supports. Our `infer_capabilities()` function is a pile of string-matching heuristics:

```python
# This is what we actually do today:
if "qwen3" in rid or "qwen2.5" in rid:
    caps.append("tool_use")
if pipeline_tag == "image-text-to-text" or "vision" in rid or "-vl-" in rid:
    caps.append("vision")
```

This breaks every time a new model family is released with a naming convention we haven't anticipated.

**Ask:** Standardized, model-author-provided capability tags. Even just extending the existing `pipeline_tag` to a `pipeline_tags` array with values like `tool-use`, `vision`, `code-generation`, `reasoning`, `structured-output` would be transformative.

---

## 4. Context length is unreliable

**Problem:** We make a *separate HTTP request* to fetch `config.json` for every model because the API's `config` field is often incomplete. Even then, we check 5 different keys across 2 levels of nesting because there's no standard. 77 models (~14%) still fall back to a 4096 default, which is wrong for most modern models.

**Ask:** A single, authoritative `context_length` field in the API response, derived from whatever config key the model actually uses. Model authors could also override this in their model card metadata.

---

## 5. GGUF/quantization discovery is fragmented

**Problem:** We separately scrape for GGUF quantized versions by searching for repos from known quantization providers (unsloth, bartowski). Only 19.4% of our models have GGUF source information. There's no unified way to discover "what quantized versions exist for model X?"

**Ask:** A relationship/variant API — given a base model, return all known quantized/distilled/fine-tuned derivatives with their format and quantization level. Something like:

```
GET /models/{id}/variants?format=gguf
```

---

## 6. Inference Providers API is inference-only, no model catalog

**Problem:** The Inference Providers API is great for running models, but it provides zero model discovery or metadata. There's no way to ask "which models are available on the inference API, and what are their specs?" We have to cross-reference the Hub API with the inference API to figure this out.

**Ask:** A model catalog endpoint for inference providers:

```
GET /inference-providers/models
```

Returning which models are available, on which providers, with what capabilities and limits.

---

## Summary of impact

If HuggingFace addressed items 1-4 above, we could:

- **Delete ~500 lines** of brittle inference/estimation/fallback code
- **Eliminate the 40-model hardcoded fallback table** entirely
- **Go from 36% to ~100%** capability coverage
- **Go from 86% to ~100%** accurate context window data
- **Auto-discover MoE configurations** instead of maintaining a manual lookup table
- Ship more accurate recommendations to users, making the whole local-LLM ecosystem more accessible

The core message: **HuggingFace is the source of truth for open-weight models, but the API treats models as git repos rather than as ML artifacts with technical specifications.** Bridging that gap would benefit not just LLMfit, but every tool in the ecosystem that needs to reason about model capabilities and requirements.
