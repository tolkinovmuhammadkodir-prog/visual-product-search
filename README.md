# Visual Product Search

A CLIP-based visual product search system with cross-encoder reranking and grounded, hallucination-resistant description generation — built and verified end-to-end on the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset.

## What this is

Given a text query like `"formal shirt"`, the system:

1. **Retrieves** candidate products with CLIP embeddings + FAISS exact similarity search (`IndexFlatIP`)
2. **Reranks** those candidates with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) using rich text (title + category + color + usage), not just the title
3. **Generates a grounded description** of the top result using Phi-3-mini — the model is given only verified product metadata and explicitly instructed not to add details that aren't in it, to avoid hallucinated claims
4. Serves all of this through a Gradio web app

## Why the reranking step exists

Plain CLIP similarity search alone does not reliably separate relevant results from near-misses. In testing on the query `"formal shirt"`, baseline FAISS retrieval ranked a correct "Formal" shirt at position #9 of 10, below several "Casual" shirts that matched on visual style but not intended use. Reranking with product metadata (not just the title) corrected this — the same item moved to position #1, and all "Casual" items were pushed to the bottom of the ranked list. This is a verified, measured result, not an assumption.

## Why FAISS `IndexFlatIP` (exact search) rather than an approximate index

At ~5,000 products, exact brute-force search is still fast and gives perfect accuracy — there's no reason to accept approximation error at this scale. At real e-commerce scale (millions of products), this would be swapped for an approximate method like HNSW or IVF specifically because linear search stops being fast enough. That's a deliberate scaling decision, not a default choice.

## Why the description generation is "grounded"

A language model asked to "describe this product" will happily invent confident, specific-sounding details (material, fit, features) that nobody verified. Here, Phi-3 receives only the structured product metadata (name, category, colour, gender, usage) and is instructed to use nothing else, so that every claim in the description should trace back to a field in the catalogue. This is enforced by the prompt, not guaranteed: a language model can still occasionally add unstated details. Decoding is greedy (`do_sample=False`), so the same product always gets the same description.

## Architecture

```
Text query
    │
    ▼
CLIP text encoder ──► FAISS similarity search ──► top-N candidates
    │
    ▼
Cross-encoder reranker (title + category + color + usage)
    │
    ▼
Reranked results ──► Gradio gallery
    │
    ▼ (optional, top result only)
Verified metadata ──► Phi-3-mini ──► grounded description
```

## Tech stack

- **CLIP** (ViT-B/32) — image/text embeddings
- **FAISS** — similarity search
- **sentence-transformers cross-encoder** (`ms-marco-MiniLM-L-6-v2`) — reranking
- **Phi-3-mini-4k-instruct** — grounded description generation
- **Gradio** — web interface

## Setup

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install faiss-cpu
pip install gradio
pip install -r requirements.txt
```

Requires a GPU runtime (e.g. Colab/Kaggle with a T4) for reasonable speed — CLIP and Phi-3 are both loaded at once.

## Running

```bash
python project3_master.py   # builds embeddings, FAISS index, loads all models, runs verification tests
python gradio_app.py        # launches the web app (run in the same session so models aren't reloaded)
```

`gradio_app.py` launches with `share=True`, producing a public URL you can open from any device.

## Files

- `project3_master.py` — dataset loading, CLIP embedding pipeline, FAISS index, grounded Phi-3 generation, cross-encoder reranking
- `gradio_app.py` — web interface tying the above together
- `requirements.txt` — Python dependencies

## Known limitations / honest scope

- Dataset subset is 5,000 products (not the full dataset) — a deliberate scoping decision for free-tier compute, not a technical ceiling
- No deployment optimization (ONNX/TensorRT export) — models run in native PyTorch/HuggingFace form
- No production serving/monitoring layer — this is a demo/portfolio application, not a production service
- The metadata-only description prompt has not yet been measured on a test set (e.g. counting how often a description adds a detail not in the metadata)
