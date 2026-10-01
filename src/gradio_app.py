"""
Project 3 — Gradio demo app
Run this in the SAME kernel/session as project3_master.py, right after it —
it reuses the already-loaded index, models, and dataframe rather than
reloading anything.

Dependency (run once before this cell):
    !pip install gradio
"""
import os
import gradio as gr

TOP_K_CANDIDATES = 15  # how many FAISS results to pull before reranking


def run_search(query, top_k, generate_description):
    if not query or not query.strip():
        return [], "Enter a search query above."

    faiss_results = search_by_text(
        query, index, ordered_ids, subset_df, model, top_k=TOP_K_CANDIDATES
    )
    if faiss_results.empty:
        return [], "No results found."

    reranked = rerank_results_v2(query, faiss_results, reranker).head(int(top_k))

    gallery_items = []
    for _, row in reranked.iterrows():
        img_path = os.path.join(IMAGE_DIR, f"{row['id']}.jpg")
        caption = f"{row['productDisplayName']} | {row['usage']} | score={row['rerank_score']:.2f}"
        gallery_items.append((img_path, caption))

    description = ""
    if generate_description:
        top_row = reranked.iloc[0]
        img_path = os.path.join(IMAGE_DIR, f"{top_row['id']}.jpg")
        context = build_grounded_context(top_row, img_path)
        description = generate_with_phi3(context, phi_tokenizer, phi_model)
        description = f"**Top result — AI-generated description:**\n\n{description}"

    return gallery_items, description


with gr.Blocks(title="Visual Product Search") as demo:
    gr.Markdown("# Visual Product Search\nCLIP retrieval + cross-encoder reranking + grounded description generation")

    with gr.Row():
        query_box = gr.Textbox(label="Search query", placeholder="e.g. formal shirt, elegant dress")
        top_k_slider = gr.Slider(minimum=1, maximum=10, value=6, step=1, label="Results to show")

    describe_checkbox = gr.Checkbox(label="Generate AI description for top result (slower — runs BLIP + Phi-3)", value=False)
    search_button = gr.Button("Search", variant="primary")

    gallery_output = gr.Gallery(label="Results (reranked)", columns=3, height="auto")
    description_output = gr.Markdown()

    search_button.click(
        fn=run_search,
        inputs=[query_box, top_k_slider, describe_checkbox],
        outputs=[gallery_output, description_output],
    )
    query_box.submit(
        fn=run_search,
        inputs=[query_box, top_k_slider, describe_checkbox],
        outputs=[gallery_output, description_output],
    )

demo.launch(share=True, allowed_paths=[IMAGE_DIR])
