import gradio as gr
import json
from retriever import (
    ingest_files, ask_question, save_answer_to_file,
    show_chunk_stats, export_answer_to_pdf, suggest_questions
)

# ✅ Global glossary dictionary
glossary = {}

# ✅ Glossary preview function
def show_glossary():
    return json.dumps(glossary, indent=2, ensure_ascii=False)

# ✅ Upload handler
def handle_upload(files):
    status = ingest_files(files)
    stats = show_chunk_stats()
    return status, stats

# ✅ Export handler
def handle_export(format, question, answer, context, model_name, include_citations):
    try:
        context_text = context if isinstance(context, str) else context.value
        sources = "DocSmith Embedded Chunks"
        if format == "PDF":
            return export_answer_to_pdf(question, answer, context_text, sources, model_name, include_citations)
        else:
            return save_answer_to_file(answer, sources if include_citations else None)
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        return None

# ✅ Radio updater to avoid Gradio error
def handle_suggestions(style):
    questions, source_info = suggest_questions(style)
    updated_radio = gr.update(choices=questions, value=questions[0] if questions else None)
    return updated_radio, source_info

# ✅ Glossary auto-suggest logic
def suggest_glossary_terms(query):
    query = query.lower().strip()
    matches = [term for term in glossary if query in term or term in query]
    return gr.update(choices=matches, visible=bool(matches))

# ✅ Answer selected suggested question
def answer_selected_question(q, show_chunks, model_name):
    answer, overview, full_context, history = ask_question(q, show_chunks, model_name)
    return q, answer, overview, full_context, history

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    demo.title = "DocMentor — Your AI mentor for every document"
    gr.Markdown("## 📚 DocMentor — Your AI mentor for every document")

    with gr.Tab("📁 Upload & Embed"):
        with gr.Row():
            file_input = gr.File(file_types=[".txt", ".pdf", ".docx"], label="Upload Documents", file_count="multiple")
            upload_btn = gr.Button("📥 Embed Documents")
        with gr.Row():
            upload_output = gr.Textbox(label="Status", interactive=False)
            chunk_viewer = gr.Markdown(label="📊 Chunk Stats")

        upload_btn.click(
            fn=handle_upload,
            inputs=[file_input],
            outputs=[upload_output, chunk_viewer]
        )

    with gr.Tab("🔍 Ask a Question"):
        with gr.Row():
            question = gr.Textbox(label="Your Question", placeholder="e.g. What is the main theme?")
            model_choice = gr.Dropdown(
                label="Answer Model",
                choices=["MiniLM (fast retrieval)", "Phi3-mini (smart synthesis)", "Gemma-2B (advanced reasoning)"],
                value="Phi3-mini (smart synthesis)"
            )
        with gr.Accordion("ℹ️ Model Info", open=False):
            gr.Markdown("""
            ### 🧠 Model Guide  
            - ⚡ **MiniLM**: Fast retrieval, returns raw chunks.  
            - 🧠 **Phi3-mini**: Lightweight synthesis, good for short answers.  
            - 🧬 **Gemma-2B**: Advanced reasoning, ideal for abstract or multi-paragraph synthesis.
            """)

        show_chunks = gr.Checkbox(label="Show Retrieved Chunks", value=True)
        show_explanation = gr.Checkbox(label="Show explanation in answer", value=False)
        answer_btn = gr.Button("🧠 Get Answer")

        # ✅ Glossary auto-suggest dropdown
        suggested_terms = gr.Dropdown(
            label="📘 Suggested Glossary Terms",
            choices=[],
            interactive=True,
            visible=False
        )

        question.change(
            fn=suggest_glossary_terms,
            inputs=[question],
            outputs=[suggested_terms]
        )

        suggested_terms.change(
            fn=lambda term: term,
            inputs=[suggested_terms],
            outputs=[question]
        )

        with gr.Row():
            answer_output = gr.Textbox(label="Answer", lines=5)
            context_preview = gr.Markdown(label="📄 Document Overview", visible=True)
            context_full = gr.Markdown(label="📄 Full Context", visible=False)

        with gr.Row():
            show_context_btn = gr.Button("📄 Show Full Context")
            hide_context_btn = gr.Button("🧹 Hide Full Context")
   
    # ✅ Persistent history box
        history = gr.Textbox(label="🕘 Query History", lines=10, interactive=False)
        # ✅ Load saved history on app start
        try:
            with open("query_history.json", "r", encoding="utf-8") as f:
                saved_history = "\n".join(json.load(f))
        except:
            saved_history = ""

        history.value = saved_history

        answer_btn.click(
            fn=ask_question,
            inputs=[question, show_chunks, model_choice, show_explanation],
            outputs=[answer_output, context_preview, context_full, history]
        )

        show_context_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[context_full]
        )

        hide_context_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[context_full]
        )

    with gr.Tab("💡 Suggested Questions"):
        gr.Markdown("Let DocSmith inspire your inquiry based on embedded content.")
        question_style = gr.Dropdown(
            label="Question Style",
            choices=["Insightful", "Factual", "Multiple Choice", "Open-ended"],
            value="Insightful"
        )
        suggest_btn = gr.Button("✨ Suggest Questions")
        suggested_questions = gr.Radio(label="Suggested Questions", choices=[])
        source_info = gr.Markdown(label="📁 Source Info")

        suggest_btn.click(
            fn=handle_suggestions,
            inputs=[question_style],
            outputs=[suggested_questions, source_info]
        )

        suggested_questions.change(
            fn=answer_selected_question,
            inputs=[suggested_questions, show_chunks, model_choice],
            outputs=[question, answer_output, context_preview, context_full, history]
        )

    with gr.Tab("📤 Export Answer"):
        gr.Markdown("Choose your format and download the synthesized answer.")
        export_format = gr.Dropdown(
            choices=["PDF", "TXT"],
            label="Export Format",
            value="PDF"
        )
        include_citations = gr.Checkbox(label="Include citation footer", value=True)
        export_btn = gr.Button("📦 Export Answer")
        export_file = gr.File(label="Download Export")

        export_btn.click(
            fn=handle_export,
            inputs=[export_format, question, answer_output, context_full, model_choice, include_citations],
            outputs=[export_file]
        )

    with gr.Tab("📘 Glossary Editor"):
        gr.Markdown("### Add or Update Glossary Terms")
        glossary_term = gr.Textbox(label="Term")
        glossary_definition = gr.Textbox(label="Definition", lines=3)
        glossary_btn = gr.Button("💾 Save Term")
        glossary_status = gr.Textbox(label="Status", interactive=False)

        def update_glossary(term, definition):
            global glossary
            term = term.lower().strip()
            glossary[term] = definition
            with open("glossary.json", "w", encoding="utf-8") as f:
                json.dump(glossary, f, indent=2, ensure_ascii=False)
            safe_term = ''.join(c for c in term if ord(c) < 256)
            return f"✅ Glossary updated: '{safe_term}'"

        glossary_btn.click(
            fn=update_glossary,
            inputs=[glossary_term, glossary_definition],
            outputs=[glossary_status]
        )

        with gr.Accordion("📘 View Glossary", open=False):
            glossary_view = gr.Textbox(label="Current Glossary", lines=10, interactive=False)
            glossary_btn.click(
                fn=show_glossary,
                inputs=[],
                outputs=[glossary_view]
            )

demo.launch()