import gradio as gr
from config import logger, CUSTOM_CSS, ReasoningMode, AppConfig, ModelConfig
from core import AdvancedReasoner, PromptEngine

# Initialize system
reasoner = AdvancedReasoner()

def get_metrics_html() -> str:
    """Generate enhanced metrics HTML"""
    m = reasoner.metrics
    cache_stats = reasoner.cache.get_stats()
    status = '<span class="status-active">Active</span>' if m.tokens_used > 0 else 'Ready'
    
    return f"""<div class="metrics-card">
    <strong>Inference:</strong> {m.inference_time:.2f}s<br>
    <strong>Avg Time:</strong> {m.avg_response_time:.2f}s<br>
    <strong>Speed:</strong> {m.tokens_per_second:.1f} tok/s<br>
    <strong>Reasoning:</strong> {m.reasoning_depth} steps<br>
    <strong>Corrections:</strong> {m.self_corrections}<br>
    <strong>Confidence:</strong> {m.confidence_score:.1f}%<br>
    <strong>Total:</strong> {m.total_conversations}<br>
    <strong>Tokens:</strong> {m.tokens_used:,}<br>
    <strong>Peak:</strong> {m.peak_tokens}<br>
    <strong>Cache:</strong> {cache_stats['hit_rate']}% hit rate<br>
    <strong>Status:</strong> {status}<br>
    <strong>Session:</strong> {reasoner.session_id[:8]}...
    </div>"""

def get_empty_analytics_html() -> str:
    """Generate empty analytics HTML"""
    return """<div class="analytics-panel">
    <h3>No data yet</h3>
    <p>Start a conversation to see analytics!</p>
    </div>"""

def create_ui() -> gr.Blocks:
    """Create enhanced Gradio interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue=AppConfig.THEME_PRIMARY,
            secondary_hue=AppConfig.THEME_SECONDARY,
            font=gr.themes.GoogleFont("Inter")
        ),
        css=CUSTOM_CSS,
        title="Advanced AI Reasoning System Pro"
    ) as demo:
        
        gr.HTML("""
        <div class="research-header">
            <h1>Advanced AI Reasoning System Pro</h1>
            <p><strong>Enhanced Implementation:</strong> Tree of Thoughts + Constitutional AI + Multi-Agent Validation + Caching + Rate Limiting</p>
            <div style="margin-top: 1rem;">
                <span class="badge">Yao et al. 2023 - Tree of Thoughts</span>
                <span class="badge">Bai et al. 2022 - Constitutional AI</span>
                <span class="badge">Enhanced with 6 Reasoning Modes</span>
                <span class="badge">Performance Optimized</span>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            # Main Chat Tab
            with gr.Tab("Reasoning Workspace"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Reasoning Workspace",
                            height=550,
                            show_copy_button=True,
                            type="messages",
                            avatar_images=(
                                "https://api.dicebear.com/7.x/avataaars/svg?seed=User",
                                "https://api.dicebear.com/7.x/bottts/svg?seed=AI"
                            )
                        )
                        
                        msg = gr.Textbox(
                            placeholder="Enter your complex problem or research question... (Max 10,000 characters)",
                            label="Query Input",
                            lines=3,
                            max_lines=10
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button("Process", variant="primary", scale=2)
                            clear_btn = gr.Button("Clear", scale=1)
                            pdf_btn = gr.Button("Download PDF", scale=1)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuration")
                        
                        reasoning_mode = gr.Radio(
                            choices=[mode.value for mode in ReasoningMode],
                            value=ReasoningMode.TREE_OF_THOUGHTS.value,
                            label="Reasoning Method",
                            info="Select the reasoning strategy"
                        )
                        
                        prompt_template = gr.Dropdown(
                            choices=list(PromptEngine.TEMPLATES.keys()),
                            value="Custom",
                            label="Prompt Template",
                            info="Pre-built prompt templates"
                        )
                        
                        enable_critique = gr.Checkbox(
                            label="Enable Self-Critique",
                            value=True,
                            info="Add validation phase"
                        )
                        
                        use_cache = gr.Checkbox(
                            label="Use Cache",
                            value=True,
                            info="Cache responses for speed"
                        )
                        
                        model = gr.Dropdown(
                            choices=[m.model_id for m in ModelConfig],
                            value=ModelConfig.LLAMA_70B.model_id,
                            label="Model",
                            info="Select AI model"
                        )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            temperature = gr.Slider(
                                AppConfig.MIN_TEMPERATURE, 
                                AppConfig.MAX_TEMPERATURE, 
                                value=AppConfig.DEFAULT_TEMPERATURE, 
                                step=0.1,
                                label="Temperature",
                                info="Higher = more creative"
                            )
                            max_tokens = gr.Slider(
                                AppConfig.MIN_TOKENS, 
                                8000, 
                                value=AppConfig.DEFAULT_MAX_TOKENS, 
                                step=500,
                                label="Max Tokens",
                                info="Maximum response length"
                            )
                        
                        gr.Markdown("### Live Metrics")
                        metrics_display = gr.Markdown(value=get_metrics_html())
                        
                        with gr.Accordion("Info", open=False):
                            gr.Markdown(f"""
                            **Session ID:** `{reasoner.session_id}`  
                            **Cache Size:** {AppConfig.CACHE_SIZE}  
                            **Rate Limit:** {AppConfig.RATE_LIMIT_REQUESTS} req/{AppConfig.RATE_LIMIT_WINDOW}s  
                            **Max History:** {AppConfig.MAX_HISTORY_LENGTH} messages
                            """)
            
            # Export Tab
            with gr.Tab("Export & History"):
                gr.Markdown("### Export Conversation History")
                
                with gr.Row():
                    export_format = gr.Radio(
                        choices=["json", "markdown", "txt", "pdf"],
                        value="markdown",
                        label="Export Format"
                    )
                    include_meta = gr.Checkbox(
                        label="Include Metadata",
                        value=True
                    )
                
                export_btn = gr.Button("Export Now", variant="primary")
                export_output = gr.Code(label="Exported Data", language="markdown", lines=20)
                download_file = gr.File(label="Download File")
                
                gr.Markdown("---")
                gr.Markdown("### Search Conversations")
                
                with gr.Row():
                    search_input = gr.Textbox(
                        placeholder="Enter keyword to search...", 
                        scale=3,
                        label="Search Query"
                    )
                    search_btn = gr.Button("Search", scale=1)
                
                search_results = gr.Markdown("No results yet. Enter a keyword and click Search.")
                
                gr.Markdown("---")
                gr.Markdown("### Conversation History")
                history_stats = gr.Markdown("Loading...")
            
            # Analytics Tab
            with gr.Tab("Analytics & Insights"):
                refresh_btn = gr.Button("Refresh Analytics", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Performance Metrics")
                        analytics_display = gr.Markdown(get_empty_analytics_html())
                    
                    with gr.Column():
                        gr.Markdown("### Cache Statistics")
                        cache_display = gr.Markdown("No cache data yet.")
                
                gr.Markdown("---")
                gr.Markdown("### Usage Distribution")
                
                with gr.Row():
                    model_dist = gr.Markdown("**Model Usage:** No data")
                    mode_dist = gr.Markdown("**Mode Usage:** No data")
            
            # Settings Tab
            with gr.Tab("Settings"):
                gr.Markdown("### Application Settings")
                
                gr.Markdown(f"""
                **Current Configuration:**
                
                | Setting | Value |
                |---------|-------|
                | Max History Length | {AppConfig.MAX_HISTORY_LENGTH} |
                | Max Conversation Storage | {AppConfig.MAX_CONVERSATION_STORAGE} |
                | Cache Size | {AppConfig.CACHE_SIZE} |
                | Cache TTL | {AppConfig.CACHE_TTL}s |
                | Rate Limit | {AppConfig.RATE_LIMIT_REQUESTS} requests per {AppConfig.RATE_LIMIT_WINDOW}s |
                | Request Timeout | {AppConfig.REQUEST_TIMEOUT}s |
                | Max Retries | {AppConfig.MAX_RETRIES} |
                | Export Directory | `{AppConfig.EXPORT_DIR}` |
                | Backup Directory | `{AppConfig.BACKUP_DIR}` |
                """)
                
                clear_cache_btn = gr.Button("Clear Cache", variant="stop")
                cache_status = gr.Markdown("")
        
        # Define pdf_file_output BEFORE event handlers
        pdf_file_output = gr.File(visible=False)
        
        # Event handlers
        def process_message(message, history, mode, critique, model_name, temp, tokens, template, cache):
            if not message.strip():
                return history, get_metrics_html()
            
            history = history or []
            mode_enum = ReasoningMode(mode)
            
            history.append({"role": "user", "content": message})
            yield history, get_metrics_html()
            
            history.append({"role": "assistant", "content": ""})
            
            for response in reasoner.generate_response(
                message, history[:-1], model_name, mode_enum, 
                critique, temp, tokens, template, cache
            ):
                history[-1]["content"] = response
                yield history, get_metrics_html()
        
        def reset_chat():
            reasoner.clear_history()
            return [], get_metrics_html()
        
        def export_conv(format_type, include_metadata):
            content, filename = reasoner.export_conversation(format_type, include_metadata)
            return content, filename
        
        def download_chat_pdf():
            """Download current chat as PDF"""
            pdf_file = reasoner.export_current_chat_pdf()
            if pdf_file:
                return pdf_file
            return None
        
        def search_conv(keyword):
            if not keyword.strip():
                return "Please enter a search keyword."
            
            results = reasoner.search_conversations(keyword)
            if not results:
                return f"No results found for '{keyword}'."
            
            output = f"### Found {len(results)} result(s) for '{keyword}'\n\n"
            for idx, entry in results[:10]:
                output += f"**{idx + 1}.** {entry.timestamp} | {entry.model}\n"
                output += f"**User:** {entry.user_message[:100]}...\n\n"
            
            if len(results) > 10:
                output += f"\n*Showing first 10 of {len(results)} results*"
            
            return output
        
        def refresh_analytics():
            analytics = reasoner.get_analytics()
            if not analytics:
                return get_empty_analytics_html(), "No cache data.", "No data", "No data"
            
            analytics_html = f"""<div class="analytics-panel">
            <h3>Session Analytics</h3>
            <p><strong>Session ID:</strong> {analytics['session_id']}</p>
            <p><strong>Total Conversations:</strong> {analytics['total_conversations']}</p>
            <p><strong>Total Tokens:</strong> {analytics['total_tokens']:,}</p>
            <p><strong>Total Time:</strong> {analytics['total_time']:.1f}s</p>
            <p><strong>Avg Time:</strong> {analytics['avg_inference_time']:.2f}s</p>
            <p><strong>Peak Tokens:</strong> {analytics['peak_tokens']}</p>
            <p><strong>Most Used Model:</strong> {analytics['most_used_model']}</p>
            <p><strong>Most Used Mode:</strong> {analytics['most_used_mode']}</p>
            <p><strong>Errors:</strong> {analytics['error_count']}</p>
            </div>"""
            
            cache_html = f"""**Cache Performance:**
            - Hits: {analytics['cache_hits']}
            - Misses: {analytics['cache_misses']}
            - Total: {analytics['cache_hits'] + analytics['cache_misses']}
            """
            
            model_dist_html = f"**Model Usage:** {analytics['most_used_model']}"
            mode_dist_html = f"**Mode Usage:** {analytics['most_used_mode']}"
            
            return analytics_html, cache_html, model_dist_html, mode_dist_html
        
        def update_history_stats():
            count = len(reasoner.conversation_history)
            if count == 0:
                return "No conversations yet."
            
            return f"""**Total Conversations:** {count}  
            **Session:** {reasoner.session_id[:8]}..."""
        
        def clear_cache_action():
            reasoner.cache.clear()
            return "Cache cleared successfully!"
        
        # Connect events
        submit_btn.click(
            process_message,
            [msg, chatbot, reasoning_mode, enable_critique, model, temperature, max_tokens, prompt_template, use_cache],
            [chatbot, metrics_display]
        ).then(lambda: "", None, msg)
        
        msg.submit(
            process_message,
            [msg, chatbot, reasoning_mode, enable_critique, model, temperature, max_tokens, prompt_template, use_cache],
            [chatbot, metrics_display]
        ).then(lambda: "", None, msg)
        
        clear_btn.click(reset_chat, None, [chatbot, metrics_display])
        
        # PDF Download button
        pdf_btn.click(download_chat_pdf, None, pdf_file_output)
        
        export_btn.click(export_conv, [export_format, include_meta], [export_output, download_file])
        search_btn.click(search_conv, search_input, search_results)
        refresh_btn.click(
            refresh_analytics, 
            None, 
            [analytics_display, cache_display, model_dist, mode_dist]
        )
        clear_cache_btn.click(clear_cache_action, None, cache_status)
        
        # Update history stats on load
        demo.load(update_history_stats, None, history_stats)
    
    return demo

if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("Starting Advanced AI Reasoning System Pro...")
        logger.info(f"Session ID: {reasoner.session_id}")
        logger.info("="*60)
        
        demo = create_ui()
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            show_api=False,
            favicon_path=None
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise