import gradio as gr
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from src.backend.database.qdrant import QdrantDatabase
from src.frontend.responses import QdrantQueryResponses, QdrantArticleResponse
import torch

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

qdrant_url = os.environ.get("QDRANT_URL", ":memory:")
qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", device=device)
qdrant_database = QdrantDatabase(client, model)

def query_database(query: str, k=5):
    response = qdrant_database.query(query, paragraphs_per_document=1, docs_per_query=k)
    response = response[query]
    articles = QdrantQueryResponses(query, response)
    return articles.query_responses

load_dataset_button = gr.Button("Load Example Dataset")

css = """
.highlight-paragraph {background-color: rgba(167, 246, 243, 0.28);}
gradio-app {height: 100vh;}

#article-add-article {flex: 0 1 0; flex-grow: 0}
#article-add-menu {align-items: center}

#app-tab-section {flex: 1 0 0; flex-grow: 1}
#article-search {display: flex;}

#article-search-query {}
#article-search-btns {}
#article-search-output {flex: 1 0 0; overflow-y: auto;}
.article-full-text {
    height: 30vh; 
    overflow-y: scroll;
    scrollbar-width: none;
    -ms-overflow-style: none;
}

.article-progress-container {
  width: 100%;
  height: 8px;
  background: #ccc;
}

.article-progress-bar {
  height: 8px;
  background: linear-gradient(120deg, var(--secondary-600) 0%, var(--primary-500) 60%, var(--primary-600) 100%);;
  width: 0%;
}

"""

js = """
<script>
function initSingleArticleScroll(container) {
    const content = container.querySelector('.prose.article-full-text');
    const bar = container.querySelector('.article-progress-bar');
    if (!content || !bar) return;

    // Avoid double attachment
    if (container.dataset.scrollAttached) return;

    content.onscroll = () => {
      const scrollTop = content.scrollTop;
      const scrollHeight = content.scrollHeight - content.clientHeight;
      const progress = (scrollTop / scrollHeight) * 100;
      bar.style.width = `${progress}%`;
    };

    container.dataset.scrollAttached = 'true';
  }

  // MutationObserver to auto-init new ones
  const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      mutation.addedNodes.forEach(node => {
        if (!(node instanceof HTMLElement)) return;

        // If the node is a new container
        if (node.classList?.contains('article-container-item')) {
          initSingleArticleScroll(node);
        }
      });
    });
  });
  
  // Start observing the body (or a specific container)
  const target = document.getElementById('article-search-output');
  if (target) {
    observer.observe(target, { childList: true, subtree: true });
  } else {
    observer.observe(document.body, { childList: true, subtree: true });
  }
</script>
"""

progress_bar = """
<div class="article-progress-container">
    <div class="article-progress-bar"></div>
</div>
"""

with gr.Blocks() as semantic_search:
    topk_default = 5
    query_state = gr.State("")
    topk_state = gr.State(topk_default)
    with gr.Row(elem_id="article-search-query"):
        query = gr.Textbox(placeholder="Your query", label="Query")
        topk = gr.Number(topk_default, label="Number of returned documents", interactive=True)
    with gr.Row(elem_id="article-search-btns"):
        clear_btn = gr.ClearButton()
        load_example_btn = gr.Button("Load Example")
        submit_btn = gr.Button("Submit", variant="primary")

    with gr.Row(elem_id="article-search-output"):
        @gr.render(inputs=[query_state, topk_state])
        def render_output(query, k):
            with gr.Group(elem_id="article-container"):
                if query:
                    articles = query_database(query, k=k)
                    for article in articles:
                        with gr.Group(elem_classes="article-container-item"):
                            article_link = gr.HTML(article.article_link)
                            article_rel_paragraph = gr.HTML(article.html_most_relevant_paragraph)
                            with gr.Accordion(label="View full article", open=False):
                                article_out = gr.HTML(article.html_article, elem_classes=["article-full-text"])
                                gr.HTML(progress_bar)

    clear_btn.add([query, query_state])

    load_example_btn.click(lambda: "venuous thrombosis",
                           inputs=[],
                           outputs=[query],
                           show_progress="hidden")
    submit_btn.click(lambda x, y: (x, y),
                     inputs=[query, topk],
                     outputs=[query_state, topk_state])


scientific_papers_demo = gr.Blocks(css=css, head=js, theme=gr.themes.Ocean(), fill_height=True)
with scientific_papers_demo:
    gr.HTML("<h1>Find relevant articles using text queries</h1>")
    semantic_search.render()

    # this does not work for some reason
    """with gr.Tabs():
        with gr.TabItem("Semantic Search", elem_id="article-search"):
            semantic_search.render()

        with gr.TabItem("RAG", elem_id="article-rag"):
            pass"""


if __name__ == "__main__":
    # load_dataset_sample()
    scientific_papers_demo.launch()

