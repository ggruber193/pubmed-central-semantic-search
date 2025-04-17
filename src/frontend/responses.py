from collections import defaultdict
from typing import Literal, Tuple

from qdrant_client.models import ScoredPoint
from qdrant_client.conversions import common_types as types

from backend.database.qdrant import ScientificPapersMainSchema, ScientificPapersChunksSchema


class QdrantArticleResponse:
    def __init__(self, query, document_point: ScoredPoint, highlight_points: list[ScoredPoint]):
        self.document_point = document_point
        self.highlight_points = highlight_points
        self.query = query

        self.paragraphs_per_section = {}
        self.highlighted_paragraphs = {}
        self._section_names_ordered = []
        self.article_id = ""
        self.abstract: list[str] = []
        self._relevant_paragraphs = []

        self._html_article = ""

        self._process_points()

    def _process_points(self):
        self.article_id = self.document_point.payload[ScientificPapersMainSchema.ARTICLE_ID]

        section_names = self.document_point.payload[ScientificPapersMainSchema.SECTION_NAMES]
        sections = self.document_point.payload[ScientificPapersMainSchema.SECTIONS]
        self._section_names_ordered = section_names
        paragraphs_per_section = {section_name: section for section_name, section in zip(section_names, sections)}

        highlighted_paragraphs = defaultdict(list)
        for point in self.highlight_points:
            paragraph_id = point.payload[ScientificPapersChunksSchema.PARAGRAPH_ID]
            section_name = point.payload[ScientificPapersChunksSchema.SECTION_NAME]
            highlighted_paragraphs[section_name].append(paragraph_id)
            self._relevant_paragraphs.append((section_name, paragraph_id))

        self.paragraphs_per_section = paragraphs_per_section
        self.highlighted_paragraphs = highlighted_paragraphs

    @staticmethod
    def section_html(paragraphs: list[str]):
        section = f"""
        <section>
            {'\n'.join([f'<p>{i}</p>' for i in paragraphs])}
        </section>
        """
        return section

    @property
    def html_article(self):
        if self._html_article:
            return self._html_article
        def article_header_html(text, h_tag: Literal["h1", "h2", "h3"] = "h2"):
            html_text = f"""
            <header>
                <{h_tag}>
                    {text}
                </{h_tag}>
            </header>
            """
            return html_text
        def section_html(paragraphs: list[str]):
            section = f"""
            <section>
                {'\n'.join([f'<p>{i}</p>' for i in paragraphs])}
            </section>
            """
            return section
        def mark_html(text):
            mark_tag = 'mark class="highlight-paragraph"'
            return f"<{mark_tag}>{text}</mark>"

        article_dict = {i: [k.strip() for k in j] for i, j in self.paragraphs_per_section.items()}
        for section, paragraph_ids in self.highlighted_paragraphs.items():
            for paragraph_id in paragraph_ids:
                for i in range(-1, 2):
                    extra_ind = paragraph_id + i
                    if 0 <= extra_ind < len(article_dict[section]):
                        paragraph = article_dict[section][extra_ind]
                        article_dict[section][extra_ind] = mark_html(paragraph)
        article_html = '\n'.join([
            f'{article_header_html(i, "h2")}\n{section_html(article_dict[i])}'
            for i in self._section_names_ordered])
        article_html = f"<article>{article_html}</article>"
        self._html_article = article_html
        return article_html

    @property
    def html_most_relevant_paragraph(self):
        section, paragraph_id = self._relevant_paragraphs[0]
        paragraph = []
        for i in range(-1, 2):
            extra_ind = paragraph_id + i
            if 0 <= extra_ind < len(self.paragraphs_per_section[section]):
                c_paragraph = self.paragraphs_per_section[section][extra_ind]
                paragraph.append(c_paragraph)
        return '\n'.join(paragraph)

    @property
    def article_link(self):
        link = "https://pmc.ncbi.nlm.nih.gov/articles/{}/".format(self.article_id)
        link_html = f'<a target="_blank" rel="noopener noreferrer" href="{link}">View full article on external site: {self.article_id}</a>'
        return link_html

class QdrantQueryResponses:
    def __init__(self, query: str, query_response: Tuple[types.QueryResponse, dict[str, types.QueryResponse]]):
        self.query_responses: list[QdrantArticleResponse] = self._process_response(query, query_response)

    @staticmethod
    def _process_response(query, query_responses: Tuple[types.QueryResponse, dict[str, types.QueryResponse]]):
        document_responses, highlight_dict = query_responses
        qdrant_article_responses = []
        for document in document_responses.points:
            highlight_points = highlight_dict[document.payload[ScientificPapersMainSchema.ARTICLE_ID]].points
            qdrant_article_responses.append(QdrantArticleResponse(query, document, highlight_points))
        return qdrant_article_responses

    def __getitem__(self, idx):
        return self.query_responses[idx]