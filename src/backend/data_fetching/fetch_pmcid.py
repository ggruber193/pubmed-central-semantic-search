from scientificpapers_rag.data_fetching.data_fields import DataFields


def fetch_from_pmcid(pmcid: str):
    import requests
    from lxml import etree
    import re

    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/fullTextXML"
    resp = requests.get(base_url.format(pmcid))
    article_xml: etree._Element = etree.fromstring(resp.content)
    article_id = article_xml.find(".//article-id[@pub-id-type='pmcid']").text
    article_id = f"PMC{article_id}"

    # article_title = article_xml.find(".//article-title").text
    article_abstract = ''.join(list(article_xml.find(".//abstract/*").itertext()))

    section_elements = [section for section in article_xml.findall(".//sec")]

    section_names = ["Abstract"] + [section.find(".//title").text for section in section_elements]

    sections = [
        '\n'.join(
            [''.join([i for i in [j.text, j.tail] if i]) for j in section.xpath(".//*[name() != 'title']")]) for
        section in section_elements]

    sections = [re.subn("\\[[^\\]]*\\]", "", i, flags=re.DOTALL)[0].split('\n') for i in sections]
    sections = [article_abstract] + ['.'.join(i) for i in sections]

    output = {
        DataFields.ARTICLE_ID: article_id,
        # "article_title": article_title,
        DataFields.SECTION_NAMES: section_names,
        DataFields.SECTIONS: sections
    }
    return output
