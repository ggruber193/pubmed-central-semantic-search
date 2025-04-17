def fetch_article_by_pmcid(pmcid: str):
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/fullTextXML"
    resp = requests.get(base_url.format(pmcid), headers=header)
    article_xml: etree._Element = etree.fromstring(resp.content)
    article_id = article_xml.find(".//article-id[@pub-id-type='pmcid']").text
    article_id = f"PMC{article_id}"

    article_title = article_xml.find(".//article-title").text
    article_abstract = article_xml.find(".//abstract/*").text

    section_elements = [section for section in article_xml.findall(".//sec")]

    section_names = [section.find(".//title").text for section in section_elements]

    sections = [
        '\n'.join([''.join([i for i in [j.text, j.tail] if i]) for j in section.xpath(".//*[name() != 'title']")]) for
        section in section_elements]

    sections = [re.subn("\\[.*\\]", "", i, flags=re.DOTALL)[0].split('\n') for i in sections]
    sections = [re.split("\\. |\n", '.'.join(i)) for i in sections]

    output = {"article_id": article_id, "article_title": article_title, "article_abstract": article_abstract,
              "section_names": section_names, "sections": sections}
    return output
