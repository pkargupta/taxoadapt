from pydantic import BaseModel, conset, Field
from typing_extensions import Annotated

class ClassifySchema(BaseModel):
    paper_id: Annotated[int, Field(strict=True, gt=-1)]
    class_options: conset(str, min_length=1, max_length=100)
    class_labels: conset(str, min_length=0, max_length=10)

nl = '\n'

init_classify_prompt = "You are a helpful assistant that identifies the class labels for the provided NLP research paper, performing multi-label classification."

main_classify_prompt = lambda node, paper: f'''Given the 'title' and 'abstract' (provided below) of an NLP research paper about {node.label}, select the class labels (tag 'class_options') that should be assigned to this paper (multi-label classification). If the research paper SHOULD NOT be labeled with any of the classes in 'class_options', then output an empty list. We provide additional information for each class option for your reference.

---
paper_id: {paper.id}
title: {paper.title}
abstract: {paper.abstract}
class_options (list of potential class labels): {"; ".join([f"{c}" for c in node.get_children()])}

Here is some additional information about each class option:
{nl.join([f"{c_label}:{nl}{nl}Description of {c_label}: {c.description}{nl}Example phrases used in {c_label} papers: {c.phrases}{nl}Example sentences used in {c_label} papers: {c.sentences}" for c_label, c in node.get_children().items()])}
---

Your output format should be in the following JSON format:
---
{{
    paper_title: {paper.title}
    class_options: {"; ".join([f"{c}" for c in node.get_children()])}
    class_labels: <list of paper's class labels or empty if none are applicable>
}}
---
'''

def classify_prompt(node, paper):
    return init_classify_prompt + '\n\n' + main_classify_prompt(node, paper)