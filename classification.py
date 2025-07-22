from pydantic import BaseModel, conset, Field
from typing_extensions import Annotated
from pydantic.types import StringConstraints

class ClassifySchema(BaseModel):
    explanation: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]
    class_labels: conset(str, min_length=0, max_length=10)

nl = '\n'

init_classify_prompt = """You are a helpful classification assistant that classifies a given paper with either one of the existing labels, or "None" if none of them tightly describe the paper. Overall, you identify if any of the class labels should be mapped to the provided research paper, performing multi-label classification."""

main_classify_prompt = lambda node, paper: f'''Given the 'title' and 'abstract' (provided below) of a research paper about {node.label}, determine which, if any, of the class labels (tag 'class_options') should be assigned to this paper (multi-label classification). If the research paper SHOULD NOT be labeled with any of the classes in 'class_options', then output 'None'. Else, if at least one class is a primary topic of the paper, then output it. Be picky in your judgement, if the paper does not TIGHTLY fall under a class_option, then DO NOT include it. We provide additional information for each class option for your reference.

---
paper_id: {paper.id}
title: {paper.title}
abstract: {paper.abstract}
class_options (list of potential class labels): {"; ".join([f"{c}" for c in node.get_children()])}; 'None' if none of the preceding options are a primary topic of the paper.

Here is some additional information about each class option:
{nl.join([f"{c_label}:{nl}{nl}Description of {c_label}: {c.description}{nl}" for c_label, c in node.get_children().items()])}
---

Your output format should be in the following JSON format:
---
{{
    explanation: <1 sentence, step-by-step explanation>
    class_labels: <list of the paper's assigned class labels or 'None' if the paper should not be mapped to any of these classes>
}}
---
'''

def classify_prompt(node, paper):
    return init_classify_prompt + '\n\n' + main_classify_prompt(node, paper)