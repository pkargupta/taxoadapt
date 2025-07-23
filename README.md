# TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora
<br>Priyanka Kargupta, Nan Zhang, Yunyi Zhang, Rui Zhang, Prasenjit Mitra, Jiawei Han</a>


Official implementation for [ACL 2025](https://2025.aclweb.org/) main track paper: [TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora](https://arxiv.org/abs/2506.10737).

![Framework Diagram of TaxoAdapt](https://github.com/pkargupta/taxoadapt/blob/main/framework.png)

TaxoAdapt is a framework that dynamically adapts an LLM-generated taxonomy to a given corpus across multiple dimensions. TaxoAdapt performs iterative hierarchical classification, expanding both the taxonomy width and depth based on corpus' topical distribution. We demonstrate its state-of-the-art performance across a diverse set of computer science conferences over the years to showcase its ability to structure and capture the evolution of scientific fields. As a multidimensional method, TaxoAdapt generates taxonomies that are 26.51% more granularity-preserving and 50.41% more coherent than the most competitive baselines judged by LLMs.

## Contents
  - [Setup](#setup)
    - [Arguments](#arguments)
  - [Custom Dataset](#custom-dataset)
  - [Video](#video)
  - [📖 Citation](#-citation)

## Setup
We use `python=3.8`, `torch=2.4.0`, and a two NVIDIA RTX A6000s. Other packages can be installed using:
```
pip install -r requirements.txt
```

To run the code with the default parameters, you can run the following command in the terminal:
```
python main.py
```
In order to run the code, you need to have a valid OpenAI API key and set it as an environment variable `OPENAI_API_KEY`. You can do this in your terminal as follows:
```export OPENAI_API_KEY='your_openai_api_key'```

### Arguments
The following are the primary arguments for TaxoAdapt (defined in main.py; modify as needed):

- `topic` $\rightarrow$ this is the topic of the corpus, e.g., "natural language processing", "robotics", etc.
- `dataset` $\rightarrow$ this is the name of the dataset, e.g., "llm_graph", "icra_2020", etc. The huggingface dataset should be added to the `construct_dataset` function in `main.py` (see below).
- `llm` $\rightarrow$ this is the LLM to be used for initial taxonomy construction, e.g., "gpt", "vllm", etc. You can replace the vLLM model in the `initializeLLM` function and the GPT model version in the `promptGPT` function of `model_definitions.py`.
- `max_depth` $\rightarrow$ this is the maximum depth of each taxonomy to be constructed.
- `init_levels` $\rightarrow$ this is the number of initial levels to be constructed in the initial taxonomy.
- `max_density` $\rightarrow$ this is the maximum density of papers to be mapped to a node (or unmapped papers at a parent node) in the taxonomies. If a leaf node has more than `max_density` papers, it will trigger depth expansion at that node. If a parent node has more than `max_density` papers that are unmapped to any of its children, it will trigger width expansion at that node.

In `main.py`, we define the different dimensions of research for a specific topic, each of which will be constructed as a separate taxonomy. You can modify the dimensions in the `args.dimensions` list.

## Custom Dataset
To use a custom dataset, you need to add it to the `construct_dataset` function in `main.py`. You may add it as follows:

```python
elif args.dataset == 'dataset_name':
        ds = load_dataset("huggingface_dataset_name")
```
We assume that the dataset has a `title` and `abstract` field for each paper. If not, you can modify the function to extract the relevant fields from your dataset.

## Video
You can find a video explanation of the TaxoAdapt framework and its results on YouTube: [TaxoAdapt Video](https://youtu.be/dKUeSm9GoyU).


## 📖 Citation
Please cite the paper and star this repo if you use TaxoAdapt and find it interesting/useful, thanks! Feel free to open an issue if you have any questions.

```bibtex
@article{kargupta2025taxoadapt,
  title={TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora},
  author={Kargupta, Priyanka and Zhang, Nan and Zhang, Yunyi and Zhang, Rui and Mitra, Prasenjit and Han, Jiawei},
  journal={arXiv preprint arXiv:2506.10737},
  year={2025}
}
```