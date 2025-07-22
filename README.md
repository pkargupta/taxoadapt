# TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora
<br>Priyanka Kargupta, Nan Zhang, Yunyi Zhang, Rui Zhang, Prasenjit Mitra, Jiawei Han</a>


Official implementation for [ACL 2025](https://2025.aclweb.org/) main track paper: [TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora](https://arxiv.org/abs/2506.10737).

![Framework Diagram of TaxoAdapt](https://github.com/pkargupta/taxoadapt/blob/main/framework.png)

TaxoAdapt is a framework that dynamically adapts an LLM-generated taxonomy to a given corpus across multiple dimensions. TaxoAdapt performs iterative hierarchical classification, expanding both the taxonomy width and depth based on corpus' topical distribution. We demonstrate its state-of-the-art performance across a diverse set of computer science conferences over the years to showcase its ability to structure and capture the evolution of scientific fields. As a multidimensional method, TaxoAdapt generates taxonomies that are 26.51% more granularity-preserving and 50.41% more coherent than the most competitive baselines judged by LLMs.

## Contents
  - [Contents](#contents)
  - [Setup](#setup)
    - [Arguments](#arguments)
  - [Custom Dataset](#custom-dataset)
  - [Video](#video)
  - [📖 Citations](#-citations)

## Setup
We use `python=3.8`, `torch=2.4.0`, and a two NVIDIA RTX A6000s. Other packages can be installed using:
```
pip install -r requirements.txt
```

In `main.py`, we define the list of `themes` (e.g., terrorism, natural_disasters, and politics) and `titles` (e.g., 2019_hk_legislative), which the former is simply used for the sake of input/output organization and the latter corresponds to the name of the input key event corpus. In order to run the following command after modifying the arguments as needed. The ground-truth episodes are defined within `run.py` as well.

```
python main.py
```
### Arguments
The following are the primary arguments for EpiMine (defined in run.py; modify as needed):

- `theme` $\rightarrow$ theme of key event
- `title` $\rightarrow$ key event to mine episodes for
- `gpu` $\rightarrow$ GPU to use; refer to nvidia-smi
- `output_dir` $\rightarrow$ default='final_output'; where to save the detected episodes.
- `lm_type` $\rightarrow$ default=`bbu`; used for computing word embeddings (`bbu` is bert-base-uncased)
- `layer` $\rightarrow$ default=12; last layer of BERT 
- `emb_dim` $\rightarrow$ default=768; Sentence and document embedding dimensions (default based on bert-base-uncased).
- `batch_size` $\rightarrow$ default=32; Batch size of episodes to segments to process (just for efficiency purposes).
- `doc_thresh` $\rightarrow$ default=0.25; Top articles to consider for candidate episode estimation.
- `vocab_min_occurrence` $\rightarrow$ default=1; Minimum frequency to be added into vocabulary.
- `eval_top` $\rightarrow$ default=5; Number of segments to consider for evaluation.
- `num` $\rightarrow$ default=5; Number of ground truth episodes for theme/key event.
- `trials` $\rightarrow$ default=10
- `api_key` $\rightarrow$ Anthropic API Key

## Custom Dataset
We provide all segmented articles for each key event in `episode_dataset/[theme]/[key_event]/[key_event]_segmented_raw.txt`. We also provide all episode-annotated articles in `groundtruth/[key_event]_groundtruth.txt`. All episode descriptions (used for article episode annotation) are provided in `groundtruth/key_event_episode_descriptions.xlsx`.

## Video
You can find a video explanation of the TaxoAdapt framework and its results on YouTube: [TaxoAdapt Video](https://www.youtube.com/watch?v=example).

## 📖 Citations
Please cite the paper and star this repo if you use TaxoAdapt and find it interesting/useful, thanks! Feel free to open an issue if you have any questions.

```bibtex
@article{kargupta2025taxoadapt,
  title={TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora},
  author={Kargupta, Priyanka and Zhang, Nan and Zhang, Yunyi and Zhang, Rui and Mitra, Prasenjit and Han, Jiawei},
  journal={arXiv preprint arXiv:2506.10737},
  year={2025}
}
```