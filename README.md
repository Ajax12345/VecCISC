# VecCISC

Official repository for the paper **"VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection"** (Findings of ACL 2026).

James Petullo, Sonny George, Dylan Cashman, Nianwen Xue

[[Paper]](https://aclanthology.org/2026.findings-acl.1305/) [[PDF]](https://aclanthology.org/2026.findings-acl.1305.pdf)

## Abstract

A standard technique for scaling inference-time reasoning is Self-Consistency, whereby multiple candidate answers are sampled from an LLM and the most common answer is selected. More recently, it has been shown that weighted majority voting (e.g. Confidence-Informed Self Consistency (CISC)), which assigns a confidence value to each candidate answer and chooses the answer with the largest accumulated score, tends to be more accurate on a wide range of popular benchmarks. In practice, weighted majority voting necessitates calling a critic LLM on each candidate’s reasoning trace to produce the answer’s confidence score. This secondary series of LLM calls greatly increases the overhead and cost of weighted majority voting, despite its potential performance benefits. To reduce this expense, we propose VecCISC, a lightweight, adaptive framework that uses a measure of semantic similarity to filter reasoning traces that are semantically equivalent to others, degenerate, or hallucinated, thus decreasing the number of candidate answers that must be evaluated by the critic. To ensure adequate experimental thoroughness, we evaluated VecCISC on five challenging, widely-adopted datasets spanning the domains of mathematics, chemistry, biology, commonsense reasoning, and the humanities. Our results demonstrate that VecCISC reduces the total token usage by 47%, while maintaining or exceeding the accuracy of CISC.


## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

| Variable | Needed for |
|---|---|
| `veccisc_data_path` | Anything that reads or writes the generated data: databases, JSON caches and batch files |
| `OPENAI_API_KEY` | Generating new data (`data_gen.py`) |

Generating the paper tables from `supporting/results/` needs neither variable.

### Data directory (`veccisc_data_path`)

The generated data is too large for git: sampled traces, critic responses and embeddings. It lives in a separate directory:

```
$veccisc_data_path/
  sc_databases/   SQLite database per model:
                  sc_v.db (GPT-4o mini), sc_v_llama8b.db, sc_v_qwen7b.db,
                  sc_v_mistral7b.db, sc_v_llama70b.db
  json_batches/   JSON export per (dataset, model), e.g. aqua_rat_gpt-4o-mini.json.
                  This is what veccisc.py reads. If a file is missing, it is rebuilt
                  from the matching database.
  batches/        batch-API request and response files written and read by data_gen.py
```

```bash
export veccisc_data_path=/path/to/veccisc-data
```

<!-- TODO: add the download link for the released data (e.g., Hugging Face Hub / Zenodo) -->

### Benchmarks

```bash
python d_sets.py
```

This command downloads AQuA-RAT (train), CommonsenseQA (validation), ARC-Challenge (test), MMLU-Pro (test) and GPQA into `datasets/`. GPQA is gated: accept its terms on the [Hugging Face page](https://huggingface.co/datasets/Idavidrein/gpqa) and run `hf auth login` first.

## Reproducing the paper

### Tables

```bash
python veccisc.py tables          # LaTeX for Tables 1-5, from supporting/results/
python veccisc.py token-savings   # headline token savings: total (SC + critic) and critic-only
python veccisc.py case-study      # LaTeX for Appendix A.3
python viz_results.py             # Figures 3-5 -> figures/
```

| Paper | Source |
|---|---|
| Table 1, 2: critic-call budget reductions (KMeans, HAC) | `results_full_variance.json`, `results_hac_full_variance.json` |
| Table 3, 4: critic-token reductions (KMeans, HAC) | `results_kmeans_token_usage.json`, `results_hac_token_usage.json` |
| Table 5: accuracy | `results_full_variance.json` + `results_hac_full_variance.json` |
| Figures 3-5: tuned T and K | `hyperparameters.json` |

### Re-running the experiments from the generated data

These steps require `veccisc_data_path`.

```bash
# grid-search T (softmax temperature), C (K for KMeans) and C_HAC (K for HAC)
python veccisc.py tune --hp-file hyperparameters.json --params T C C_HAC

# run every method 10 times per (dataset, model) with the tuned hyperparameters
python veccisc.py results --cluster kmeans --out results_kmeans_token_usage.json
python veccisc.py results --cluster HAC    --out results_hac_token_usage.json
```

Results are written to `supporting/results/`. KMeans and the random-ablation baselines are stochastic, so per-run numbers vary slightly; `cluster_num` (critic calls) and the SC/CISC baselines are deterministic.

### Generating data from scratch

`data_gen.py` builds the databases through provider batch APIs. Requests go to OpenAI for GPT-4o mini and the embeddings, Groq for the Llama models, and Together for Qwen and Mistral. Settings: *n* = 20 samples per question, base-LLM temperature 0.7, critic temperature 0.3. 1,000 questions were sampled per dataset; all 546 GPQA questions were used.

```bash
M=gpt-4o-mini
python data_gen.py init-db --model $M
python data_gen.py sample  --model $M --num 1000       # or: migrate --src-model gpt-4o-mini  (same questions for every model)
python data_gen.py expand-sc --model $M
python data_gen.py prepare-sc-batch --model $M         # submit the printed .jsonl to the provider's batch API
python data_gen.py import-sc-batch --model $M --batch-file <output.jsonl>
python data_gen.py expand-confidence --model $M
python data_gen.py prepare-confidence-batch --model $M # submit, then:
python data_gen.py import-confidence-batch --model $M --batch-file <output.jsonl>
python data_gen.py init-vectors --model $M
python data_gen.py prepare-vector-batch --model $M     # submit to the OpenAI batch API, then:
python data_gen.py import-vector-batch --model $M --batch-file <output.jsonl>
```

Run `python data_gen.py --help` for all options.

### Database schema

Each database stores the results for one model across all five benchmarks.

- **`sampled_problems`**: problems sampled from the benchmarks. Columns: `id` (UUID), `name` (benchmark), `data` (JSON: question, options, correct answer).
- **`problem_sc_responses`**: the self-consistency samples (about 20 per problem). Columns: `problem_id` → `sampled_problems.id`, `body` (request sent to the provider), `resp` (JSON with the parsed answer under `json` and the full raw reasoning trace under `resp`). A few problems have fewer than 20 samples because of provider errors.
- **`problem_confidence_responses`**: the critic's confidence for each sample. Columns: `sc_id` → `problem_sc_responses.id`, `body`, `resp` (parsed confidence plus the critic's reasoning).
- **`problem_sc_response_vectors`**: the `text-embedding-3-small` embedding of each full reasoning trace. Column: `sc_id` → `problem_sc_responses.id`.

## Citation

```bibtex
@inproceedings{petullo-etal-2026-veccisc,
    title = "{V}ec{CISC}: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection",
    author = "Petullo, James  and
      George, Sonny  and
      Cashman, Dylan  and
      Xue, Nianwen",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1305/",
    doi = "10.18653/v1/2026.findings-acl.1305",
    pages = "26184--26200",
    ISBN = "979-8-89176-395-1",
}
```
