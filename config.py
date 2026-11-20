"""Central path configuration for the VecCISC repository.

Large generated artifacts (the per-model SQLite databases, the cached JSON
exports of those databases, and OpenAI/Groq/Together batch files) live outside
the repository in the directory named by the ``veccisc_data_path`` environment
variable:

    $veccisc_data_path/
        sc_databases/   one SQLite database per model (see README)
        json_batches/   per (dataset, model) JSON caches read by veccisc.py
        batches/        batch-API request/response .jsonl files (data_gen.py)
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = os.path.join(REPO_ROOT, 'datasets')
PROMPTS_DIR = os.path.join(REPO_ROOT, 'prompts')
SUPPORTING_DIR = os.path.join(REPO_ROOT, 'supporting')
RESULTS_DIR = os.path.join(SUPPORTING_DIR, 'results')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')

DATA_PATH_ENV = 'veccisc_data_path'


def data_path(*parts: str) -> str:
    """Return a path inside the external data directory ($veccisc_data_path)."""
    root = os.environ.get(DATA_PATH_ENV)
    if not root:
        raise RuntimeError(
            f'The "{DATA_PATH_ENV}" environment variable is not set. Point it at '
            'the directory containing sc_databases/ and json_batches/, e.g.\n'
            f'    export {DATA_PATH_ENV}=/path/to/veccisc-data'
        )

    return os.path.join(os.path.expanduser(root), *parts)
