import typing, sqlite3
import json, os
import config

# One SQLite database per model, stored in $veccisc_data_path/sc_databases/
DB_FILES = {
    'gpt-4o-mini': 'sc_v.db',
    'llama-3.1-8b-instant': 'sc_v_llama8b.db',
    'Qwen/Qwen2.5-7B-Instruct-Turbo': 'sc_v_qwen7b.db',
    'mistralai/Mistral-7B-Instruct-v0.1': 'sc_v_mistral7b.db',
    'llama-3.3-70b-versatile': 'sc_v_llama70b.db',
}

SCHEMA = '''
create table if not exists sampled_problems (id uuid, name text, data jsonb, added timestamp);

create table if not exists problem_sc_responses (id uuid,
  problem_id uuid,
  model text,
  body jsonb,
  resp jsonb,
  added timestamp);

create table if not exists problem_confidence_responses (id uuid,
  sc_id uuid,
  model text,
  body jsonb,
  resp jsonb,
  added timestamp);

create table if not exists problem_sc_response_vectors (id uuid,
    sc_id uuid,
    vector jsonb);

create index if not exists problem_sc_index1 on problem_sc_responses(id);
create index if not exists problem_confidence_index1 on problem_confidence_responses(id);
create index if not exists problem_vector1 on problem_sc_response_vectors(sc_id);
'''


def db_path(model:str) -> str:
    """Absolute path of the SQLite database holding all results for `model`."""
    return config.data_path('sc_databases', DB_FILES[model])


def init_db(model:str) -> str:
    """Create an empty results database (with schema) for `model`."""
    path = db_path(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    return path


class sqliteDB:
    def __init__(self, model:str = None, as_dict:bool = True, DB_URL:str = None) -> None:
        '''
        Open the results database for `model` (a key of DB_FILES), or an
        explicit database file via DB_URL.
        '''
        assert (model is None) != (DB_URL is None), 'pass exactly one of model or DB_URL'
        self.as_dict = as_dict
        self.conn = None
        self.cur = None
        self.DB_URL = DB_URL if DB_URL is not None else db_path(model)

    def __enter__(self):
        if not os.path.exists(self.DB_URL):
            raise FileNotFoundError(
                f'Database not found: {self.DB_URL}. Check ${config.DATA_PATH_ENV}, '
                'or create a new database with mdb.init_db(model).'
            )

        self.conn = sqlite3.connect(self.DB_URL)
        if self.as_dict:
            self.conn.row_factory = sqlite3.Row

        self.cur = self.conn.cursor()
        return self

    def execute(self, sql:str, row:list) -> None:
        row = [json.dumps(j) if isinstance(j, (dict, list)) else j for j in row]
        self.cur.execute(sql, row)

    def __iter__(self) -> typing.Iterator:
        for i in self.cur:
            d = {}
            for a, b in dict(i).items():
                try:
                    d[a] = json.loads(b)

                except:
                    d[a] = b

            yield d

    def commit(self) -> None:
        self.conn.commit()

    def __exit__(self, *_):
        self.conn.close()
