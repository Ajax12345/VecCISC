"""Data-generation pipeline for VecCISC.

For each model, all results are stored in one SQLite database under
$veccisc_data_path/sc_databases/ (see mdb.py and the README). Requests are sent
through batch APIs (OpenAI, Groq, Together): the prepare-* steps write a .jsonl
request file to $veccisc_data_path/batches/, which you submit to the provider;
the import-* steps read the provider's output file from the same folder.

Pipeline for one model (n = 20 samples per question):
    1. init-db                   create the empty database
    2. sample | migrate          sample problems, or copy them from another model's db
    3. expand-sc                 create 20 self-consistency requests per problem (T = 0.7)
    4. prepare-sc-batch          -> submit to provider -> import-sc-batch --batch-file <output>
    5. expand-confidence         create one critic request per sampled answer (T = 0.3)
    6. prepare-confidence-batch  -> submit to provider -> import-confidence-batch --batch-file <output>
    7. init-vectors, prepare-vector-batch -> submit to OpenAI -> import-vector-batch --batch-file <output>
       (embeddings: text-embedding-3-small)
"""

import llm, mdb, d_sets, config
import typing, datetime
import os, json, re
import uuid, ast
import argparse

CQA_SC_LLM = llm.LLM('cqa_sc')
CONFIDENCE_EVALUATOR_LLM = llm.LLM('confidence_evaluator')

DATASETS = {i.NAME: i for i in d_sets.DS}


def gen_random_uuid() -> str:
    return str(uuid.uuid4())


def sample_problems(model:str, sample_num:typing.Union[int, None], dset:d_sets) -> None:
    with mdb.sqliteDB(model, as_dict = True) as conn:
        data = dset.load() if sample_num is None else dset.load().sample(sample_num)
            
        for i in data:
            conn.execute('insert into sampled_problems values (?, ?, ?, datetime("now"))', [gen_random_uuid(), dset.NAME, i.to_dict()])

        conn.commit()


def expand_sampled_to_sc(dset:d_sets, sc_num:int, params:dict) -> None:
    with mdb.sqliteDB(params['model'], as_dict = True) as conn:
        conn.execute('select p.* from sampled_problems p where p.name = ?', [dset.NAME])
        sampled_problems = [*conn]

    with mdb.sqliteDB(params['model'], as_dict = True) as conn:
        for problem in sampled_problems:
            for _ in range(sc_num):
                body = CQA_SC_LLM({
                    'retries': 2,
                    'temperature': params['temperature'],
                    'model': params['model'],
                    'params': {
                        'system': {},
                        'user': {
                            'question': problem['data']['question'],
                            'options':problem['data']['options']
                        }
                    }
                }, display_prompts=False, batch_format=True)

                conn.execute('''
                    insert into problem_sc_responses values 
                            (?, ?, ?, ?, null, null)
                ''', [gen_random_uuid(), problem['id'], params['model'], body])

        conn.commit()


def sc_to_confidence(model:str, params:dict) -> None:
    with mdb.sqliteDB(model, as_dict = True) as conn:
        conn.execute('''
            select p.id, s.data, p.resp from problem_sc_responses p 
            join sampled_problems s on s.id = p.problem_id
            left join problem_confidence_responses c on c.sc_id = p.id
            where p.model = ? and p.resp is not null and c.sc_id is null
        ''', [model])
        sc_results = [*conn]

    for ind in range(0, len(sc_results), 500):

        with mdb.sqliteDB(model, as_dict = True) as conn:
            for sc in sc_results[ind:ind+500]:
                body = CONFIDENCE_EVALUATOR_LLM({
                    'retries': 2,
                    'temperature': params['temperature'],
                    'model': params['model'],
                    'params': {
                        'system': {},
                        'user': {
                            'question': sc['data']['question'],
                            'options': sc['data']['options'],
                            'answer': sc['resp']['json']['answer'],
                            'reasoning': sc['resp']['resp'],
                        }
                    }
                }, display_prompts=False, batch_format=True)

                conn.execute('''
                        insert into problem_confidence_responses values 
                                (?, ?, ?, ?, null, null)
                    ''', [gen_random_uuid(), sc['id'], params['model'], body])

            conn.commit()


def prepare_sc_batches(model:str) -> str:
    with mdb.sqliteDB(model, as_dict = True) as conn:
        conn.execute('select p.* from problem_sc_responses p where p.resp is null and p.model = ? limit 40000', [model])
        data = [*conn]

    
    dt = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    os.makedirs(config.data_path('batches'), exist_ok=True)
    with open(loc:=os.path.join(config.data_path('batches'), f'batch_sc_{model.replace("/", "")}_{dt}.jsonl'), 'w') as f:
        for sc_p in data:
            s = {
                'custom_id': sc_p['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                'body': {**sc_p['body']},
                
            }
            if model != 'gpt-4o-mini':
                s['body']['max_completion_tokens'] = 2000
                s['body']['max_tokens'] = 2000

            f.write(json.dumps(s)+'\n')

    return loc


def prepare_confidence_batches(model:str) -> str:
    with mdb.sqliteDB(model, as_dict = True) as conn:
        conn.execute('''
        select p.* from problem_confidence_responses p 
        where p.resp is null and p.model = ?
        limit 35000''', [model])
        data = [*conn]

    
    dt = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    os.makedirs(config.data_path('batches'), exist_ok=True)
    with open(loc:=os.path.join(config.data_path('batches'), f'batch_confidence_{model.replace("/", "")}_{dt}.jsonl'), 'w') as f:
        for confidence_p in data:
            s = {
                'custom_id': confidence_p['id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                'body': {**confidence_p['body']},
            }
            if model != 'gpt-4o-mini':
                s['body']['max_completion_tokens'] = 2000
                s['body']['max_tokens'] = 2000

            f.write(json.dumps(s)+'\n')

    return loc


def fallback_json_parser_sc(resp:str) -> None:
    if(answer:=re.findall(r'''(?<="answer":)\s*[\."\w']+''', resp, re.I)):
        return {'answer': answer[0].strip().strip('''"''').lstrip('''"'''), 'reasoning': ""}

    if (answer:=re.findall(r'option [A-Z]{1}', resp)):
        return {'answer': answer[-1].strip('option '), 'reasoning': ""}
    
    if (answer:=re.findall(r'\W[A-Z]{1}\)', resp)):
        return {'answer': answer[-1][1:-1], 'reasoning': ""}

    try:
        js = re.findall(r'```json[\w\W]+```', resp)[0].lstrip('```json\n').rstrip('\n```')
        try:
            return ast.literal_eval(js)
        except:
            for i in range(0, 5):
                try:
                    return ast.literal_eval(js[:-i])
                except:
                    pass
    
    except:
        
        pass


def fallback_json_parser_confidence(resp:str) -> None:
    if(answer:=re.findall(r'''(?<="confidence":)\s*[\."\w']+''', resp, re.I)):
        return {'confidence': answer[0].strip().strip('''"''').lstrip('''"''')}

    if (answer:=re.findall(r'0\.\d+|(?<=\W)1(?=\W)|(?<=\W)0(?<=\W)', resp)):
        return {'confidence': answer[-1]}

    try:
        js = re.findall(r'```json[\w\W]+```', resp)[0].lstrip('```json\n').rstrip('\n```')
    
        try:
            return ast.literal_eval(js)
        except:
            for i in range(1, 3):
                try:
                    return ast.literal_eval(js[:-i])
                except:
                    pass

    except:
        pass


def import_sc_batch(model:str, batch_f_name:str) -> None:
    with open(os.path.join(config.data_path('batches'), batch_f_name)) as f:
        data = [json.loads(i.strip('\n')) for i in f]

    MODEL = llm.LLM(None)
    for ind in range(0, len(data), 500):
        with mdb.sqliteDB(model, as_dict = True) as conn:
            for i in data[ind:ind+500]:
                _id = i['custom_id']
                resp = i['response']['body']['choices'][0]['message']['content']
                try:
                    m = MODEL.parse_json_response(resp)
                except:
                    
                    m = fallback_json_parser_sc(resp)

                if m is not None:
                    conn.execute(
                        '''
                        update problem_sc_responses set resp = ?, added = datetime('now')
                        where id = ?
                        ''', [{'json': m, 'resp': resp}, _id]
                    )

            conn.commit()


def import_confidence_batch(model:str, batch_f_name:str) -> None:
    with open(os.path.join(config.data_path('batches'), batch_f_name)) as f:
        data = [json.loads(i.strip('\n')) for i in f]

    MODEL = llm.LLM(None)
    for ind in range(0, len(data), 500):
        with mdb.sqliteDB(model, as_dict = True) as conn:
            for i in data[ind:ind+500]:
                _id = i['custom_id']
                resp = i['response']['body']['choices'][0]['message']['content']
                try:
                    m = MODEL.parse_json_response(resp)
                    
                except:
                    m = fallback_json_parser_confidence(resp)

                if m is not None:
                    conn.execute(
                            '''
                            update problem_confidence_responses set resp = ?, added = datetime('now')
                            where id = ?
                            ''', [{'json': m}, _id]
                        )

            conn.commit()


def initialize_sc_vector_rows(model:str) -> None:
    with mdb.sqliteDB(model, as_dict = True) as conn:
        conn.execute('''
        select p.* from problem_sc_responses p
        left join problem_sc_response_vectors v on p.id = v.sc_id
        where v.id is null and p.resp is not null and p.model = ?
        ''', [model])
    
        data = [*conn]

    with mdb.sqliteDB(model, as_dict = True) as conn:
        for sc in data:
            conn.execute('''insert into problem_sc_response_vectors values 
                        (?, ?, null)''', [gen_random_uuid(), sc['id']])
        
        conn.commit()


def prepare_vector_batches(model:str) -> str:
    with mdb.sqliteDB(model, as_dict = True) as conn:
        conn.execute('''
        select v.*, s.resp from problem_sc_response_vectors v
        join problem_sc_responses s on s.id = v.sc_id
        where s.model = ? and v.vector is null
        limit 48000
        ''', [model])
        data = [*conn]

    dt = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    os.makedirs(config.data_path('batches'), exist_ok=True)
    with open(loc:=os.path.join(config.data_path('batches'), f'batch_vector_{model.replace("/", "")}_{dt}.jsonl'), 'w') as f:
        for sc in data:
            s = {
                'custom_id': sc['id'],
                "method": "POST",
                "url": "/v1/embeddings",
                'body': {
                    "input": llm.truncate_embedding_text(sc['resp']['resp']),
                    "model": "text-embedding-3-small",
                    "encoding_format": "float",
                }
            }
            f.write(json.dumps(s)+'\n')

    return loc


def import_vector_batch(model:str, batch_f_name:str) -> None:
    with open(os.path.join(config.data_path('batches'), batch_f_name)) as f:
        data = [json.loads(i.strip('\n')) for i in f]

    for ind in range(0, len(data), 500):
        with mdb.sqliteDB(model, as_dict = True) as conn:
            for i in data[ind:ind+500]:
                _id = i['custom_id']
                vector = i['response']['body']['data'][0]['embedding']
                conn.execute(
                    '''
                    update problem_sc_response_vectors set vector = ?
                    where id = ?
                    ''', [vector, _id]
                )

            conn.commit()


def migrate_sampled(src_model:str, dst_model:str, dset = None) -> None:
    '''Copy sampled problems from one model's database to another, so every model answers the same questions.'''
    with mdb.sqliteDB(src_model, as_dict = True) as conn:
        if dset is None:
            conn.execute('select * from sampled_problems', [])
        else:
            conn.execute('select * from sampled_problems where name = ?', [dset.NAME])
        
        data = [*conn]

    with mdb.sqliteDB(dst_model, as_dict = True) as conn:
        for i in data:
            conn.execute('insert into sampled_problems values (?, ?, ?, ?)', [*i.values()])
        
        conn.commit()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['init-db', 'sample', 'migrate', 'expand-sc', 'prepare-sc-batch',
                                    'import-sc-batch', 'expand-confidence', 'prepare-confidence-batch',
                                    'import-confidence-batch', 'init-vectors', 'prepare-vector-batch',
                                    'import-vector-batch'])
    p.add_argument('--model', required=True, choices=[*mdb.DB_FILES])
    p.add_argument('--dataset', choices=[*DATASETS], help='default: all datasets')
    p.add_argument('--num', type=int, default=1000, help='problems to sample per dataset (0 = all; the paper used all of GPQA)')
    p.add_argument('--sc-num', type=int, default=20, help='self-consistency samples per problem')
    p.add_argument('--temperature', type=float, help='default: 0.7 for sampling, 0.3 for the critic')
    p.add_argument('--src-model', choices=[*mdb.DB_FILES], help='source database for "migrate"')
    p.add_argument('--batch-file', help='provider output file in $veccisc_data_path/batches, for import-* steps')
    args = p.parse_args()

    dsets = [DATASETS[args.dataset]] if args.dataset else d_sets.DS
    model = args.model

    if args.step == 'init-db':
        print(mdb.init_db(model))
    elif args.step == 'sample':
        for ds in dsets:
            sample_problems(model, args.num or None, ds)
    elif args.step == 'migrate':
        for ds in dsets:
            migrate_sampled(args.src_model, model, ds)
    elif args.step == 'expand-sc':
        for ds in dsets:
            expand_sampled_to_sc(ds, args.sc_num, {'model': model, 'temperature': args.temperature or 0.7})
    elif args.step == 'prepare-sc-batch':
        print(prepare_sc_batches(model))
    elif args.step == 'import-sc-batch':
        import_sc_batch(model, args.batch_file)
    elif args.step == 'expand-confidence':
        sc_to_confidence(model, {'model': model, 'temperature': args.temperature or 0.3})
    elif args.step == 'prepare-confidence-batch':
        print(prepare_confidence_batches(model))
    elif args.step == 'import-confidence-batch':
        import_confidence_batch(model, args.batch_file)
    elif args.step == 'init-vectors':
        initialize_sc_vector_rows(model)
    elif args.step == 'prepare-vector-batch':
        print(prepare_vector_batches(model))
    elif args.step == 'import-vector-batch':
        import_vector_batch(model, args.batch_file)


if __name__ == '__main__':
    main()
