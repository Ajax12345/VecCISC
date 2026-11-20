"""VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace
Clustering and Candidate Answer Selection (Findings of ACL 2026).

Evaluation code: Self-Consistency (SC), CISC and VecCISC (KMeans / HAC / random
ablation), hyperparameter tuning, and the LaTeX tables reported in the paper.
Run `python veccisc.py --help` for the available commands.
"""

import mdb, d_sets, config
import sklearn.cluster, math
import collections, json, os
import typing, random, re
import argparse
from scipy.spatial.distance import cosine

MODELS = ['gpt-4o-mini', 
          'llama-3.1-8b-instant', 
          'Qwen/Qwen2.5-7B-Instruct-Turbo', 
          'mistralai/Mistral-7B-Instruct-v0.1', 
          'llama-3.3-70b-versatile']

ROW_NAMES = {'cluster_num':'Average Call Reduction', 
     'sc': 'SC Baseline', 
     'weights': 'CISC', 
     'weights_random': 'VecCISC (random)',
     'cluster':'VecCISC + KMeans',  
     'cluster_hac':'VecCISC + HAC',
}

ROW_NAMES_TOKENS = {
    'token_usage_CISC': 'CISC',
    'token_usage_VecCISC_random': 'VecCISC (rand-trace)',
    'token_usage_VecCISC': 'VecCISC (min-centroid)'
}

MODEL_COSTS = { #as of 2025-11-10 19:40:35
    'gpt-4o-mini': 0.15,
    'llama-3.1-8b-instant': 0.05,
    'Qwen/Qwen2.5-7B-Instruct-Turbo': 0.3,
    'mistralai/Mistral-7B-Instruct-v0.1': 0.2,
    'llama-3.3-70b-versatile': 0.59
}

MODELS_TO_NAME = {
    'gpt-4o-mini': 'GPT 4o-mini',
    'llama-3.1-8b-instant': 'Llama3.1 8B',
    'Qwen/Qwen2.5-7B-Instruct-Turbo': 'Qwen2.5 7B',
    'mistralai/Mistral-7B-Instruct-v0.1': 'Mistral 7B',
    'llama-3.3-70b-versatile': 'Llama3.3 70B'
}

DS_TO_NAME = {
    'gpaq': 'GPQA',
    'aqua_rat': 'AQuA-RAT',
    'arch_challenging': 'ARCH-Challenging',
    'mmlu_pro': 'MMLU-Pro',
    'commonsense_qa': 'CommonsenseQA'
}


def load_data(dset:d_sets, model:str, from_json:bool = True) -> dict:
    '''Load every sampled answer, critic confidence and trace embedding for (dset, model).

    Reads the cached export $veccisc_data_path/json_batches/<dataset>_<model>.json if it
    exists (and from_json is True); otherwise queries the model\'s SQLite database and
    writes that cache.
    '''
    problem_correct_answers = {}
    problem_results = collections.defaultdict(list)
    problem_src = {}
    full_path = config.data_path('json_batches', f'{dset.NAME}_{model.replace("/", "")}.json')
    if not from_json or not os.path.exists(full_path):
        with mdb.sqliteDB(model, as_dict=True) as conn:
            conn.execute('''
            select sp.id, 
                sp.name, 
                json_extract(sp.data, '$.question') as question,
                json_extract(sp.data, '$.correct_answer') as correct_answer,
                json_extract(json_extract(sc.resp, '$.json'), '$.answer') as answer,
                json_extract(sc.resp, '$.resp') as reasoning_trace,
                length(json_extract(cr.body, '$.messages[1].content')) / 4 as token_num,
                json_extract(json_extract(cr.resp, '$.json'), '$.confidence') as confidence,
                v.vector
            from sampled_problems sp
            join problem_sc_responses sc on sc.problem_id = sp.id
            join problem_confidence_responses cr on cr.sc_id = sc.id
            join problem_sc_response_vectors v on v.sc_id = sc.id
            where sc.resp is not null and sp.name = ?
            ''', [dset.NAME])
            for i in conn:
                problem_correct_answers[i['id']] = i['correct_answer']
                problem_results[i['id']].append(i)
                problem_src[i['id']] = [i['name'], i['question']]

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            json.dump(d:={'problem_results': problem_results, 
                    'problem_correct_answers': problem_correct_answers,
                    'problem_src': problem_src}, f)


    else:
        with open(full_path) as f:
            d = json.load(f)

    return d

def sc_token_usage() -> None:
    '''Average SC-sampling and CISC-critic prompt tokens per problem (requires the databases).'''
    d = collections.defaultdict(dict)
    totals = []
    for model in MODELS:
        for dset in d_sets.DS:
            with mdb.sqliteDB(model, as_dict=True) as conn:
                conn.execute('''
                select 
                    sp.id, 
                    sum(length(json_extract(sc.body, '$.messages[1].content'))/4) token_num_sc, 
                    sum(length(json_extract(cr.body, '$.messages[1].content'))/4) token_num_cisc,
                    count(*) c
                from sampled_problems sp
                join problem_sc_responses sc on sc.problem_id = sp.id
                join problem_confidence_responses cr on cr.sc_id = sc.id
                join problem_sc_response_vectors v on v.sc_id = sc.id
                where sc.resp is not null and v.vector is not null and sp.name = ?
                group by sp.id
                ''', [dset.NAME])
                k = [*conn]
                avg_usage_sc = sum(i['token_num_sc'] for i in k)/len(k) if k else 0
                avg_usage_cisc = sum(i['token_num_cisc'] for i in k)/len(k) if k else 0
                d[model][dset.NAME] = {
                    'sc': avg_usage_sc,
                    'cisc': avg_usage_cisc
                }
                totals.append(d[model][dset.NAME])

    with open(os.path.join(config.SUPPORTING_DIR, 'sc_token_usage.json'), 'w') as f:
        json.dump(dict(d), f, indent=4)

    print(totals)

def token_wise_savings() -> None:
    '''Print the average total (SC + critic) and critic-only token savings of VecCISC over CISC.'''
    with open(os.path.join(config.SUPPORTING_DIR, 'sc_token_usage.json')) as f:
        usage = json.load(f)

    for i in ['results_kmeans_token_usage.json', 'results_hac_token_usage.json']:
        results = []
        r1 = []
        with open(os.path.join(config.RESULTS_DIR, i)) as f:
            data = json.load(f)
            for model, d in data.items():
                for ds, vals in d.items():
                    base_sc = usage[model][ds]['sc']
                    veccisc = min(vals['token_usage_VecCISC']['avg'], 
                                  vals['token_usage_VecCISC_random']['avg'])
                    
                    base_cisc = vals['token_usage_CISC']['avg']
                    results.append((base_sc + base_cisc - veccisc)/(base_sc + base_cisc))
                    r1.append((base_cisc - veccisc)/base_cisc)
        print(sum(results)/len(results))
        print(sum(r1)/len(r1))

def perform_sc(data:dict, config_:dict, suppress_cluster:bool = False) -> dict:
    '''Evaluate SC, CISC and VecCISC on one (dataset, model) result set.

    config_ keys:
        T        softmax temperature applied to critic confidences (CISC weighting)
        C        number of clusters K for KMeans (also the sample size for the random ablation)
        C_HAC    number of clusters K for HAC
        D        HAC distance_threshold (None when n_clusters is set)
        cluster  'kmeans' (default) or 'HAC'

    Returns accuracies ('sc', 'weights' = CISC, 'cluster' = VecCISC with min-centroid
    representatives, 'cluster_abl' = VecCISC with random representatives,
    'weights_random' = VecCISC (random)), the average number of critic calls
    ('cluster_num') and average critic tokens ('token_usage_*').
    '''
    problem_correct_answers = data['problem_correct_answers']
    problem_results = data['problem_results']

    final_results = collections.defaultdict(int)
    total = 0

    def best_answer(d:dict) -> str:
        return max(d, key=lambda x:d[x])

    def best_entry(entry:typing.List[dict]) -> dict:
        # min-centroid: the trace closest (cosine) to the mean embedding of its cluster
        avg_vector = [sum(i)/len(i) for i in zip(*[b['vector'] for b in entry])]
        return min(entry, key=lambda x:cosine(x['vector'], avg_vector))

    def raw_weights_to_softmax(d:dict, T:float) -> dict:
        s_denom_sampled = sum(pow(math.e, j/T) for b in d.values() for j in b)
        return {a:sum(pow(math.e, j/T)/s_denom_sampled for j in b) for a, b in d.items()}

    def clip_weights(w:float) -> float:
        return min(round(w, 2), 1)

    for problem_id, sc_results in problem_results.items():

        answer_majority_count = collections.defaultdict(int)
        weights = collections.defaultdict(list)
        weights_and_vectors = collections.defaultdict(list)
        weights_from_clusters = collections.defaultdict(list)
        weights_from_clusters_random = collections.defaultdict(list)

        for I, j in enumerate(sc_results):
            try:
                answer_majority_count[j['answer']] += 1
                weights[j['answer']].append(clip_weights(float(j['confidence'])))
                final_results['token_usage_CISC'] += int(j['token_num'])
                weights_and_vectors[j['answer']].append({
                    'id': I,
                    'confidence': clip_weights(float(j['confidence'])),
                    'token_num': int(j['token_num']),
                    'vector': j['vector'],
                })

            except:
                pass

        weights_random = {a:random.sample(b, min(config_['C'], len(b))) for a, b in weights.items()}

        if not suppress_cluster:
            for answer, entries in weights_and_vectors.items():
                if len(entries) == 1:
                    weights_from_clusters[answer] = [entries[0]['confidence']]
                    continue

                # cluster the reasoning-trace embeddings within each answer group
                if config_.get('cluster', 'kmeans') == 'kmeans':
                    ac = sklearn.cluster.KMeans(n_clusters=min(config_['C'], len(entries)))

                elif config_['cluster'] == 'HAC':
                    ac = sklearn.cluster.AgglomerativeClustering(
                        n_clusters=min(config_['C_HAC'], len(entries)),
                        metric = 'cosine',
                        linkage='average',
                        distance_threshold = config_['D']
                    )

                ac.fit([i['vector'] for i in entries])
                sub_clusters = collections.defaultdict(list)
                for label, entry in zip(ac.labels_, entries):
                    sub_clusters[label].append(entry)

                best_entry_selections = [best_entry(j) for j in sub_clusters.values()]
                random_entry_selections = [random.choice(j) for j in sub_clusters.values()]

                final_results['token_usage_VecCISC'] += sum(j['token_num'] for j in best_entry_selections)
                final_results['token_usage_VecCISC_random'] += sum(j['token_num'] for j in random_entry_selections)

                weights_from_clusters[answer] = [j['confidence'] for j in best_entry_selections]
                weights_from_clusters_random[answer] = [j['confidence'] for j in random_entry_selections]

            final_results['cluster_num'] += sum(len(b) for b in weights_from_clusters.values())

        weights = raw_weights_to_softmax(weights, config_['T'])

        if not suppress_cluster:
            weights_from_clusters = raw_weights_to_softmax(weights_from_clusters, config_['T'])
            weights_from_clusters_random = raw_weights_to_softmax(weights_from_clusters_random, config_['T'])

        correct_answer = problem_correct_answers[problem_id]

        final_results['sc'] += correct_answer == best_answer(answer_majority_count)
        final_results['weights'] += correct_answer == best_answer(weights)
        if not suppress_cluster:
            final_results['cluster'] += correct_answer == best_answer(weights_from_clusters)
            if weights_from_clusters_random:
                final_results['cluster_abl'] += correct_answer == best_answer(weights_from_clusters_random)

        final_results['weights_random'] += correct_answer == best_answer(weights_random)

        total += 1

    return {a:b/total for a, b in final_results.items()}


def tune_T(model:str, loc:str) -> dict:
    k = {}
    for u in d_sets.DS:
        data = load_data(u, model)
        T = 0.01
        options = []
        while T < 10:
            options.append([T, perform_sc(data,
                                           {'D': 0.02, 'T':T, 'C': 5},
                                        suppress_cluster=True)])
            T += 0.01
        
        k[u.NAME] = {'T':max(options, key=lambda x:x[1]['weights'])[0]}
    
    with open(os.path.join(config.RESULTS_DIR, loc)) as f:
        hp = json.load(f)

    hp[model] = k

    with open(os.path.join(config.RESULTS_DIR, loc), 'w') as f:
        json.dump(hp, f)

    return k


def tune_C(model:str, loc:str) -> dict:

    with open(os.path.join(config.RESULTS_DIR, loc)) as f:
        hp = json.load(f)

    for u in d_sets.DS:
        print(u.NAME)
        data = load_data(u, model)

        options = [(C, perform_sc(data, {'D': 0.02, 'T':hp[model][u.NAME]['T'], 
                                         'C': C, 
                                }))
                   for C in range(2, 20)]
        
        hp[model][u.NAME]['C'] = max(options, key=lambda x:x[1]['cluster'])[0]

    with open(os.path.join(config.RESULTS_DIR, loc), 'w') as f:
        json.dump(hp, f, indent=4)

def tune_C_HAC(model:str, loc:str) -> dict:
    with open(os.path.join(config.RESULTS_DIR, loc)) as f:
        hp = json.load(f)

    for u in d_sets.DS:
        data = load_data(u, model, from_json=True)

        options = [(C, perform_sc(data, {'D': None, 
                        'T':hp[model][u.NAME]['T'],
                        'C': C, 
                        'C_HAC': C, 
                        'cluster': 'HAC'}))

                   for C in range(2, 20)]
        
        hp[model][u.NAME]['C_HAC'] = max(options, key=lambda x:x[1]['cluster'])[0]

    with open(os.path.join(config.RESULTS_DIR, loc), 'w') as f:
        json.dump(hp, f, indent=4)
    
def produce_results(model:str, params:dict) -> dict:
    '''Run each method 10 times per dataset with tuned hyperparameters; record max/min/avg.'''
    with open(os.path.join(config.RESULTS_DIR, params['hp'])) as f:
        hp = json.load(f)
    

    k = {}
    for u in d_sets.DS:
        data = load_data(u, model, from_json=True)
        options = collections.defaultdict(list)
        for _ in range(10):
            for a, b in perform_sc(data, {'D': None, 
                'T':hp[model][u.NAME]['T'], 
                'C': hp[model][u.NAME]['C'],
                'C_HAC': hp[model][u.NAME]['C_HAC'],
                'cluster': params['cluster']}).items():
                
                options[a].append(b)

        
        k[u.NAME] = {a:{'max':max(b), 'min': min(b), 'avg': sum(b)/len(b)} 
            for a, b in options.items()}
    
    return k


def load_results(f_name:str) -> dict:
    with open(os.path.join(config.RESULTS_DIR, f_name)) as f:
        data = json.load(f)

    m = collections.defaultdict(dict)
    for model, d in data.items():
        for ds, results in d.items():
            m[ds][model] = {j:{v:round(h, 3)*(100 if h < 1 else 1) 
                    for v, h in k.items()} 
                            for j,k in results.items()}
            
    return m

def reformat_data(data:dict, ROW_NAMES:dict = ROW_NAMES) -> dict:
    d = [{'dataset': DS_TO_NAME[ds], 
          'rows':[[ROW_NAMES[r], [[results[model][r]['max'], results[model][r]['avg']] for model in MODELS if r in results[model]]]
                   for r in ROW_NAMES]} 

         for ds, results in data.items()]
    
    return d

def generate_performance_table(data:dict) -> dict:
    d = reformat_data(data)
    
    def fm(u, i):
        l = sorted(set(i))
        if u == l[-1]:
            return '\\textbf{'+str(u)+'}'
        
        if u == l[-2]:
            return '\\underline{'+str(u)+'}'
        
        return str(u)
    
    print(r'''
\begin{table*}[htb]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{l|l|cccccc}
\toprule
\textbf{Dataset} & \textbf{Method} & \textbf{GPT 4o-mini} & \textbf{Llama3.1 8B} & \textbf{Qwen2.5 7B} & \textbf{Mistral 7B}  & \textbf{Llama3.3 70B} \\ \midrule
& & \multicolumn{2}{c}{Best / Average} & \multicolumn{2}{c}{Best / Average} &  \\ 
\cline{3-7}
    ''')
    
    for w, j in enumerate(d):
        print('\\multirow{4}{*}{'+j['dataset']+'}')
        final_rows = j['rows'][1:]
        block = [*zip(*[[round(m, 3) for n in r for m in n] for _,r in final_rows])]
        b = [*zip(*[[fm(u, i) for u in i] for i in block])]
        for (row_name, _), row in zip(final_rows, b):
            if row_name not in ['SC Baseline', 'CISC', 'VecCISC (random)']:
                print('& \\cellcolor{gray!20} '+f'{row_name} & ' + ' & '.join('\\cellcolor{gray!20}'+f'{row[i]} / {row[i+1]}' for i in range(0, len(row), 2)) + ' \\\\')
            else:
                print('& '+f'{row_name} & ' + ' & '.join(f'{row[i]} / {row[i+1]}' for i in range(0, len(row), 2)) + ' \\\\')

        if w < len(d) - 1:
            print('\\midrule')


    print(r'''
\bottomrule
\end{tabular}
}
\caption{Comparison between baselines and our methods across different LLMs and datasets. In each column, the best results are in \textbf{bold}, and the second-best results are \underline{underlined}. VecCISC's clustering approach maintains or improves upon CISC's performance across models and datasets, with \textbf{VecCISC + HAC}'s average performance accounting for the most consistent method. Experiments on each $(dataset, model)$ combination where performed 10 times for each method, and the best and average scores were recorded.}
\label{tab:performance}
\end{table*}
    ''')

def generate_token_reduction_table(name:str, tab_name:str, data:dict) -> None:
    d = reformat_data(data, ROW_NAMES=ROW_NAMES_TOKENS)
    
    def fm(u, i):
        l = sorted(set(i))
        if u == l[0]:
            return '\\textbf{'+str(u)+'}'
        
        if u == l[1]:
            return '\\underline{'+str(u)+'}'
        
        return str(u)
    
    print(r'''
\begin{table*}[htbp]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{l|l|cccccc}
\toprule
\textbf{Dataset} & \textbf{Method} & \textbf{GPT 4o-mini} & \textbf{Llama3.1 8B} & \textbf{Qwen2.5 7B} & \textbf{Mistral 7B}  & \textbf{Llama3.3 70B} \\ \midrule
\cline{3-7}
    ''')

    all_token_reductions = []

    for w, j in enumerate(d):
        print('\\multirow{3}{*}{'+j['dataset']+'}')
        final_rows = j['rows']
        block = [*zip(*[[round(m, 1) for n in r for m in n] for _,r in final_rows])]
        all_token_reductions.extend([round(100*(max(t)-min(t))/max(t), 2) for t in block])
        b = [*zip(*[[fm(u, i) for u in i] for i in block])]
        for (row_name, _), row in zip(final_rows, b):
            if row_name not in ['SC Baseline', 'CISC', 'VecCISC (random)']:
                print('& \\cellcolor{gray!20} '+f'{row_name} & ' + ' & '.join('\\cellcolor{gray!20}'+f'{row[i+1]}' for i in range(0, len(row), 2)) + ' \\\\')
            else:
                print('& '+f'{row_name} & ' + ' & '.join(f'{row[i+1]}' for i in range(0, len(row), 2)) + ' \\\\')


        if w < len(d) - 1:
            print('\\midrule')
    
    all_token_reductions = [i-(i*0.06) for i in all_token_reductions]
    avg_token_reduction = round(sum(all_token_reductions)/len(all_token_reductions), 2)
    print('avg token_reduction (accounting for CISC prompt diff)', avg_token_reduction)

    print(r'''
\bottomrule
\end{tabular}
}
\caption{'''+f'{name} token reductions with cluster candidate selection variants, averaged across 10 runs.' +r''' In each column, the best results are in \textbf{bold}, and the second-best results are \underline{underlined}.'''+f'{name} records a {avg_token_reduction}\\% reduction in tokens expended on '+'$LLM_{critic}$ calls.'+r'''}
\label{tab:'''+tab_name+r'''_token_reduction}
\end{table*}
    ''')
    

def budget_improvement_table(name:str, tab_name:str, data:dict) -> None:
    d = reformat_data(data)
    print(r'''
\begin{table*}[ht]
\centering
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccc}
\hline
\textbf{Dataset} & \textbf{Budget} & \textbf{GPT-4o mini} & \textbf{Llama 3.1 8B} & \textbf{Qwen2.5 7B} & \textbf{Mistral 7B}  & \textbf{Llama3.3 70B} \\
\hline
    ''')

    all_reductions = []
    for w, j in enumerate(d):
        print('\\multirow{1}{*}{'+j['dataset']+'}')
        for _, r in j['rows'][:1]:
            v = [a for a,_ in r]
            all_reductions.extend([round((1-j/20)*100,2) for j in v])
            l = " & ".join([str(round(j,2)).ljust(5,'0')+ f'{{\\scriptsize  (-{str(round((1-j/20)*100,2)).ljust(5,"0")}\\%) }}' for j in v])
            print(f'''& 20 & {l} \\\\'''+\
'''\\hline''')

    avg_reduction = round(sum(all_reductions)/len(all_reductions), 2)
    print(r'''
\end{tabular}
\caption{'''+f'{name} confidence evaluator budget reductions, averaged across 10 runs. '+r'''\textbf{min-centroid} cluster candidate selection was used for all experiments. '''+f'{name} records a {avg_reduction}\\% reduction in '+'$LLM_{critic}$ calls.' +r'''}
\label{tab:llm_budget_reductions_'''+tab_name+r'''}
\end{table*}  
    ''')


def generate_all_tables() -> None:
    data = load_results('results_full_variance.json')
    hac_data = load_results('results_hac_full_variance.json')
    data = {a:{j:{**k, 'cluster_hac': hac_data[a][j]['cluster']} for j, k in b.items()} 
                for a, b in data.items()}
    
    
    generate_performance_table(data)
    #print('-'*40)
    budget_improvement_table('\\textbf{VecCISC + KMeans}', 'kmeans', data)
    #print('-'*40)
    budget_improvement_table('\\textbf{VecCISC + HAC}', 'hac', hac_data)

    #print('-'*40)

    token_usage_kmeans = load_results('results_kmeans_token_usage.json')
    token_usage_hac = load_results('results_hac_token_usage.json')

    generate_token_reduction_table('\\textbf{VecCISC + KMeans}', 'kmeans', token_usage_kmeans)
        
    generate_token_reduction_table('\\textbf{VecCISC + HAC}', 'hac', token_usage_hac)

def tune(loc:str, params:dict) -> None:
    '''Grid-search T, C (KMeans K) and/or C_HAC (HAC K) for every model; writes supporting/results/<loc>.'''
    if params['T']:
        print('tuning T')
        for model in MODELS:
            print(tune_T(model, loc))

    if params['C']:
        print('tuning C')
        for model in MODELS:
            tune_C(model, loc)

    if params['C_HAC']:
        print('tuning C_HAC')
        for model in MODELS:
            tune_C_HAC(model, loc)


def produce_all_results(params:dict) -> None:
    '''Run produce_results for every model and write supporting/results/<params["f_name"]>.'''
    d = {i:produce_results(i, params) for i in MODELS}

    with open(os.path.join(config.RESULTS_DIR, params['f_name']), 'w') as f:
        json.dump(d, f, indent=4)


def display_case_study() -> None:
    '''Print the LaTeX for Appendix A.3 (Case Study 1) from supporting/case_study.json.'''
    with open(os.path.join(config.SUPPORTING_DIR, 'case_study.json')) as f:
        data = json.load(f)

    c = 0
    for i, a in enumerate(sorted(data), 1):
        print(r"""\noindent\rule{\columnwidth}{0.5pt}
\textbf{Cluster ID}: """+str(i)+r""" \\
\textbf{\# of Traces}: """+str(len(data[a]))+r"""\\
\noindent\rule{\columnwidth}{0.5pt}""")
        for t in data[a]:
            print(r"""\textbf{Trace """+str(c:=c+1)+"""}: """+ re.sub('\n+', ' ', t)+r"""\\""")


def full_token_usage_critic_prompt() -> None:
    '''Compare average critic prompt length for VecCISC vs. the original CISC prompt format
    (the source of the 6% adjustment in generate_token_reduction_table).'''
    veccisc, cisc = [], []
    for model in MODELS:
        with mdb.sqliteDB(model, as_dict=True) as conn:
            conn.execute('''
            select json_extract(p.body, '$.messages[1].content') prompt
            from problem_confidence_responses p
            ''', [])

            for p in conn:
                prompt = p['prompt']
                cisc_prompt_base = prompt[len('## Instructions\n\nYou are a reviewer of answers given to multiple-choice questions. Given a multiple-choice question, a set of answer options for the question, a given answer, and the reasoning that is associated with the given answer, rate your confidence that the given answer is correct.\n\nTo rate your confidence, provide a value on a scale of 0 to 1. The more confident you are that the provided answer is correct for the question, the closer to (or equal) to 1 the confidence score should be.'):]
                veccisc.append(len(prompt)//4)
                cisc.append(len('''Now I will rate my confidence in the proposed answer on a scale of 0-100.
Proposed confidence: (''' + cisc_prompt_base)//4)
    
        break
    
    print(sum(veccisc)/len(veccisc), sum(cisc)/len(cisc))


def main() -> None:
    p = argparse.ArgumentParser(description='VecCISC experiments.')
    sub = p.add_subparsers(dest='command', required=True)

    sub.add_parser('tables', help='print LaTeX for Tables 1-5 from supporting/results/')

    t = sub.add_parser('tune', help='grid-search hyperparameters (needs $veccisc_data_path)')
    t.add_argument('--hp-file', default='hyperparameters.json', help='file in supporting/results/ to update')
    t.add_argument('--params', nargs='+', choices=['T', 'C', 'C_HAC'], default=['T', 'C', 'C_HAC'])

    r = sub.add_parser('results', help='run all methods 10x per (dataset, model) (needs $veccisc_data_path)')
    r.add_argument('--cluster', choices=['kmeans', 'HAC'], required=True)
    r.add_argument('--hp-file', default='hyperparameters.json')
    r.add_argument('--out', required=True, help='output file name in supporting/results/')

    sub.add_parser('sc-token-usage', help='recompute supporting/sc_token_usage.json (needs the databases)')
    sub.add_parser('token-savings', help='print average total and critic-only token savings')
    sub.add_parser('critic-prompt-tokens', help='compare critic prompt lengths, VecCISC vs. CISC (needs the databases)')
    sub.add_parser('case-study', help='print LaTeX for Appendix A.3 (Case Study 1)')

    args = p.parse_args()

    if args.command == 'tables':
        generate_all_tables()
    elif args.command == 'tune':
        tune(args.hp_file, {k: k in args.params for k in ['T', 'C', 'C_HAC']})
    elif args.command == 'results':
        produce_all_results({'cluster': args.cluster, 'hp': args.hp_file, 'f_name': args.out})
    elif args.command == 'sc-token-usage':
        sc_token_usage()
    elif args.command == 'token-savings':
        token_wise_savings()
    elif args.command == 'critic-prompt-tokens':
        full_token_usage_critic_prompt()
    elif args.command == 'case-study':
        display_case_study()


if __name__ == '__main__':
    main()
