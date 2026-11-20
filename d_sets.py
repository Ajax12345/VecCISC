import typing, json
import random, string, os
import config


def _loc(name:str) -> str:
    return os.path.join(config.DATASETS_DIR, name)


def load_datasets() -> None:
    """Download the benchmark splits used in the paper into datasets/.

    GPQA is gated on the Hugging Face Hub: accept its terms at
    https://huggingface.co/datasets/Idavidrein/gpqa and run `hf auth login` first.
    """
    from datasets import load_dataset  # Hugging Face `datasets`

    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    load_dataset("deepmind/aqua_rat", "raw")['train'].to_json(_loc('aqua_rat_train.json'))
    load_dataset("tau/commonsense_qa")['validation'].to_json(_loc('commonsense_qa_validation.json'))
    load_dataset("allenai/ai2_arc", "ARC-Challenge")['test'].to_json(_loc('arch_challenge_test.json'))
    load_dataset("TIGER-Lab/MMLU-Pro")['test'].to_json(_loc('mmlu_pro_test.json'))

    for i in ['gpqa_diamond', 'gpqa_extended', 'gpqa_main']:
        load_dataset("Idavidrein/gpqa", i)['train'].to_json(_loc(f'{i}.json'))

    if not os.path.exists(GPQA.LOC):
        format_gpqa()

class ROW:
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'question': self.question,
            'option_mapping': self.option_mapping,
            'options': self.options,
            'correct_answer': self.correct_answer
        }

class AQUA_RAT_ROW(ROW):
    def __init__(self, id:int, row:dict) -> None:
        self.id = id
        self.row = row

    @property
    def question(self) -> str:
        return self.row['question']
    
    @property
    def option_mapping(self) -> dict:
        mapping = {}
        for i in self.row['options']:
            a, *b = i.split(')')
            mapping[a] = ')'.join(b)

        return mapping
    
    @property
    def options(self) -> str:
        return '\n'.join(self.row['options'])
    
    @property
    def correct_answer(self) -> str:
        return self.row['correct']

class AQUA_RAT:
    LOC = _loc('aqua_rat_train.json')
    NAME = 'aqua_rat'
    def __init__(self, data:typing.List[AQUA_RAT_ROW]) -> None:
        self.data = data
    
    def __iter__(self) -> typing.Iterator:
        yield from self.data

    def __getitem__(self, ind:int) -> AQUA_RAT_ROW:
        return self.data[ind]
    
    def sample(self, num:int) -> typing.List[AQUA_RAT_ROW]:
        return random.sample(self.data, num)

    @classmethod
    def load(cls) -> 'AQUA_RAT':
        with open(cls.LOC) as f:
            data = [json.loads(i) for i in f]

        return cls([AQUA_RAT_ROW(i, a) for i, a in enumerate(data)])

class CQA_ROW(ROW):
    def __init__(self, id:int, row:dict) -> None:
        self.id = id
        self.row = row
    
    @property
    def question(self) -> str:
        return self.row['question']
    
    @property
    def option_mapping(self) -> dict:
        return dict(zip(self.row['choices']['label'], self.row['choices']['text']))

    @property
    def options(self) -> str:
        return '\n'.join(f'{a}) {b}' for a, b in zip(self.row['choices']['label'], self.row['choices']['text']))
    
    @property
    def correct_answer(self) -> str:
        return self.row['answerKey']

class CQA:
    LOC = _loc('commonsense_qa_validation.json')
    NAME = 'commonsense_qa'
    def __init__(self, data:typing.List[CQA_ROW]) -> None:
        self.data = data

    def sample(self, num:int) -> typing.List[CQA_ROW]:
        return random.sample(self.data, num)
    
    def __getitem__(self, ind:int) -> CQA_ROW:
        return self.data[ind]

    @classmethod
    def load(cls) -> 'CQA':
        with open(cls.LOC) as f:
            data = [json.loads(i) for i in f]

        return cls([CQA_ROW(i, a) for i, a in enumerate(data)])
    
class ARCH_ROW(ROW):
    def __init__(self, id:int, row:dict) -> None:
        self.id = id
        self.row = row
    
    @property
    def question(self) -> str:
        return self.row['question']
    
    @property
    def option_mapping(self) -> dict:
        return dict(zip(self.row['choices']['label'], self.row['choices']['text']))
    
    @property
    def options(self) -> str:
        return '\n'.join(f'{a}) {b}' for a, b in zip(self.row['choices']['label'], self.row['choices']['text']))
    
    @property
    def correct_answer(self) -> str:
        return self.row['answerKey']

class ARCH:
    LOC = _loc('arch_challenge_test.json')
    NAME = 'arch_challenging'
    def __init__(self, data:typing.List[ARCH_ROW]) -> None:
        self.data = data

    def sample(self, num:int) -> typing.List[ARCH_ROW]:
        return random.sample(self.data, num)
    
    def __getitem__(self, ind:int) -> ARCH_ROW:
        return self.data[ind]

    @classmethod
    def load(cls) -> 'ARCH':
        with open(cls.LOC) as f:
            data = [json.loads(i) for i in f]

        return cls([ARCH_ROW(i, a) for i, a in enumerate(data)])

class MMLU_PRO_ROW(ROW):
    def __init__(self, id:int, row:dict) -> None:
        self.id = id
        self.row = row
    
    @property
    def question(self) -> str:
        return self.row['question']
    
    def __getitem__(self, k:str) -> str:
        return self.row[k]
    
    @property
    def option_mapping(self) -> dict:
        return dict(zip(string.ascii_uppercase, self.row['options']))
    
    @property
    def options(self) -> str:
        return '\n'.join(f'{a}) {b}' for a, b in zip(string.ascii_uppercase, self.row['options']))
    
    @property
    def correct_answer(self) -> str:
        return self.row['answer']

class MMLU_PRO:
    LOC = _loc('mmlu_pro_test.json')
    NAME = 'mmlu_pro'
    def __init__(self, data:typing.List[MMLU_PRO_ROW]) -> None:
        self.data = data

    def sample(self, num:int) -> typing.List[MMLU_PRO_ROW]:
        return random.sample(self.data, num)
    
    def __getitem__(self, ind:int) -> MMLU_PRO_ROW:
        return self.data[ind]

    @classmethod
    def load(cls) -> 'MMLU_PRO':
        with open(cls.LOC) as f:
            data = [json.loads(i) for i in f]

        return cls([MMLU_PRO_ROW(i, a) for i, a in enumerate(data)])
    

class GPQA_ROW(ROW):
    def __init__(self, id:int, row:dict, mapping:dict) -> None:
        self.id = id
        self.row = row
        self.mapping = mapping
    
    @property
    def question(self) -> str:
        return self.row['question']
    
    def __getitem__(self, k:str) -> str:
        return self.mapping[k]
    
    @property
    def option_mapping(self) -> dict:
        return self.row['options']
    
    @property
    def options(self) -> str:
        return '\n'.join(f'{a}) {b}' for a, b in self.row['options'].items())
    
    @property
    def correct_answer(self) -> str:
        return self.row['correct_answer']

class GPQA:
    LOC = _loc('gpqa.jsonl')
    LOC_ORIGINAL = [_loc('gpqa_extended.json'),
                    _loc('gpqa_main.json'),
                    _loc('gpqa_diamond.json')]
    
    NAME = 'gpaq'
    def __init__(self, data:typing.List[GPQA_ROW]) -> None:
        self.data = data

    def sample(self, num:int) -> typing.List[GPQA_ROW]:
        return random.sample(self.data, min(num, len(self.data)))
    
    def __getitem__(self, ind:int) -> GPQA_ROW:
        return self.data[ind]

    @classmethod
    def load(cls) -> 'GPQA_ROW':
        mapping = {}
        for i in cls.LOC_ORIGINAL:
            with open(i) as f:
                for j in f:
                    r = json.loads(j)
                    mapping[r['Question'].strip()] = r

        with open(cls.LOC) as f:
            data = [json.loads(i) for i in f]

        return cls([GPQA_ROW(i, a, mapping[a['question'].strip()]) for i, a in enumerate(data)])


DS = [AQUA_RAT, CQA, ARCH, MMLU_PRO, GPQA]

def format_gpqa() -> None:
    """Merge the three GPQA subsets into datasets/gpqa.jsonl with shuffled, lettered options."""
    final_result = []
    seen_questions = []
    for i in ['gpqa_diamond', 'gpqa_extended', 'gpqa_main']:
        with open(_loc(f'{i}.json')) as f:
            data = [json.loads(i) for i in f]
        
    
        for j in data:
            if j['Question'] in seen_questions:
                continue
        
            seen_questions.append(j['Question'])

            options = [j['Correct Answer']]
            c = 0
            while (T:=f'Incorrect Answer {(c:=c+1)}') in j:
                options.append(j[T])

            assert len(options) > 1
            random.shuffle(options)
            d = dict(zip(string.ascii_uppercase, options))

            final_result.append({
                "question": j['Question'],
                "options": d,
                "correct_answer": {b:a for a, b in d.items()}[j['Correct Answer']],
                "correct_text": j['Correct Answer']
            })
    
    with open(GPQA.LOC, 'w') as f:
        for i in final_result:
            f.write(json.dumps(i)+'\n')


if __name__ == '__main__':
    load_datasets()
