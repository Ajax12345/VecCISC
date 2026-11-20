import openai, typing
import re, ast, os
import tiktoken
import config

_CLIENT = None

def client() -> openai.OpenAI:
    """OpenAI client, created on first use so that scripts that only analyse
    cached results do not require an API key."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    return _CLIENT

def token_num(s:str, encoding:str = 'cl100k_base') -> int:
    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(s))

def truncate_embedding_text(s:str) -> str:
    '''Trim the start of `s` so it fits in text-embedding-3-small's 8192-token window.'''
    if token_num(s, 'cl100k_base') <= 8192:
        return s

    j = s.split()
    for i in range(1, len(j)):
        if token_num(truncated:=' '.join(j[i:]), 'cl100k_base') <= 8192:
            return truncated

def get_embedding(text:str) -> typing.List[float]:
    return client().embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    ).data[0].embedding

def query_gpt(system:str, user:str, temperature:float, model:str) -> str:
    completion = client().chat.completions.create(
        model=model,
        store=True,
        temperature = temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )

    return completion.choices[0].message.content


class LLM:
    '''A prompt template pair, loaded from prompts/system/<name>.txt and prompts/user/<name>.txt.'''
    def __init__(self, name:str) -> None:
        self.name = name

    def parse_json_response(self, response:str) -> typing.Any:
        return ast.literal_eval(re.findall(r'```json[\w\W]+', response)[0].lstrip('```json\n').rstrip('\n```'))

    def __call__(self, config_:dict,
                 display_prompts:bool = False,
                 batch_format:bool = False) -> typing.Any:
        '''
        config_:
            - retries
            - temperature
            - model
            - params
                - system
                - user

        With batch_format=True, returns the request body for a batch-API call
        instead of querying the model.
        '''
        with open(os.path.join(config.PROMPTS_DIR, 'system', f'{self.name}.txt')) as f:
            system = f.read().format(**config_['params']['system'])

        with open(os.path.join(config.PROMPTS_DIR, 'user', f'{self.name}.txt')) as f:
            user = f.read().format(**config_['params']['user'])

        if batch_format:
            return {
                "model": config_['model'],
                'temperature': config_['temperature'],
                'messages': [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            }

        for _ in range(config_['retries']):
            resp = query_gpt(system, user, config_['temperature'], config_['model'])
            if display_prompts:
                print('user prompt')
                print(user)
                print('-'*50)
                print(resp)
            try:
                return {'json': self.parse_json_response(resp), 'resp': resp}

            except Exception as e:
                print('could not parse model response:', e)
