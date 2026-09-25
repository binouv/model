"""Build a reproducible local-only bootstrap corpus and disjoint evaluation.
Sources: installed CPython files (PSF license) and assistant-authored synthetic
RU/EN/code/reasoning. NOT a general web-scale pretraining corpus.
"""
from __future__ import annotations
import ast,collections,hashlib,json,random,re,shutil,sysconfig
from pathlib import Path
import numpy as np
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
SEEDS={'train':7111,'validation':7141,'test':7151}

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def pair(prompt,completion):return prompt+'\n'+completion

def reasoning_row(rng,i,split):
    ru=i%2==0; task=i%8; a=rng.randrange(0,100);b=rng.randrange(0,100);c=rng.randrange(1,10)
    # Novel surface families are a separate OOD test, not used by tokenizer/training.
    if task==0:
        p=f'Вопрос: Сколько будет {a} + {b}?\nОтвет:' if ru else f'Question: What is {a} + {b}?\nAnswer:'
        ans=str(a+b); completion=f' {a} + {b} = {a+b}. Итог: {a+b}.' if ru else f' {a} + {b} = {a+b}. Final: {a+b}.'
        return 'arithmetic',p,completion,ans
    if task==1:
        p=f'Question: Start with {a}, add {b}, then subtract {c}.\nAnswer:'
        ans=str(a+b-c);return 'multistep',p,f' After addition: {a+b}. After subtraction: {a+b-c}. Final: {a+b-c}.',ans
    if task==2:
        keys=rng.sample(['oak','pine','reed','ash','birch','elm','fern','ivy','moss','rose','sage','flax'],4)
        vals=rng.sample(['red','blue','green','gold','black','white','pink','gray','cyan','lime'],5)
        writes=list(zip(keys,vals));rng.shuffle(writes); k=rng.choice(keys); writes.append((k,vals[-1]))
        p='Память: '+', '.join(f'{k0}={v}' for k0,v in writes)+f'. Каково последнее значение {k}?\nОтвет:'
        return 'memory_update',p,f' {vals[-1]}.',vals[-1]
    if task==3:
        keys=rng.sample(['amber','brook','cedar','delta','ember','field','grove','hill','iris','jade','kite','lake'],5)
        facts=list(zip(keys[:-1],keys[1:]));rng.shuffle(facts)
        p='Memory: '+', '.join(f'{x}->{y}' for x,y in facts)+f'. Follow 3 links from {keys[0]}.\nAnswer:'
        return 'memory_chain',p,f' {keys[0]} -> {keys[1]} -> {keys[2]} -> {keys[3]}. Final: {keys[3]}.',keys[3]
    if task==4:
        nums=[rng.randrange(20) for _ in range(5)];total=sum(nums)
        p='Вопрос: Найди сумму чисел '+', '.join(map(str,nums))+'.\nОтвет:'
        return 'list_sum',p,' '+ ' + '.join(map(str,nums))+f' = {total}. Итог: {total}.',str(total)
    if task==5:
        op=rng.choice(['+','-','*']); v=a+b if op=='+' else a-b if op=='-' else a*b
        p=f'Python:\nx = {a}\ny = {b}\nprint(x {op} y)\nOutput:'
        return 'code_trace',p,' '+str(v)+'\n',str(v)
    if task==6:
        nums=[rng.randrange(20) for _ in range(4)]; out=sorted(nums)
        p=f'Вопрос: Упорядочи по возрастанию список {nums}.\nОтвет:'
        return 'sorting',p,' '+str(out)+'.',str(out)
    xs=rng.sample(['A','B','C','D','E','F'],3)
    p=f'Facts: Every {xs[0]} is {xs[1]}. Every {xs[1]} is {xs[2]}. Is every {xs[0]} a {xs[2]}?\nAnswer:'
    return 'logic',p,f' Yes. {xs[0]} implies {xs[1]}, and {xs[1]} implies {xs[2]}. Therefore {xs[0]} implies {xs[2]}.','Yes'

RU_TOPICS=[
('память','Записи в памяти связывают ключи со значениями. При обновлении одного ключа остальные записи не должны меняться. Порядок хранения не обязан совпадать с порядком рассуждения.'),
('проверка','Правильный конечный ответ не всегда означает правильное решение. Несколько ошибочных действий иногда приводят к тому же числу. Поэтому полезно проверять промежуточные шаги.'),
('программа','Программа состоит из инструкций, которые выполняются в заданном порядке. Переменная хранит значение. Функция получает аргументы и возвращает результат.'),
('цикл','Цикл повторяет действия для каждого элемента коллекции. Если условие остановки никогда не выполняется, программа может продолжать работу бесконечно. Границы цикла нужно проверять отдельно.'),
('список','Список сохраняет порядок элементов. Индекс указывает на конкретный элемент. Перед обращением по индексу важно убедиться, что он входит в допустимый диапазон.'),
('словарь','Словарь сопоставляет ключи и значения. Один и тот же ключ обозначает одну запись. Новое присваивание заменяет прежнее значение, но не создаёт второго независимого ключа.'),
('ошибка','Ошибка в программе является наблюдением о её поведении. Полезно воспроизвести проблему на небольшом примере, проверить входные данные и сравнить ожидаемый результат с фактическим.'),
('сравнение','Для честного сравнения моделей нужны одинаковые задачи и одинаковые правила оценки. Тестовые примеры не должны участвовать в обучении. Размер модели сам по себе не измеряет качество.'),
('граф','Граф содержит вершины и связи между ними. Направленная связь задаёт допустимый переход. Поиск пути отличается от выбора ближайшей вершины: иногда нужно сохранить несколько альтернатив.'),
('неопределённость','При неполных данных может существовать несколько объяснений. Уверенность следует пересматривать после нового наблюдения. Преждевременный выбор единственной гипотезы может уничтожить полезную информацию.'),
('текст','Смысл предложения зависит не только от отдельных слов, но и от их порядка. Местоимение может ссылаться на ранее названный предмет. Для понимания такого предложения нужен контекст.'),
('арифметика','Сложение объединяет количества, а вычитание находит разность. При нескольких операциях нужно соблюдать порядок действий. Проверка обратной операцией помогает заметить ошибку.'),
('эксперимент','Эксперимент начинается с конкретной проверяемой гипотезы. До получения результата фиксируют протокол и метрики. Незавершённый запуск нельзя записывать как успешный.'),
('файл','Файл хранит последовательность байтов. Кодировка определяет, как байты превращаются в символы. При сохранении текста важно явно указывать кодировку и проверять повторное чтение.'),
('рекурсия','Рекурсивная функция вызывает себя для меньшей подзадачи. Базовый случай останавливает вычисление. Если размер подзадачи не уменьшается, рекурсия может не завершиться.'),
('план','Сложную задачу удобно разбить на проверяемые этапы. После каждого этапа сохраняют результат и проверяют ограничения. Когда предположение не подтвердилось, план нужно изменить, а не скрывать неудачу.'),
]
EN_TOPICS=[
('memory','A memory record maps a key to a value. Updating one key should leave the other records unchanged. Physical storage order need not match the order of reasoning.'),
('verification','A correct final answer does not guarantee a correct solution. Wrong intermediate steps may occasionally produce the same result. Checking the complete trace reveals such collisions.'),
('functions','A function takes inputs and produces an output. Pure functions do not modify hidden state. Keeping interfaces explicit makes a program easier to test and reuse.'),
('graphs','A directed graph contains vertices and directed edges. Following a path requires respecting each edge. A local greedy choice can discard an alternative that is needed later.'),
('uncertainty','Incomplete observations can support several explanations. New evidence should revise confidence. Keeping alternatives is useful when an early irreversible decision would lose information.'),
('testing','A test compares observed behavior with an expected result. Boundary cases matter because ordinary examples may miss indexing or rounding mistakes. Reproducibility requires saving inputs and configuration.'),
('recursion','A recursive procedure solves a problem by solving smaller instances. A base case stops the recursion. The procedure must make progress toward that case on every recursive call.'),
('language','A sentence is more than a bag of words. Word order and context change its meaning. A pronoun may refer to an earlier entity, so interpreting it requires remembering prior text.'),
('data','Training examples influence the patterns a model learns. Validation data should be separate from training data. Good performance on repeated templates does not establish broad generalization.'),
('files','A file stores bytes. An encoding maps bytes to characters. A reliable program records the encoding, validates its input, and checks that saving and loading preserve the content.'),
('planning','A plan decomposes a goal into smaller actions. Each action should have a clear success condition. When evidence contradicts an assumption, the plan should be revised.'),
('algorithms','An algorithm specifies a sequence of operations. Its correctness depends on invariants that hold at every step. An efficient but incorrect shortcut is not a valid replacement.'),
]

def make_corpus():
    data=ROOT/'data';data.mkdir(parents=True,exist_ok=True)
    (ROOT/'licenses').mkdir(parents=True,exist_ok=True)
    sources=[]; rows={s:[] for s in SEEDS}; seen=set()
    def add(split,kind,text,source):
        h=sha(text)
        if h in seen:return
        seen.add(h);rows[split].append({'id':h,'kind':kind,'text':text,'source':source})
    std=Path(sysconfig.get_path('stdlib'))
    files=list(std.glob('*.py'))
    for d in ['asyncio','collections','concurrent','email','encodings','html','http','importlib','json','logging','multiprocessing','unittest','urllib','xml']:
        files.extend((std/d).rglob('*.py'))
    for p in sorted(set(files)):
        text=p.read_text(errors='replace')
        if len(text)<100 or len(text)>160000:continue
        rel=str(p.relative_to(std));bucket=int(sha(rel)[:8],16)%20
        split='test' if bucket==0 else 'validation' if bucket==1 else 'train'
        sources.append({'path':str(p),'relative':rel,'sha256':sha(text),'split':split,'license':'PSF-2.0 and file-specific included notices','bytes':len(text.encode())})
        # Non-overlapping complete-line chunks; all chunks of one file have one split.
        acc=[];n=0
        for line in text.splitlines(keepends=True):
            acc.append(line);n+=len(line)
            if n>=1800:
                add(split,'python_source',''.join(acc),'CPython/'+rel);acc=[];n=0
        if acc:add(split,'python_source',''.join(acc),'CPython/'+rel)
        try:
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    doc=ast.get_docstring(node)
                    if doc and 80<len(doc)<6000:add(split,'english_docstring',doc,'CPython/'+rel)
        except SyntaxError:pass
    license=std/'LICENSE.txt'
    if license.exists():shutil.copy2(license,ROOT/'licenses/CPython_LICENSE.txt')
    evalprobes={}
    for split,seed in SEEDS.items():
        rng=random.Random(seed);n=24000 if split=='train' else 1600
        probes=[]
        for i in range(n):
            kind,p,c,answer=reasoning_row(rng,i,split)
            h=sha(p+'\n'+c)
            if h in seen:continue
            add(split,kind,p+'\n'+c,'assistant-authored-synthetic-v1')
            if split!='train' and sum(z['kind']==kind for z in probes)<12:
                probes.append({'id':h,'kind':kind,'prompt':p+'\n','gold_completion':c,'final_answer':answer})
        evalprobes[split]=probes
    # Original educational prose: topic-held-out split, no template split claims.
    for lang,topics in [('ru',RU_TOPICS),('en',EN_TOPICS)]:
        for i,(topic,body) in enumerate(topics):
            split='validation' if i%8==6 else 'test' if i%8==7 else 'train'
            prefix=f'Тема: {topic}.\n' if lang=='ru' else f'Topic: {topic}.\n'
            add(split,lang+'_prose',prefix+body,'assistant-authored-prose')
    # These small corpora are a bootstrap. Add many distinct executable Python functions,
    # not fixed class labels. Numeric interpolation alone is NOT broad code competence.
    for split,seed in SEEDS.items():
        rng=random.Random(seed+600);n=2000 if split=='train' else 100
        for i in range(n):
            k=rng.randrange(10000);f=rng.choice(['offset','multiply','clamp','prefix','lookup'])
            if f=='offset': text=f'def add_{k}(x):\n    """Return x increased by {k}."""\n    return x + {k}\n'
            elif f=='multiply':text=f'def scale_{k}(x):\n    """Multiply x by {k}."""\n    return x * {k}\n'
            elif f=='clamp':text=f'def clamp_{k}(x):\n    if x < 0:\n        return 0\n    if x > {k}:\n        return {k}\n    return x\n'
            elif f=='prefix':text=f'def first_{k}(values):\n    """Return at most {k} initial elements."""\n    return values[:{k}]\n'
            else:text=f'def lookup_{k}(records):\n    """Read key {k} without modifying memory."""\n    return records.get({k})\n'
            add(split,'synthetic_python',text,'assistant-authored-synthetic-python')
    for split,rr in rows.items():
        random.Random(SEEDS[split]).shuffle(rr)
        with (data/f'{split}.jsonl').open('w') as f:
            for row in rr:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tok=Tokenizer.train((z['text'] for z in rows['train']),8192);tok.save(data/'tokenizer.json')
    stats={}
    for split,rr in rows.items():
        streams=collections.defaultdict(list);ids=[]
        for row in rr:
            tt=tok.encode(row['text'],bos=True,eos=True)
            assert tok.decode(tt)==row['text']
            ids.extend(tt);streams[row['kind']].extend(tt)
        np.asarray(ids,dtype=np.uint16).tofile(data/f'{split}.bin')
        for kind,tt in streams.items():np.asarray(tt,dtype=np.uint16).tofile(data/f'{split}_{kind}.bin')
        stats[split]={'documents':len(rr),'tokens':len(ids),'bytes':sum(len(z['text'].encode()) for z in rr),'by_kind':{k:len(v) for k,v in streams.items()}}
        if split!='train':(data/f'{split}_generation.jsonl').write_text(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in evalprobes[split]))
    sets={s:{z['id'] for z in rr} for s,rr in rows.items()}
    assert not sets['train']&sets['test'] and not sets['train']&sets['validation'] and not sets['validation']&sets['test']
    payload={'status':'completed','seeds':SEEDS,'tokenizer':{'type':'frequency lexical subwords + lossless UTF8 byte fallback; NOT BPE','vocab':tok.vocab_size,'trained_on':'train only'},'sources':sources,'splits':stats,'exact_document_overlap':0,'semantic_family_overlap':'synthetic tasks share generators; held seeds do NOT establish independent broad reasoning','external_dataset_downloads':False,'note':'Local CPython and assistant-authored bootstrap; not a foundation-model pretraining corpus.'}
    (data/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(json.dumps({'status':'completed','splits':stats,'vocab':tok.vocab_size},indent=2))
if __name__=='__main__':make_corpus()
