import json,re,difflib,unicodedata as ud,collections
d=json.load(open('newrun.json'))
MON={'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
def post(i):
    mo=MON[re.search(r'_([a-z]{3})\d',i).group(1)]; yr=int(re.search(r'_(\d{4})_',i).group(1))
    return yr>2026 or (yr==2026 and mo>=6)
items=[it for it in d['items'] if it['cityId'] in {'argos','orestiada'} and not post(it['itemId'])]
SYS={'ours':'oc-runpod-fixed-2026-08-10','scribe':'scribe-v2-clean'}

def noacc(s): return ''.join(c for c in ud.normalize('NFD',s) if ud.category(c)!='Mn')
def sig(s): return s.replace('ς','σ')
def phon(s):
    s=sig(noacc(s.lower()))
    for a,b in [('ει','ι'),('οι','ι'),('υι','ι'),('η','ι'),('υ','ι'),('ω','ο'),('αι','ε'),
                ('μπ','b'),('ντ','d'),('γκ','g'),('γγ','g'),('αυ','af'),('ευ','ef')]:
        s=s.replace(a,b)
    return s
GREEK_NUM=set('μηδέν ένα ενός μία μια δύο δυο τρία τρεις τέσσερα τέσσερις πέντε έξι εφτά επτά οχτώ οκτώ εννιά εννέα δέκα έντεκα δώδεκα είκοσι τριάντα σαράντα πενήντα εξήντα εβδομήντα ογδόντα ενενήντα εκατό διακόσια τριακόσια χίλια χιλιάδες εκατομμύρια εκατομμύριο'.split())
FUNC=set('ο η το του της των τον την τα οι ένας μια ένα στο στη στην στον στα στους στις σε με και κι που πως ως για από ή αν να θα δεν μη μην αυτό αυτή αυτός εγώ εμείς εσείς μας σας τους τις'.split())

def classify(op,r,h):
    if op=='del': return 'deletion'
    if op=='ins': return 'insertion'
    rl,hl=r.lower(),h.lower()
    if rl==hl: return 'identical'
    if sig(rl)==sig(hl): return 'final_sigma'
    if noacc(rl)==noacc(hl): return 'accent_tonos'
    if sig(noacc(rl))==sig(noacc(hl)): return 'final_sigma'
    if phon(rl)==phon(hl): return 'homophone'
    if (rl in GREEK_NUM or hl in GREEK_NUM) or re.search(r'\d',r+h): return 'number_date'
    if rl in FUNC and hl in FUNC: return 'article_pronoun'
    if len(rl)<=2 or len(hl)<=2: return 'acronym_abbreviation'
    # morphology: same stem, different ending
    n=min(len(rl),len(hl))
    stem=0
    for i in range(n):
        if noacc(rl)[i]==noacc(hl)[i]: stem+=1
        else: break
    if stem>=max(3,int(0.6*max(len(rl),len(hl)))): return 'inflection_or_nearmiss'
    if difflib.SequenceMatcher(a=noacc(rl),b=noacc(hl)).ratio()>=0.75: return 'inflection_or_nearmiss'
    return 'substitution_phonetic_or_word'

def toks(s): return re.findall(r'\w+',(s or '').lower(),re.UNICODE)
def ops(ref,hyp):
    sm=difflib.SequenceMatcher(a=ref,b=hyp,autojunk=False); out=[]
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='replace':
            for k in range(max(i2-i1,j2-j1)):
                a=ref[i1+k] if i1+k<i2 else None; b=hyp[j1+k] if j1+k<j2 else None
                out.append(('sub' if a and b else ('del' if a else 'ins'),a,b,i1+k))
        elif tag=='delete':
            for k in range(i1,i2): out.append(('del',ref[k],None,k))
        elif tag=='insert':
            for k in range(j1,j2): out.append(('ins',None,hyp[k],i1))
    return out

res={k:collections.Counter() for k in SYS}
unres={k:[] for k in SYS}
for it in items:
    ref=toks(it['referenceText']); pp=it['perProvider']
    for who,pid in SYS.items():
        if pid not in pp: continue
        h=toks(pp[pid].get('scoredHypothesis') or pp[pid].get('hypothesisText'))
        for op,r,hh,idx in ops(ref,h):
            c=classify(op,r,hh)
            res[who][c]+=1
            if c=='substitution_phonetic_or_word' and len(unres[who])<300:
                ctx=' '.join(ref[max(0,idx-4):idx+5])
                unres[who].append({'ref':r,'hyp':hh,'context':ctx})
for who in SYS:
    tot=sum(res[who].values())
    print(f'=== {who}: {tot} errors')
    for c,n in res[who].most_common():
        print(f'  {n:5d} ({n/tot:5.1%})  {c}')
    print()
json.dump(unres,open('unclassified.json','w'),ensure_ascii=False,indent=1)
print('sent to LLM stage:',{k:len(v) for k,v in unres.items()})
