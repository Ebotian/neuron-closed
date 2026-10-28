import unittest
from types import SimpleNamespace
import torch
from torch import nn
from discover import readout_gradients
from validate import lexical_score
from self_expand import parse_expansion

class Tokenizer:
    def apply_chat_template(self,messages,tokenize,enable_thinking,add_generation_prompt):
        return messages[0]['content']
    def __call__(self,text,add_special_tokens=False):
        return SimpleNamespace(input_ids=[1+(ord(x)%4) for x in text])

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Module();self.mlp.down_proj=nn.Linear(5,4,bias=False)
    def forward(self,x):return self.mlp.down_proj(x)

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([Layer()])
    def forward(self,x):return self.layers[0](x)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(8,5);self.lm=LanguageModel();self.out=nn.Linear(4,8,bias=False)
    def forward(self,input_ids,use_cache=False):
        return SimpleNamespace(logits=self.out(self.lm(self.emb(input_ids))))

class PipelineTests(unittest.TestCase):
    def test_shared_forward_grad_matches_fd(self):
        torch.manual_seed(15);model=ToyModel().eval();model.requires_grad_(False)
        tok=Tokenizer();lm=model.lm
        norms={0:lm.layers[0].mlp.down_proj.weight.detach().norm(dim=0)}
        t1={'name':'a','q_pos':{1:1.},'q_neg':{2:1.}}
        t2={'name':'b','q_pos':{3:1.},'q_neg':{4:1.}}
        rows=[{'id':'p1','prompt':'math'}]
        g=readout_gradients(tok,model,lm,[0],rows,{'a':t1,'b':t2},{0:1.},norms,'cpu',256)
        for target, key in [(t1,'a'),(t2,'b')]:
            grad=float(g[key][0][0,2])
            pp=lexical_score(tok,model,lm,'math',target,[{'layer':0,'neuron':2,'dose':.001}],{0:1.},norms,'cpu')
            pm=lexical_score(tok,model,lm,'math',target,[{'layer':0,'neuron':2,'dose':-.001}],{0:1.},norms,'cpu')
            self.assertAlmostEqual(grad,(pp-pm)/.002,delta=.001)
    def test_self_generation_strict_parse(self):
        r=parse_expansion('foo {"positive_phrases":["a","b","c"],"neutral_phrases":["x","y","z"]}')
        self.assertEqual(len(r['positive_phrases']),3)
        with self.assertRaises(ValueError):parse_expansion('garbage')

if __name__=='__main__':unittest.main()

class DataSamplingTests(unittest.TestCase):
    def test_all_traits_have_bilingual_pilot(self):
        from engine import read_jsonl
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        rows=read_jsonl(root/'probes/contrasts.jsonl')
        for trait in ['hostility','enthusiasm','pessimism']:
            disc=[r for r in rows if r['trait']==trait and r['split']=='discovery']
            held=[r for r in rows if r['trait']==trait and r['split']=='heldout']
            self.assertEqual(len(disc),6)
            self.assertEqual(len(held),12)
            self.assertEqual(sum(r['language']=='zh' for r in held[:6]),3)
            self.assertEqual(sum(r['language']=='en' for r in held[:6]),3)
