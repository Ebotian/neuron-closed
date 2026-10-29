import json,math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch

from semantic import compile_target,sparse_readout,next_token_objective,readout_vector,validate_target
from inverse import score_weights,top_coordinates,fixed_budget_control,minimum_energy_control,combinations
from validate import controls
from analyze import group_analysis,exact_sign_p
from protocol import validate_inputs,seal_discovery,require_discovery_seal

class ToyTokenizer:
    all_special_ids=[0]
    def __call__(self,text,add_special_tokens=False):
        return SimpleNamespace(input_ids=[(ord(c)%7)+1 for c in text])

class ToyDown:
    def __init__(self,W):self.weight=W

class ToyLM:
    def __init__(self,W):self.layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=ToyDown(W)))]

class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.tok=ToyTokenizer()
        self.W=torch.randn(8,4,generator=torch.Generator().manual_seed(4))
        self.spec={'description':'angry','neutral_description':'calm'}
        self.rows=[{'positive':'very harsh','negative':'very calm','prompt':'x'}]
    def test_lexical(self):
        t=compile_target('lexical',self.spec,self.tok,self.W,self.W,self.rows)
        validate_target(t)
        v=readout_vector(t,self.W)
        self.assertEqual(v.shape,(4,))
        logits=torch.randn(8)
        perm=next_token_objective(logits+3.0,t)-next_token_objective(logits,t)
        self.assertAlmostEqual(float(perm),0,places=5)
    def test_distribution(self):
        t=compile_target('distribution',self.spec,self.tok,self.W,self.W,self.rows,distribution_top=3,temperature=.1)
        validate_target(t)
        self.assertEqual(len(t['q_pos']),3)
        self.assertEqual(len(t['q_neg']),3)
    def test_counterfactual_no_heldout(self):
        t=compile_target('counterfactual',self.spec,self.tok,self.W,self.W,self.rows)
        validate_target(t)
        self.assertIn('unigram',t['assumptions'])
    def test_reject_equal_counterfactual(self):
        with self.assertRaises(ValueError):
            compile_target('counterfactual',self.spec,self.tok,self.W,self.W,
                           [{'positive':'same','negative':'same'}])

class InverseTests(unittest.TestCase):
    def test_formula_fixed_budget(self):
        u,p=fixed_budget_control([3,-4],.5)
        self.assertAlmostEqual(u[0],.3)
        self.assertAlmostEqual(u[1],-.4)
        self.assertAlmostEqual(p,2.5)
    def test_minimal_energy(self):
        x=minimum_energy_control([3,4],1,.3)
        self.assertTrue(x['feasible'])
        self.assertAlmostEqual(x['required_budget'],.2)
        self.assertAlmostEqual(sum(a*b for a,b in zip([3,4],x['doses'])),1)
        self.assertFalse(minimum_energy_control([3,4],1,.1)['feasible'])
    def test_weight_projection(self):
        W=torch.tensor([[1.,0.,-1.],[0.,2.,0.]])
        r=score_weights(ToyLM(W),[0],torch.tensor([1.,0.]),{0:2.},
                        {0:torch.tensor([1.,2.,1.])})
        self.assertTrue(torch.equal(r[0],torch.tensor([2.,0.,-2.])))
        top=top_coordinates(r,2)
        self.assertEqual(set((x['neuron'],x['sign']) for x in top),{(0,1),(2,-1)})
    def test_combo_surrogate(self):
        top=[{'layer':0,'neuron':i,'slope':s} for i,s in enumerate([3,-4,1])]
        a=combinations(top,budget=.2,target_delta=.1)
        self.assertEqual(a[0]['support'],2)
        self.assertAlmostEqual(a[0]['surrogate_fixed_budget_delta'],1.0)
    def test_random_owns_sign(self):
        scores={0:torch.tensor([5.,-3.,-1.,-2.])}
        top=[{'layer':0,'neuron':0,'sign':1,'slope':5.}]
        norms={0:torch.ones(4)}
        arms=controls(top,scores,norms,random_k=1,seed=11,norm_control=False)
        rand=[a for a in arms if a['group']=='random_own_sign']
        self.assertEqual(len(rand),1)
        self.assertEqual(rand[0]['dose_sign'],-1)
        self.assertEqual(len({(a['layer'],a['neuron']) for a in arms if a['group']!='sign_flip'}),2)

class AnalysisTests(unittest.TestCase):
    def test_exact_sign(self):
        self.assertAlmostEqual(exact_sign_p([1]*8),.0078125)
        self.assertAlmostEqual(exact_sign_p([1]*7+[-1]),.0703125)
        self.assertIsNone(exact_sign_p([0,0]))
    def test_prompt_not_coordinate_as_n(self):
        rows=[]
        for i in range(3):
            row={'key':'h_a_weight','source_trait':'h','test_trait':'h','prompt_id':str(i),
                 'compiler':'a','solver':'weight','lexical_score':0,'teacher_forced_margin':0,
                 'slope_discovery':None,'interventions':[],'matched_pair':None}
            rows.append({**row,'group':'baseline','arm_id':'BASE'})
            for j in range(8):
                rows.append({**row,'group':'top','arm_id':f'T{j}','matched_pair':j,
                             'interventions':[{'dose':.05}],
                             'lexical_score':1,'teacher_forced_margin':1})
                rows.append({**row,'group':'random_own_sign','arm_id':f'R{j}','matched_pair':j,
                             'interventions':[{'dose':.05}],
                             'lexical_score':.5,'teacher_forced_margin':.5})
        summary,pairs,interactions=group_analysis(rows)
        self.assertTrue(all(r['n_prompts']==3 for r in summary))
        self.assertEqual(pairs[0]['n_heldout_prompts'],3)
        self.assertAlmostEqual(pairs[0]['mean_top_minus_random'],.5)
        self.assertAlmostEqual(pairs[0]['exploratory_two_sided_exact_sign_p'],.25)
    def test_freeze_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'x.json').write_text('{}')
            seal_discovery(d);require_discovery_seal(d)
            (d/'x.json').write_text('{"a":1}')
            with self.assertRaises(RuntimeError):require_discovery_seal(d)
            with self.assertRaises(RuntimeError):seal_discovery(d)
    def test_split_validation(self):
        spec={'traits':{'a':{}}}
        rows=[{'trait':'a','split':x,'id':x,'prompt':'p','positive':'a','negative':'b'}
              for x in ('discovery','heldout')]
        self.assertEqual(validate_inputs(spec,rows)['n_rows'],2)
        with self.assertRaises(ValueError):validate_inputs(spec,rows+[rows[0]])

if __name__=='__main__':unittest.main()

class PairInteractionsTests(unittest.TestCase):
    def test_nonlinear_interaction_exact_component_doses(self):
        rows=[]
        def make(group,arm,j):
            return {'key':'t_lexical_jacobian','source_trait':'t','test_trait':'t','prompt_id':'p',
                    'group':group,'arm_id':arm,'lexical_score':j,
                    'teacher_forced_margin':j,'combo_id':'C000' if group!='baseline' else None,
                    'interventions':[], 'compiler':'lexical','solver':'jacobian',
                    'slope_discovery':None,'matched_pair':None}
        rows=[make('baseline','BASE',2.),make('analytic_pair','C000',10.),
              make('pair_component','C000P0',4.),make('pair_component','C000P1',5.)]
        _,_,interactions=group_analysis(rows)
        self.assertEqual(len(interactions),1)
        self.assertAlmostEqual(interactions[0]['mean_interaction_transfer'],3.)
