import json
import tempfile
import unittest
from pathlib import Path
import torch
from audit import guarded_manifest, json_digest, file_hash
from design import build_arms, physical_coordinates
from report import summarize, build_report, ranks, pearson
from stability import stable_stats
from choices import parse_choice, choice_prompt

class ResearchDesignTests(unittest.TestCase):
    def setUp(self):
        self.top=[{'layer':0,'neuron':0,'sign':1},{'layer':1,'neuron':2,'sign':-1}]
        self.grad={0:{'mean':torch.tensor([4.,-3.,1.,-2.,.1,.8])},
                   1:{'mean':torch.tensor([-.2,.5,-5.,1.,-1.,2.])}}
        self.dims={0:6,1:6}

    def test_random_own_sign_independent_of_parent(self):
        arms=build_arms(self.top,self.grad,self.dims,2,1729)
        by_id={a['arm_id']:a for a in arms}
        for i in range(2):
            own, parent=by_id[f'R{i:03d}'],by_id[f'P{i:03d}']
            self.assertEqual(own['coordinate'],parent['coordinate'])
            self.assertEqual(own['sign'],1 if own['predicted_gradient']>=0 else -1)
            self.assertEqual(parent['sign'],by_id[f'G{i:03d}']['sign'])
            self.assertEqual(parent['matched_pair'],i)
        self.assertEqual(len(physical_coordinates(arms)),4)
        # Crucially, group views are not 2 separate sampled random coordinates.
        self.assertEqual(len([a for a in arms if a['group']=='random_own_sign']),2)
        self.assertEqual(len([a for a in arms if a['group']=='random_parent_sign']),2)

    def test_random_zero_allowed_but_empty_top_not(self):
        self.assertEqual(len(build_arms(self.top,self.grad,self.dims,0,1)),2)
        with self.assertRaises(ValueError):build_arms([],self.grad,self.dims,0,1)

    def test_no_hardcoded_width(self):
        arms=build_arms(self.top,self.grad,{0:6,1:6},2,42)
        self.assertTrue(all(a['neuron']<6 for a in arms))
        with self.assertRaises(ValueError):build_arms(self.top,self.grad,{0:2,1:2},0,42)

    def test_report_two_polarities_on_one_coordinate(self):
        arms=build_arms([self.top[0]],self.grad,self.dims,1,2)
        coords=physical_coordinates(arms)
        rows=[{'probe_id':'p','coordinate':'baseline','physical_sign':0,'margin':0.}]
        for l,n in coords:
            coord=f'L{l}N{n}'
            # Distinct slopes to expose parent/own sign mistakes
            grad=self.grad[l]['mean'][n].item()
            rows.extend([{'probe_id':'p','coordinate':coord,'physical_sign':sgn,
                          'margin':sgn*grad*.1} for sgn in (+1,-1)])
        summary=summarize(arms,rows,.1)
        per={a['arm_id']:a for a in summary}
        self.assertAlmostEqual(per['G000']['observed_mean_direction'],.4)
        self.assertAlmostEqual(per['R000']['observed_mean_direction'],abs(per['R000']['predicted_gradient'])*.1)
        self.assertAlmostEqual(per['P000']['observed_mean_direction'],
                               .1*per['P000']['sign']*per['P000']['predicted_gradient'])

    def test_full_report_and_missing_guard(self):
        top=[self.top[0]]
        arms=build_arms(top,self.grad,self.dims,1,2)
        coords=physical_coordinates(arms)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            manifest={'schema':'verification-v2','arms':arms,'dose':.1,
                      'heldout_probes':[{'id':'p'}],
                      'physical_coordinates':coords}
            (d/'manifest.json').write_text(json.dumps(manifest))
            records=[{'probe_id':'p','coordinate':'baseline','physical_sign':0,'margin':0.}]
            for layer,n in coords:
                for sign in (+1,-1):
                    records.append({'probe_id':'p','coordinate':f'L{layer}N{n}',
                                    'physical_sign':sign,
                                    'margin':.1*sign*float(self.grad[layer]['mean'][n])})
            path=d/'verify.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
            text=build_report(d)
            self.assertIn('random_own_sign',text)
            self.assertIn('random_parent_sign',text)
            self.assertEqual(len(json.loads((d/'analysis.json').read_text())['matched_pairs']),1)
            path.write_text('\n'.join(json.dumps(r) for r in records[:-1])+'\n')
            with self.assertRaises(RuntimeError):build_report(d)

    def test_resume_manifest_rejects_modified_dose(self):
        with tempfile.TemporaryDirectory() as d:
            guarded_manifest(d,{'dose':.05,'sha':'x'})
            guarded_manifest(d,{'dose':.05,'sha':'x'})
            with self.assertRaises(RuntimeError):guarded_manifest(d,{'dose':.1,'sha':'x'})
        with tempfile.TemporaryDirectory() as d:
            Path(d,'orphan.jsonl').write_text('anything')
            with self.assertRaises(RuntimeError):guarded_manifest(d,{'dose':.05})

    def test_same_file_digest_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'input.jsonl');p.write_text('a')
            h=file_hash(p);p.write_text('b');self.assertNotEqual(h,file_hash(p))

    def test_stability_detects_one_probe_outlier(self):
        # First coordinate looks important only due to one massive outlier
        per=torch.tensor([[100.,0.,1.,1.,0.],[0.,10.,1.,1.,0.],[0.,10.,1.,1.,0.]])
        summary=stable_stats({0:{'per_probe':per}},ks=(1,2))
        self.assertEqual(summary['num_probes'],3)
        self.assertGreater(summary['top64_probe_dominance'][0]['dominance_max_abs_probe_fraction'],.99)
        self.assertLess(summary['top_k']['1']['mean_overlap'],1.0)

    def test_choice_mirrors_and_parse(self):
        task={'scenario':'Pick', 'trait_option':'Avoid', 'other_option':'Proceed'}
        self.assertIn('Option A: Avoid',choice_prompt(task,False))
        self.assertIn('Option B: Avoid',choice_prompt(task,True))
        self.assertEqual(parse_choice('A\n'), 'A')
        self.assertEqual(parse_choice('B.'), 'B')
        self.assertIsNone(parse_choice('Answer: A'))

    def test_report_requires_unique_physical_results(self):
        with self.assertRaises(ValueError):summarize([], [{'probe_id':'p','coordinate':'baseline',
                                                           'physical_sign':0,'margin':0.}]*2,.1)

if __name__=='__main__':unittest.main()
