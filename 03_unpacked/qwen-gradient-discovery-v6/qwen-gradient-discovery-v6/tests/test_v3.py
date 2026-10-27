"""v3 audit regression tests: no GPU or external model needed."""
import json
import os
import tempfile
import sys
import zipfile
import unittest
from pathlib import Path
from unittest.mock import patch
from audit import (check_scan_file,check_discovery_source, fingerprint_with_stat_cache,
                   sha256_file,file_hash)
from report import exact_sign_test, paired_signflip_test, distribution, build_report
from choices import summarize_choices, parse_choice
from gate import gate_decision
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from migrate_repo_packages import migrate

class AuditV3Tests(unittest.TestCase):
    def test_strict_scan_probe_lock(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'contrasts.jsonl';f.write_text('original')
            meta={'probe_file_sha256':file_hash(f)}
            self.assertEqual(check_scan_file(f,meta),file_hash(f))
            f.write_text('edited after seeing results')
            with self.assertRaisesRegex(ValueError,'Discovery probe SHA256 mismatch'):
                check_scan_file(f,meta)
            with self.assertRaises(ValueError):check_scan_file(f,{})

    def test_specificity_source_not_same_as_independent_probe_file(self):
        with tempfile.TemporaryDirectory() as d:
            disc=Path(d)/'contrasts.jsonl';spec=Path(d)/'specificity.jsonl'
            disc.write_text('original');spec.write_text('other dataset')
            meta={'probe_file_sha256':file_hash(disc)}
            self.assertEqual(check_discovery_source(meta,disc),file_hash(disc))
            with self.assertRaises(ValueError):check_discovery_source(meta,spec)

    def test_exact_sign_tests(self):
        self.assertAlmostEqual(exact_sign_test([1]*8)['p_two_sided'],.0078125)
        self.assertAlmostEqual(exact_sign_test([1]*7+[-1])['p_two_sided'],.0703125)
        self.assertAlmostEqual(paired_signflip_test([1]*8)['p_two_sided'],.0078125)
        self.assertEqual(exact_sign_test([0,0])['p_two_sided'],1)
        self.assertAlmostEqual(distribution([1,2,3,4])['iqr'],1.5)

    def test_choices_strict_letter_parser(self):
        self.assertEqual(parse_choice(' A.\n'),'A')
        self.assertIsNone(parse_choice('A because I agree'))
        self.assertIsNone(parse_choice('B, because...'))
        self.assertIsNone(parse_choice('A\nSome additional explanation'))

    def test_choices_pairs_include_invalid(self):
        tasks=[{'id':'t1','trait':'hostility'},{'id':'t2','trait':'hostility'}]
        arms=[{'arm_id':'a1','trait':'hostility'}]
        # Four paired cases; the arm becomes invalid in exactly one of them.
        values=[(True,False),(False,None),(True,True),(False,True)]
        rows=[]
        for (t,o),(baseline,arm) in zip([(t['id'],o) for t in tasks for o in (0,1)],values):
            for name,value in (('baseline',baseline),('a1',arm)):
                rows.append({'task_id':t,'order':o,'arm_id':name,'trait_option_selected':value})
        s=summarize_choices(rows,tasks,arms)['a1']
        self.assertEqual(s['n_total_paired'],4)
        self.assertEqual(s['arm_invalid'],1)
        self.assertEqual(s['n_both_valid'],3)
        self.assertAlmostEqual(s['paired_selection_delta_total'],0)
        self.assertAlmostEqual(s['invalid_rate_delta_arm_minus_baseline'],.25)
        self.assertAlmostEqual(s['paired_delta_both_valid'],0)
        with self.assertRaises(RuntimeError):summarize_choices(rows[:-1],tasks,arms)
        with self.assertRaises(RuntimeError):summarize_choices(rows+rows[:1],tasks,arms)

    def test_fingerprint_cache_requires_opt_in_and_checks_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);model=d/'model';model.mkdir()
            (model/'config.json').write_text('{}')
            w=model/'model.safetensors';w.write_bytes(b'abc123')
            with patch.dict(os.environ,{'QWEN_FINGERPRINT_CACHE':str(d/'cache')}):
                original,mode=fingerprint_with_stat_cache(model,False)
                self.assertEqual(mode,'full-sha256')
                reuse,mode=fingerprint_with_stat_cache(model,True)
                self.assertEqual(mode,'stat-cache')
                self.assertEqual(original,reuse)
                w.write_bytes(b'changed!')
                updated,mode=fingerprint_with_stat_cache(model,True)
                self.assertEqual(mode,'full-sha256')
                self.assertNotEqual(original,updated)

    def test_full_report_exact_pair_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            arms=[];coords=[]
            for i in range(8):
                coords.extend([(0,i),(0,i+100)])
                arms.extend([{'arm_id':f'G{i:03d}','group':'gradient_top','layer':0,'neuron':i,
                              'coordinate':f'L0N{i}','sign':1,'predicted_gradient':2.,'matched_to':None},
                             {'arm_id':f'R{i:03d}','group':'random_own_sign','layer':0,'neuron':i+100,
                              'coordinate':f'L0N{i+100}','sign':1,'predicted_gradient':.2,
                              'matched_to':f'G{i:03d}'},
                             {'arm_id':f'P{i:03d}','group':'random_parent_sign','layer':0,'neuron':i+100,
                              'coordinate':f'L0N{i+100}','sign':1,'predicted_gradient':.2,
                              'matched_to':f'G{i:03d}'}])
            manifest={'schema':'verification-v2','arms':arms,'physical_coordinates':coords,
                      'dose':.1,'heldout_probes':[{'id':'h1'}]}
            (d/'manifest.json').write_text(json.dumps(manifest))
            records=[{'probe_id':'h1','coordinate':'baseline','physical_sign':0,'margin':0}]
            for l,n in coords:
                for sign in (-1,1):
                    records.append({'probe_id':'h1','coordinate':f'L{l}N{n}',
                                    'physical_sign':sign,'margin':sign*(1 if n<100 else .1)})
            (d/'verify.jsonl').write_text('\n'.join(map(json.dumps,records))+'\n')
            txt=build_report(d)
            self.assertIn('SD (across coordinates)',txt)
            self.assertNotIn('Mean predicted Δmargin',txt)
            analysis=json.loads((d/'analysis.json').read_text())
            sign=analysis['paired_inference']['ranking']['sign_test']
            self.assertEqual(sign['wins'],8)
            self.assertAlmostEqual(sign['p_two_sided'],.0078125)

    def test_exploratory_gate_checks_both_thresholds_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            scan=Path(d)/'scan'; verify=Path(d)/'verify';scan.mkdir();verify.mkdir()
            scan_meta={'schema':'gradient-scan-v2','probe_file_sha256':'same',
                       'model_fingerprint':{'x':1},'code_fingerprint':'code'}
            (scan/'manifest.json').write_text(json.dumps(scan_meta))
            vm={'schema':'verification-v2','scan_identity':file_hash(scan/'manifest.json'),
                'probe_file_sha256':'same','model_fingerprint':{'x':1},
                'code_fingerprint':'code','trait':'hostility'}
            (verify/'manifest.json').write_text(json.dumps(vm))
            def analysis(wins):
                return {'matched_pairs':[{'i':i} for i in range(8)],
                        'paired_inference':{'ranking':{'independent_parent_matches':True,
                          'sign_test':{'n_effective':8,'wins':wins,'p_two_sided':.0703125}}}}
            (verify/'analysis.json').write_text(json.dumps(analysis(7)))
            (scan/'hostility_stability.json').write_text(json.dumps({'num_probes':3,'top_k':{'64':{'mean_overlap':.55}}}))
            result=gate_decision(scan,verify)
            self.assertTrue(result['pass'])
            self.assertAlmostEqual(result['sign_p_two_sided'],.0703125)
            (verify/'analysis.json').write_text(json.dumps(analysis(6)))
            self.assertFalse(gate_decision(scan,verify)['pass'])
            (verify/'analysis.json').write_text(json.dumps(analysis(8)))
            (scan/'hostility_stability.json').write_text(json.dumps({'num_probes':3,'top_k':{'64':{'mean_overlap':.49}}}))
            self.assertFalse(gate_decision(scan,verify)['pass'])

class RepoMigrationTests(unittest.TestCase):
    def test_text_migration_dryrun_and_apply(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'01_packages').mkdir()
            (p/'.gitignore').write_text('01_packages/*.zip\n!01_packages/qwen-persona-matrix.zip\n')
            (p/'00_README.md').write_text('# Research index\n')
            with zipfile.ZipFile(p/'01_packages/qwen-persona-matrix.zip','w') as z:
                z.writestr('project/README.md','# README\n')
                z.writestr('project/model.pt',b'\0\1\2')
            result=migrate(p,False)
            self.assertEqual(result['mode'],'dry-run')
            self.assertFalse((p/'03_unpacked').exists())
            migrated=migrate(p,True)
            self.assertEqual(migrated['mode'],'applied')
            self.assertTrue((p/'03_unpacked/qwen-persona-matrix/project/README.md').exists())
            self.assertFalse((p/'03_unpacked/qwen-persona-matrix/project/model.pt').exists())
            self.assertNotIn('!01_packages/qwen-persona-matrix.zip',(p/'.gitignore').read_text())
            self.assertIn('qwen-persona-matrix.zip',(p/'00_README.md').read_text())
            self.assertEqual(migrate(p,True)['mode'],'applied')

    def test_migration_rejects_archive_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'01_packages').mkdir()
            with zipfile.ZipFile(p/'01_packages/bad.zip','w') as z:z.writestr('../escape.txt','oops')
            with self.assertRaises(ValueError):migrate(p,False)

if __name__=='__main__':unittest.main()
