"""v4: experimental-unit correctness and repository migration integration tests."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from choices import summarize_choices
from gate import gate_decision
from audit import file_hash
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from migrate_repo_packages import migrate
from dedup_check import check_duplicates, version_family


class ChoicesTaskLevelTests(unittest.TestCase):
    def test_mirrored_orders_not_independent_and_no_pvalues(self):
        tasks = [{'id': 't1','trait':'risk'}, {'id':'t2','trait':'risk'},
                 {'id':'t3','trait':'risk'}]
        arms = [{'arm_id':'r','trait':'risk'}]
        rows = []
        for i,task in enumerate(tasks):
            for order in (0, 1):
                base = False if order == 0 else True
                arm = True if order == 0 else False
                if i == 1 and order == 1: arm = None
                for name,value in [('baseline',base),('r',arm)]:
                    rows.append({'task_id':task['id'],'order':order,'arm_id':name,
                                 'trait_option_selected':value})
        result = summarize_choices(rows,tasks,arms)['r']
        self.assertEqual(result['n_tasks'],3)
        self.assertEqual(result['n_ordered_measurements'],6)
        self.assertEqual(len(result['task_effects']),3)
        self.assertIsNone(result['inferential_p_value'])
        self.assertIsNone(result['paired_sign_test_both_valid'])
        self.assertIn('DESCRIPTIVE ONLY', result['inference_status'])
        self.assertAlmostEqual(result['paired_selection_delta_total'],
                               sum(x['delta_trait_choice_total'] for x in result['task_effects'])/3)
        self.assertAlmostEqual(result['invalid_rate_delta_arm_minus_baseline'], 1/6)

    def test_duplicated_orders_cannot_generate_new_task_n(self):
        tasks = [{'id':'t1','trait':'a'}]
        arms = [{'arm_id':'a1','trait':'a'}]
        rows=[]
        for order in (0,1):
            for aid, v in [('baseline',False),('a1',True)]:
                rows.append({'task_id':'t1','order':order,'arm_id':aid,
                             'trait_option_selected':v})
        data=summarize_choices(rows,tasks,arms)['a1']
        self.assertEqual(data['n_tasks'],1)
        self.assertEqual(data['n_ordered_measurements'],2)
        self.assertIsNone(data['task_delta_sd'])
        self.assertIsNone(data['inferential_p_value'])


class MigrationV4Tests(unittest.TestCase):
    def make_workspace(self, root):
        p=Path(root)/'qwentest'
        (p/'01_packages').mkdir(parents=True)
        (p/'.gitignore').write_text(
            '01_packages/*.zip\n# Keep this unrelated note\n'
            '# Legacy exception for qwen-persona-matrix.zip\n'
            '# Temporarily unignore 01_packages archive\n'
            '!01_packages/qwen-persona-matrix.zip\n')
        (p/'00_README.md').write_text(
            '# Qwen research\n\n'
            '## Previous MD5 records\n\n'
            '| Package | MD5 |\n|---|---|\n'
            '| `01_packages/qwen-persona-matrix.zip` | `abcde` |\n\n'
            '## Other data\n\n| Name | Value |\n|---|---|\n| A | 1 |\n')
        for name,text in [('qwen-gradient-discovery.zip','A'),
                          ('qwen-gradient-discovery-v2.zip','A'),
                          ('qwen-gradient-discovery-v3.zip','B'),
                          ('qwen-persona-matrix.zip','C')]:
            with zipfile.ZipFile(p/'01_packages'/name,'w') as z:
                z.writestr('project/notes.txt',text)
        return p

    def test_migration_idempotent_legacy_table_and_comments(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make_workspace(d)
            with self.assertRaisesRegex(FileNotFoundError,'workspace'):
                migrate(p.parent,False)
            dry=migrate(p,False)
            self.assertTrue(dry['legacy_md5_table_retired'])
            self.assertFalse((p/'03_unpacked').exists())
            migrate(p,True)
            readme=(p/'00_README.md').read_text()
            self.assertEqual(readme.count('| Archive | MD5 | SHA256 |'),1)
            self.assertNotIn('| Package | MD5 |',readme)
            self.assertIn('## Other data',readme)
            ignore=(p/'.gitignore').read_text()
            self.assertNotIn('Legacy exception',ignore)
            self.assertNotIn('Temporarily unignore',ignore)
            self.assertIn('Keep this unrelated note',ignore)
            self.assertNotIn('!01_packages/qwen-persona-matrix.zip',ignore)
            def inventory():
                return {str(q.relative_to(p)):q.read_bytes() for q in p.rglob('*') if q.is_file()}
            before=inventory()
            migrate(p,True)
            self.assertEqual(before,inventory())
            self.assertTrue((p/'03_unpacked/qwen-gradient-discovery-v3/project/notes.txt').is_file())
            self.assertTrue((p/'03_unpacked/qwen-gradient-discovery-v2/project/notes.txt').is_file())

    def test_mixed_legacy_table_refused_instead_of_dropping_other_records(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make_workspace(d)
            (p/'00_README.md').write_text('| File | MD5 |\n|---|---|\n'
                                                  '| `01_packages/a.zip` | x |\n'
                                                  '| `report.pdf` | y |\n')
            with self.assertRaisesRegex(ValueError,'mixes ZIP packages'):
                migrate(p,False)
            self.assertFalse((p/'03_unpacked').exists())

    def test_duplicate_checker_respects_family(self):
        self.assertEqual(version_family('qwen-gradient-discovery-v3.1'),
                         version_family('qwen-gradient-discovery'))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'03_unpacked'
            for name in ('qwen-gradient-discovery', 'qwen-gradient-discovery-v2',
                         'qwen-persona-matrix'):
                dest=root/name/'project';dest.mkdir(parents=True)
                (dest/'file.txt').write_text('shared')
            result=check_duplicates(root)
            self.assertEqual(result['n_cross_family_conflicts'],1)
            (root/'qwen-persona-matrix/project/file.txt').write_text('unique')
            result=check_duplicates(root)
            self.assertEqual(result['n_cross_family_conflicts'],0)
            self.assertEqual(result['n_allowed_hash_groups'],1)


class GateV4Tests(unittest.TestCase):
    def test_missing_loo_gives_actionable_failure(self):
        with tempfile.TemporaryDirectory() as d:
            scan=Path(d)/'scan';verify=Path(d)/'verify';scan.mkdir();verify.mkdir()
            (scan/'manifest.json').write_text('{}')
            (verify/'manifest.json').write_text('{}')
            (verify/'analysis.json').write_text('{}')
            with self.assertRaisesRegex(FileNotFoundError,'python stability.py'):
                gate_decision(scan,verify)

if __name__=='__main__': unittest.main()
