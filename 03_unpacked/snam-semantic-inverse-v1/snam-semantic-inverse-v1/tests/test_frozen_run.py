import json,tempfile,unittest
from pathlib import Path
from audit import guarded_manifest
from protocol import frozen_plan,require_plan,stage_plan

class FrozenRunTests(unittest.TestCase):
    def test_full_local_freeze_guard_and_stage_idempotency(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model=p/'model';model.mkdir()
            (model/'config.json').write_text('{"model_type":"toy"}')
            (model/'model-00001.safetensors').write_bytes(b'fixture-not-a-real-weight-shard')
            sem=p/'spec.json';sem.write_text(json.dumps({'traits':{'t':{'description':'x','neutral_description':'y'}}}))
            probes=p/'pairs.jsonl'
            probes.write_text('\n'.join(json.dumps({'id':s,'trait':'t','split':s,'prompt':'x',
                                       'positive':'abc','negative':'def'}) for s in ('discovery','heldout'))+'\n')
            plan=frozen_plan(model,sem,probes)
            dest=p/'out';guarded_manifest(dest,plan)
            found=require_plan(dest,model_check=False)
            self.assertEqual(found['semantic_sha256'],plan['semantic_sha256'])
            stage_plan(dest,'discovery',{'top_k':4});stage_plan(dest,'discovery',{'top_k':4})
            with self.assertRaises(RuntimeError):stage_plan(dest,'discovery',{'top_k':9})
            probes.write_text(probes.read_text()+'\n')
            with self.assertRaises(RuntimeError):require_plan(dest,model_check=False)

if __name__=='__main__':unittest.main()
