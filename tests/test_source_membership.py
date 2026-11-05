import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('membership',ROOT/'preprocessing_scripts/label_source_membership.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SourceMembershipTests(unittest.TestCase):
    def test_every_source_record_retained_and_label_is_source_defined(self):
        rows,conflicts,summary=m.label(m.SOURCES)
        self.assertEqual(len(rows),17292)
        self.assertEqual(summary['class_counts'],{'1':16762,'0':530})
        self.assertEqual(len({r['record_id'] for r in rows}),len(rows))
        for source,path,target in m.SOURCES:
            with path.open(encoding='utf-8-sig') as f:
                originals={r['tool_id']:r for r in csv.DictReader(f)}
            selected=[r for r in rows if r['source_dataset']==source]
            self.assertEqual({r['source_record_id'] for r in selected},set(originals))
            for r in selected:
                self.assertEqual(r['is_ai_tool'],target)
                self.assertEqual(r['description'],originals[r['source_record_id']]['description'])
                self.assertEqual(r['original_is_ai_tool'],originals[r['source_record_id']]['is_ai_tool'])

    def test_outputs_reproduce(self):
        files,_=m.build()
        for name,data in files.items():self.assertEqual((m.OUT/name).read_bytes(),data)

    def test_conflicts_flag_without_relabeling_or_dropping(self):
        with tempfile.TemporaryDirectory() as d:
            sources=[]
            for source,target in [('ai',1),('non_ai',0)]:
                path=Path(d)/(source+'.csv')
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['tool_id','tool_name','domain','description','is_ai_tool']);w.writeheader()
                    w.writerow(dict(tool_id='same_id',tool_name='Same App',domain='https://example.com/',description='',is_ai_tool='uncertain'))
                sources.append((source,path,target))
            rows,conflicts,summary=m.label(sources)
            self.assertEqual([r['is_ai_tool'] for r in rows],[1,0])
            self.assertTrue(all(r['potential_label_conflict'] for r in rows))
            self.assertEqual({r['signal'] for r in conflicts},{'name','domain'})
            self.assertEqual(len(rows),2)
            self.assertEqual(len({r['record_id'] for r in rows}),2)

if __name__=='__main__':unittest.main()
