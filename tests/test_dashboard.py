import csv
import io
import json
import unittest
from ai_discovery.dashboard_data import SAMPLE,ACTIVITY,ROOT,load_report,analyze,flattened,csv_export


class DashboardTests(unittest.TestCase):
    def test_sample_source_and_counts_are_explicit(self):
        report=load_report(SAMPLE.read_bytes());rows=flattened(report)
        self.assertEqual(len(rows),21)
        self.assertEqual(sum(r['ai_status']=='AI' for r in rows),9)
        self.assertEqual(sum(r['ai_status']=='Conventional' for r in rows),12)
        self.assertEqual(report['dashboard_context']['online_calls_made'],0)
        sf=next(r for r in rows if r['application'].casefold()=='salesforce')
        self.assertEqual(sf['risk_score'],60)
        self.assertEqual(sf['ai_status'],'Conventional')

    def test_upload_recomputes_usage_and_does_not_inherit_cached_company_controls(self):
        reference=load_report(SAMPLE.read_bytes())
        content=b'application_name,domain,employee_id,department,bytes_uploaded\nSalesforce,salesforce.com,new_user,Finance,100\n'
        report=analyze(content,'Different Company',reference)
        row=report['results'][0]
        self.assertEqual(row['observed_identity']['event_count'],1)
        self.assertEqual(row['governance']['risk_score'],10)
        self.assertIsNone(row['governance']['evidence']['approved_status']['value'])
        self.assertEqual(row['governance']['company'],'Different Company')
        self.assertNotIn('new_user',json.dumps(report))

    def test_unknown_and_rejected_records_remain_visible_without_host_guessing(self):
        content=b'application_name,domain,employee_id,department,bytes_uploaded\nDifferent Product,chatgpt.com,u1,Finance,12000000\nUnfamiliar,unseen.example,u2,Engineering,bad\n,invalid,u3,HR,0\n'
        report=analyze(content,'Fictional',load_report(SAMPLE.read_bytes()))
        self.assertEqual(len(report['results']),2)
        self.assertTrue(all(r['ai_status']=='Unknown' for r in flattened(report)))
        self.assertEqual(report['ingestion']['summary']['accepted_events'],2)
        self.assertEqual(report['ingestion']['summary']['rejected_events'],1)
        self.assertTrue(report['ingestion']['rejected_rows'])

    def test_demo_controls_are_explicit_and_csv_exports_escape_formulas(self):
        report=analyze(ACTIVITY.read_bytes(),'NovaTech',load_report(SAMPLE.read_bytes()),True)
        sf=next(r for r in flattened(report) if r['application'].casefold()=='salesforce')
        self.assertEqual(sf['risk_score'],60)
        self.assertTrue(report['dashboard_context']['demo_controls'])
        self.assertEqual(report['dashboard_context']['online_calls_made'],0)
        self.assertIn("'=SUM",csv_export([{'name':'=SUM(1)'}]).decode('utf-8-sig'))

    def test_invalid_uploads_and_saved_report_scope_fail(self):
        for content in [b'name,domain\nTool,example.com\n',b'application_name,domain\n']:
            with self.assertRaises(ValueError):analyze(content,'Fictional')
        with self.assertRaises(ValueError):load_report(b'{"results":[]}')

    def test_dashboard_renders_and_upload_button_runs(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'dashboard/app.py')).run(timeout=30)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.metric[0].value,'10')
        provenance_headers={'AI label source','Purpose label source','Usage data provenance','Contains synthetic evidence'}
        tables=[frame.value for frame in app.dataframe if provenance_headers<=set(frame.value.columns)]
        self.assertEqual(len(tables),1)
        self.assertEqual(set(tables[0]['Usage data provenance']),{'synthetic'})
        self.assertTrue(tables[0]['Contains synthetic evidence'].all())
        app.sidebar.radio[1].set_value('NovaTech').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.metric[0].value,'9')
        app.sidebar.radio[0].set_value('Analyze activity CSV').run()
        app.sidebar.checkbox[0].set_value(True).run()
        next(s for s in app.sidebar.selectbox if s.label=='Sample activity').set_value('NovaTech').run()
        app.sidebar.button[0].click().run(timeout=60)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state['upload_run'].raw['dashboard_context']['mode'],'fresh_offline_analysis')
        self.assertEqual(len(app.session_state['upload_run'].apps),21)
        app.sidebar.text_input[-1].set_value('no matching application').run()
        self.assertEqual(len(app.exception),0)

if __name__=='__main__':unittest.main()
