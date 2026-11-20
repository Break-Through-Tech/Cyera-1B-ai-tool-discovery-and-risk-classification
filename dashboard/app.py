"""Run from repository root: python -m streamlit run dashboard/app.py."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import pandas as pd
import streamlit as st
from ai_discovery.dashboard_data import ROOT,SAMPLE,ACTIVITY,load_report,analyze,flattened,csv_export

st.set_page_config(page_title='AI Usage & Governance',page_icon='🔎',layout='wide')
st.markdown('''<style>
.block-container {padding-top:2rem; max-width:1450px}
[data-testid="stMetric"] {border:1px solid #334155;border-radius:12px;padding:16px}
h1 {letter-spacing:-.03em} .stCaption {color:#94a3b8}
</style>''',unsafe_allow_html=True)
st.title('AI Usage & Governance')
st.write('Explore application usage, distinguish AI-focused tools, and investigate governance priorities.')
st.caption('Fictional enterprise demonstration · Evidence-based priorities, not breach probabilities')

@st.cache_data
def sample_bytes():return SAMPLE.read_bytes()

if 'report' not in st.session_state:
    st.session_state.report=load_report(sample_bytes())
    st.session_state.source_name='NovaTech saved Component 4 report'

with st.sidebar:
    st.header('Data source')
    mode=st.radio('Choose an input',['Saved NovaTech report','Analyze activity CSV','Open saved report'])
    activity=None;uploaded=None;metadata=None;demo=False;company=''
    if mode=='Analyze activity CSV':
        company=st.text_input('Fictional company',value='NovaTech')
        use_sample=st.checkbox('Use the sample activity CSV',value=True)
        if use_sample:
            activity=ACTIVITY.read_bytes()
            st.caption('30,000 synthetic activity events. Usage and risk are recalculated on Analyze.')
        else:
            file=st.file_uploader('Activity CSV',type=['csv'])
            if file:activity=file.getvalue()
        reuse=st.checkbox('Reuse exact saved product decisions',value=True,help='Reuses product classifications/categories, not old usage or company controls. It does not call Jev.')
        demo=st.checkbox('Use provided fictional controls and approvals',value=False,help='Explicitly adopts invented catalog approval values for this fictional company. These are not verified vendor controls.')
        metadata=st.file_uploader('Optional synthetic control metadata CSV',type=['csv'])
        st.caption('Metadata needs tool_name and domain; security fields remain synthetic. Uploading metadata alone does not adopt its approval values.')
    elif mode=='Open saved report':
        uploaded=st.file_uploader('Component 4 report JSON',type=['json'])
    if st.button('Analyze / load data',type='primary'):
        try:
            with st.spinner('Preparing findings…'):
                if mode=='Saved NovaTech report':
                    report=load_report(sample_bytes());source='NovaTech saved Component 4 report'
                elif mode=='Open saved report':
                    if uploaded is None:raise ValueError('Choose a report JSON file')
                    report=load_report(uploaded.getvalue());source=uploaded.name
                else:
                    if activity is None:raise ValueError('Choose an activity CSV')
                    report=analyze(activity,company,load_report(sample_bytes()) if reuse else None,demo,metadata.getvalue() if metadata else None)
                    source='Fresh offline activity analysis'
                st.session_state.report=report;st.session_state.source_name=source
                st.session_state.pop('question_result',None)
        except (ValueError,KeyError,OSError,UnicodeError) as error:st.error('Unable to analyze this input: '+str(error))
    st.divider()
    st.markdown('**Jev status: pending integration**')
    st.caption('No live Jev, Gemini or web calls run in this dashboard. Known decisions retain their saved source; unfamiliar tools stay unknown.')
    st.download_button('Download sample activity',ACTIVITY.read_bytes(),'novatech_usage.csv','text/csv')

report=st.session_state.report;context=report['dashboard_context'];rows=flattened(report)
st.subheader(context['company'])
st.caption('Viewing: '+st.session_state.source_name+' · '+context['mode'].replace('_',' '))
st.info(context['classification_note'])
columns=st.columns(5)
values=[len(rows),sum(r['ai_status']=='AI' for r in rows),sum(r['ai_status']=='Conventional' for r in rows),sum(r['ai_status']=='Unknown' for r in rows),sum(r['risk_level']=='high' for r in rows)]
for column,label,value in zip(columns,['Applications','AI-focused','Conventional','Unknown AI status','High priority'],values):column.metric(label,value)
st.caption('Overview counts cover the whole analyzed file. “Conventional” describes primary purpose, not the absence of AI features. High priority can apply to conventional applications too.')

with st.sidebar:
    st.header('Explore results')
    department=st.selectbox('Department',['All',*sorted({d for r in rows for d in r['departments']})])
    ai=st.selectbox('AI status',['All','AI','Conventional','Unknown'])
    priority=st.selectbox('Governance priority',['All','high','medium','low','unknown'])
    approval=st.selectbox('Approval',['All',*sorted({r['approval'] for r in rows})])
    search=st.text_input('Search application or endpoint')
visible=[r for r in rows if (department=='All' or department in r['departments']) and (ai=='All' or r['ai_status']==ai) and (priority=='All' or r['risk_level']==priority) and (approval=='All' or r['approval']==approval) and (not search or search.casefold() in (r['application']+' '+r['endpoint']).casefold())]
frame=pd.DataFrame([{**r,'departments':', '.join(r['departments'])} for r in visible])
overview,applications,investigation,about=st.tabs(['Overview','Applications','Investigate','Workflow & evidence'])
with overview:
    st.markdown('#### Usage and priorities')
    st.caption(f'{len(visible)} applications match the current filters. User counts are per application and must not be summed across tools.')
    a,b=st.columns(2)
    if visible:
        a.markdown('**Applications by priority**')
        a.bar_chart(pd.Series({level:sum(r['risk_level']==level for r in visible) for level in ('high','medium','low','unknown')},name='Applications'))
        b.markdown('**Applications by department**')
        b.bar_chart(pd.Series({d:sum(d in r['departments'] for r in visible) for d in sorted({d for r in visible for d in r['departments']})},name='Applications'))
    else:st.info('No applications match these filters.')
    summary=report.get('ingestion',{}).get('summary',{})
    st.write('Activity accounting:',{'input events':summary.get('input_rows'),'accepted':summary.get('accepted_events'),'rejected':summary.get('rejected_events')})
    daily=context.get('daily_events')
    if daily:
        st.markdown('**Recorded daily activity**')
        daily_frame=pd.DataFrame({'Date':pd.to_datetime(list(daily)),'Events':list(daily.values())}).set_index('Date')
        st.line_chart(daily_frame)
        st.caption('Event volume in synthetic input—not distinct users, adoption growth, or real-world behavior. Invalid/missing timestamps: '+str(context.get('events_missing_or_invalid_timestamp',0)))
    st.markdown('#### Applications to investigate')
    queue=[r for r in visible if r['needs_review'] or r['risk_level'] in ('high','medium')]
    if queue:st.dataframe(pd.DataFrame([{k:r[k] for k in ('application','ai_status','risk_level','risk_score','evidence_status')} for r in queue]),hide_index=True,width='stretch')
    else:st.caption('No review items under the current filters. This is not an assurance of safety.')

with applications:
    st.markdown('#### Application findings')
    if not frame.empty:
        st.dataframe(frame.drop(columns=['application_id']),hide_index=True,width='stretch')
    else:st.info('No matching applications.')
    st.caption('Purpose categories may be catalog-assigned or provisional model predictions. The source/status columns distinguish them.')
    st.download_button('Download filtered findings CSV',csv_export(visible),'application_findings.csv','text/csv')

with investigation:
    if visible:
        choices={r['application_id']:r['application']+' — '+r['endpoint'] for r in visible}
        selected=st.selectbox('Select an application',list(choices),format_func=choices.get)
        item=next(r for r in report['results'] if r['application_id']==selected);flat=next(r for r in visible if r['application_id']==selected)
        g=item['governance'];st.markdown('#### '+flat['application'])
        a,b,c=st.columns(3);a.metric('Priority',flat['risk_level']);b.metric('Score',flat['risk_score'] if flat['risk_score'] is not None else 'Unknown');c.metric('Recorded users',flat['users'] if flat['users'] is not None else 'Unknown')
        st.write('Departments:',', '.join(flat['departments']) or 'Unknown')
        st.write('AI decision:',flat['ai_status'],'· Source:',flat['classification_source'])
        st.write('Purpose:',flat['category'],'· Source:',flat['category_source'],'· Status:',flat['category_status'])
        st.markdown('**Why it was flagged**')
        reasons=g.get('reasons',[])
        if reasons:st.dataframe(pd.DataFrame([{'Rule':r['rule'],'Points':r['points'],'Explanation':r['explanation'],'Evidence':r.get('evidence_provenance','unknown')} for r in reasons]),hide_index=True,width='stretch')
        else:st.write('No triggered rules recorded. Missing evidence can still require review.')
        if g.get('missing_evidence_rules'):st.warning('Missing or pending evidence: '+', '.join(g['missing_evidence_rules']))
        if g.get('contains_synthetic_evidence'):st.warning('This assessment includes fictional activity or control settings. It is not a verified vendor-security assessment.')
        if g.get('data_quality_warnings'):
            for warning in g['data_quality_warnings']:st.warning(warning)
        st.caption(g.get('caveat',''))
        with st.expander('Control evidence and provenance'):
            fields=g.get('evidence',{})
            st.dataframe(pd.DataFrame([{'Control':k,'Value':str(v.get('value')),'Provenance':v.get('provenance','unknown'),'Source':v.get('source','Unknown')} for k,v in fields.items()]),hide_index=True,width='stretch')
        with st.expander('Full application record'):st.json(item)
    else:st.info('Change the filters to select an application.')

with about:
    st.markdown('#### What this demonstration does')
    st.write('Activity CSV → application identities → exact catalog / saved product decisions → purpose categories → usage-based and supplied-evidence governance rules → findings.')
    st.write('Saved-report mode shows an existing pipeline result. CSV mode freshly ingests events and recalculates usage and risk. It can reuse exact saved product decisions; those are explicitly labeled and are not new Jev inference.')
    st.write('Jev is planned for AI/non-AI classification. It is not connected in this version. No key, API billing, model training or web research is required for the offline demonstration.')
    st.markdown('**Limitations**')
    st.write('Synthetic usage can differ from real companies. Source-assigned labels and provisional predictions can be wrong. Shared domains and missing product evidence can leave applications unresolved. Scores are explained prototype priorities—not breach probabilities, compliance certification or proof of safety.')
    with st.expander('Rejected activity records'):
        rejected=report.get('ingestion',{}).get('rejected_rows',[])
        if rejected:st.dataframe(pd.DataFrame(rejected),hide_index=True,width='stretch')
        else:st.write('No rejected records in this result.')
    st.markdown('#### Optional local evidence questions')
    st.caption('Uses the report-search assistant locally. Requires the existing local MiniLM encoder and retrieval packages; no Gemini calls.')
    question=st.text_input('Question',placeholder='Why was Salesforce flagged?')
    if st.button('Find report evidence'):
        try:
            from ai_discovery.retrieval import build,EvidenceIndex,DEFAULT_ENCODER,load_encoder
            from ai_discovery.ask import answer
            if not DEFAULT_ENCODER.exists():raise ValueError('Local search encoder is not installed. Core dashboard analysis still works.')
            @st.cache_resource
            def search_encoder():return load_encoder(DEFAULT_ENCODER)
            encoder=search_encoder()
            with st.spinner('Retrieving cited report facts…'),tempfile.TemporaryDirectory(prefix='dashboard_evidence_') as temp:
                folder=Path(temp);source=folder/'analysis.json';source.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                build([source],folder/'index',DEFAULT_ENCODER,encoder)
                result=answer(EvidenceIndex(folder/'index',encoder),question,context['company'])
            st.session_state.question_result=result
        except (ValueError,OSError,ImportError,KeyError) as error:st.error(str(error))
    if st.session_state.get('question_result'):
        response=st.session_state.question_result;st.write(response['answer'])
        with st.expander('Question filters and report citations'):st.json(response)

st.divider()
st.download_button('Download complete analysis JSON',json.dumps(report,indent=2,allow_nan=False)+'\n','analysis.json','application/json')
st.caption('The export includes methods, reasons, evidence gaps and provenance. It does not turn unknown evidence into confirmed facts.')
