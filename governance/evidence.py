"""Component 4 evidence assembly. Exact identities; no inherited risk labels."""
import copy
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from ai_discovery.ingest import clean_name, normalize_endpoint
from .risk import FIELDS, normalize_metadata, read_csv


def options(parser):
    parser.add_argument('--governance-catalog', type=Path, action='append', default=[],
                        help='Dataset B CSV; repeat to include AI and non-AI catalogs')
    parser.add_argument('--governance-catalog-provenance', choices=['synthetic','verified','unknown'], default='unknown')
    parser.add_argument('--catalog-approval-company', help='Explicitly adopt synthetic catalog approvals for this company')
    parser.add_argument('--governance-scenarios', type=Path)
    parser.add_argument('--governance-online', action='store_true', help='Research missing vendor controls with Gemini')
    parser.add_argument('--governance-repair-only',action='store_true',help='Re-extract saved research only; never issue search calls')
    parser.add_argument('--governance-plans',type=Path,help='Explicit company plans keyed by application_id')
    parser.add_argument('--governance-timeout-seconds',type=float,default=90)
    parser.add_argument('--governance-max-apps',type=int,help='Limit uncached apps for a live smoke test')
    parser.add_argument('--governance-api-model', default='gemini-3.8-flash')
    parser.add_argument('--governance-research-cache', type=Path, help='Read a previous governance research cache; writes a new snapshot to output')


def input_paths(args):
    return [*args.governance_catalog,args.governance_scenarios,args.governance_research_cache,args.governance_plans]


def assemble(records, company, catalogs=(), catalog_provenance='unknown', approval_company=None,
             scenarios=None, overrides=None):
    if catalog_provenance not in ('synthetic','verified','unknown'):
        raise ValueError('Invalid catalog provenance')
    if approval_company and (approval_company != company or catalog_provenance != 'synthetic'):
        raise ValueError('Catalog approval adoption requires matching company and synthetic provenance')
    index=defaultdict(list); warnings=[]
    for path in catalogs:
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for number,row in enumerate(read_csv(path),1):
            try:
                name=clean_name(row['tool_name']).casefold()
                endpoint=normalize_endpoint(row['domain'])[4]
                if not name:raise ValueError('Missing catalog name')
            except (KeyError,ValueError):
                warnings.append({'source':str(path),'row':number,'reason':'invalid_catalog_identity'})
                continue
            index[(name,endpoint)].append((row,dict(source=str(path),source_sha256=digest,row=number)))
    result={'version':1,'applications':{}}
    endpoints=defaultdict(list)
    for record in records:
        app=record['observed_identity']; key=record['application_id']
        endpoints[app['observed_endpoint']].append(key)
        rows=index.get((app['normalized_name'],app['observed_endpoint']),[])
        fields={}
        for field in FIELDS:
            if field=='approved_status' and not approval_company:continue
            choices=[]
            for row,source in rows:
                value=normalize_metadata(row)[field]
                if value is not None:choices.append((value,source))
            if not choices:continue
            values={json.dumps(value,sort_keys=True) for value,_ in choices}
            if len(values)>1:
                warnings.append({'application_id':key,'field':field,'reason':'conflicting_catalog_values'})
                fields[field]={'value':None,'provenance':'unknown','reason':'conflicting_catalog_values',
                               'candidates':[dict(value=v,**s) for v,s in choices]}
                continue
            value,source=choices[0]
            fields[field]=dict(value=value,provenance=catalog_provenance,**source)
            if field=='approved_status':fields[field]['company']=company
        result['applications'][key]=fields
    if scenarios:
        fixture=json.loads(Path(scenarios).read_text(encoding='utf-8'))
        if fixture.get('synthetic') is not True or company not in fixture.get('companies',{}):
            raise ValueError('Scenario must be synthetic and contain the requested company')
        digest=hashlib.sha256(Path(scenarios).read_bytes()).hexdigest()
        for endpoint,fields in fixture['companies'][company].items():
            if set(fields)-set(FIELDS):raise ValueError('Scenario may contain governance fields only')
            matches=endpoints.get(normalize_endpoint(endpoint)[4],[])
            if len(matches)!=1:
                warnings.append({'endpoint':endpoint,'reason':'scenario_identity_absent_or_ambiguous'})
                continue
            for field,value in fields.items():
                result['applications'][matches[0]][field]=dict(value=value,provenance='synthetic',
                    source=str(scenarios),source_sha256=digest,company=company)
    if overrides is not None:
        if overrides.get('version')!=1 or set(overrides)-{'version','applications'} or not isinstance(overrides.get('applications'),dict):
            raise ValueError('Expected explicit evidence version 1')
        if set(overrides['applications'])-set(result['applications']):raise ValueError('Evidence identity absent from ingestion')
        for key,fields in overrides['applications'].items():
            if not isinstance(fields,dict):raise ValueError('Evidence fields must be an object')
            result['applications'][key].update(copy.deepcopy(fields))
    return result,warnings


def prepare(records, company, args, overrides=None):
    evidence,warnings=assemble(records,company,args.governance_catalog,args.governance_catalog_provenance,
                               args.catalog_approval_company,args.governance_scenarios,overrides)
    plans={}
    if args.governance_plans:
        document=json.loads(args.governance_plans.read_text(encoding='utf-8'))
        if document.get('version')!=1 or not isinstance(document.get('applications'),dict):raise ValueError('Invalid governance plans schema')
        plans=document['applications']
        if set(plans)-{r['application_id'] for r in records}:raise ValueError('Company plan identity absent from ingestion')
        for plan in plans.values():
            if not isinstance(plan,dict) or not company or plan.get('company')!=company or plan.get('provenance') not in ('synthetic','verified') or not all(isinstance(plan.get(f),str) and plan[f].strip() for f in ('plan','source')):
                raise ValueError('Company plans require matching company, plan, source and synthetic/verified provenance')
    if args.governance_repair_only and not args.governance_online:
        raise ValueError('--governance-repair-only requires --governance-online for extraction calls')
    outcomes=[]; cache=None
    if args.governance_online:
        from .research import enrich_missing
        checkpoint=args.out.with_name(args.out.name+'.governance-checkpoint.json')
        # Output-derived checkpoints must never replace an explicit input.
        inputs=[]
        for value in vars(args).values():
            if isinstance(value,Path):inputs.append(value)
            elif isinstance(value,list):inputs.extend(v for v in value if isinstance(v,Path))
        if checkpoint.resolve() in {p.resolve() for p in inputs}:
            raise ValueError('Governance checkpoint conflicts with an input/output path')
        evidence,outcomes,cache=enrich_missing(records,evidence,args.governance_api_model,args.governance_research_cache,
            timeout_seconds=args.governance_timeout_seconds,checkpoint_path=checkpoint,max_apps=args.governance_max_apps,repair_only=args.governance_repair_only,company_plans=plans)
    return evidence,dict(warnings=warnings,research_outcomes=outcomes),cache
