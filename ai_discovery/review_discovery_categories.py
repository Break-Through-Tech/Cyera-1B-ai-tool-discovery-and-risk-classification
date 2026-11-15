"""Evaluate a saved category model on existing online-search evidence.

No web calls, training, discovery-label changes, or catalog assignments.
Run from the repository root; see README.md for exact commands.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 'discovery-category-review-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        if len(fields)!=len(set(fields)):
            raise ValueError(f'Duplicate columns: {path}')
        rows=list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Malformed CSV: {path}')
    return rows


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def import_file(path,name):
    if not path.is_file():
        raise ValueError(f'Missing helper: {path}; see --prep-script / --training-script')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(name,endpoint,prep):
    host,path=prep.endpoint(endpoint)
    if not prep.key(name) or not host:
        raise ValueError('Invalid discovery/review name or endpoint')
    return json.dumps([prep.key(name),host+path],ensure_ascii=False,separators=(',',':'))


def load_reviews(path,prep,taxonomy):
    reviews={}
    for row in csv_rows(path) if path else []:
        required={'application_name','endpoint','evidence_sha256','reviewed_category','reviewer','reason'}
        if not required<=row.keys():
            raise ValueError('Review CSV needs '+', '.join(sorted(required)))
        category=row['reviewed_category'].strip()
        if not category:
            continue
        if category not in taxonomy or not row['reviewer'].strip() or not row['reason'].strip():
            raise ValueError('Reviewed categories require an exact taxonomy label, reviewer and reason')
        key=identity(row['application_name'],row['endpoint'],prep)
        if key in reviews:
            raise ValueError('Duplicate reviewed identity')
        reviews[key]=row
    return reviews


def candidate(record,prep,selected_report=None):
    """Fail closed for non-AI, unknown, identity ambiguity or multiple reports."""
    evidence_hash=fingerprint(record.get('reports',[]))
    result={'evidence_sha256':evidence_hash,'model_text':'','description_only_text':'','sections':{},
            'report_index':None,'status':'pending_ai_detection'}
    if record.get('classification_status')!='classified':
        return result
    if record.get('is_ai_tool') is False:
        result['status']='not_applicable'; return result
    if record.get('is_ai_tool') is not True:
        return result
    if record.get('technical_error') or record.get('identity_supported') is not True:
        result['status']='pending_identity_review'; return result
    reports=record.get('reports',[])
    if selected_report is None:
        if len(reports)!=1:
            result['status']='needs_report_selection' if reports else 'missing_evidence'
            return result
        selected_report=0
    if not 0<=selected_report<len(reports):
        raise ValueError('Selected report_index is outside the current reports list')
    extracted=prep.extract_report(reports[selected_report])
    result.update(report_index=selected_report,model_text=extracted['model_text'],
                  description_only_text=extracted['description_only_text'],sections=extracted['sections'])
    result['status']='ready' if extracted['extraction_status']=='extracted' and extracted['model_text'].strip() else 'pending_purpose_evidence'
    return result


def validate_review(review,record,evidence_hash):
    if not review: return
    if review['evidence_sha256']!=evidence_hash:
        raise ValueError('Human category review is stale; evidence changed')
    if record.get('is_ai_tool') is not True or record.get('classification_status')!='classified' or record.get('identity_supported') is not True or record.get('technical_error'):
        raise ValueError('Category review cannot override unresolved/non-AI detection or identity')


def overlap_indexes(rows,splits,prep):
    indexes={k:defaultdict(set) for k in ['name','endpoint','description','text']}
    groups=defaultdict(set)
    for r in rows:
        sid=r['source_id']
        if sid not in splits:
            raise ValueError('Prepared source missing from split manifest')
        groups[r['group_id']].add(splits[sid]['partition'])
        host,path=prep.endpoint(r.get('authority','')+r.get('product_path',''))
        signals={'name':prep.key(r.get('tool_name')),'endpoint':host+path if host else '',
                 'description':prep.key(r.get('description','')),'text':prep.key(r['model_text'])}
        for kind,value in signals.items():
            if value: indexes[kind][value].add(r['group_id'])
    if any(len(v)>1 for v in groups.values()):
        raise ValueError('A group crosses development/test partitions')
    return indexes,groups


def detect_overlap(record,evidence,indexes,partitions,prep):
    host,path=prep.endpoint(record['endpoint'])
    # Even a shared-host signal conservatively flags possible overlap, not identity.
    signals={'name':prep.key(record['application_name']),'endpoint':host+path,
             'description':prep.key(evidence['description_only_text']),'text':prep.key(evidence['model_text'])}
    matches=[]; parts=set()
    for kind,value in signals.items():
        for group in sorted(indexes[kind].get(value,set())) if value else []:
            found=sorted(partitions[group]); parts.update(found)
            matches.append({'signal':kind,'group_id':group,'partitions':found})
    status='development_overlap' if 'development' in parts else 'prior_test_overlap' if 'test' in parts else 'no_detected_overlap'
    return status,matches


def evaluation(results,taxonomy):
    """Only explicit reviewed labels are truth; catalog/model labels are never truth."""
    from sklearn.metrics import accuracy_score,classification_report,f1_score
    out={}
    for name,eligible in [('all_reviewed',[r for r in results if r['reviewed_category']]),
                          ('no_detected_overlap',[r for r in results if r['reviewed_category'] and r['overlap_status']=='no_detected_overlap']),
                          ('development_overlap',[r for r in results if r['reviewed_category'] and r['overlap_status']=='development_overlap']),
                          ('prior_test_overlap',[r for r in results if r['reviewed_category'] and r['overlap_status']=='prior_test_overlap'])]:
        decided=[r for r in eligible if r['suggested_category']]
        data={'reviewed_rows':len(eligible),'predicted_rows':len(decided),
              'coverage':len(decided)/len(eligible) if eligible else None}
        if decided:
            y=[r['reviewed_category'] for r in decided]; pred=[r['suggested_category'] for r in decided]
            present=[c for c in taxonomy if c in set(y)]
            data.update(accuracy=float(accuracy_score(y,pred)),
                        macro_f1_all14=float(f1_score(y,pred,labels=taxonomy,average='macro',zero_division=0)),
                        macro_f1_reviewed_classes=float(f1_score(y,pred,labels=present,average='macro',zero_division=0)),
                        reviewed_classes=present,
                        per_category=classification_report(y,pred,labels=taxonomy,output_dict=True,zero_division=0))
        out[name]=data
    return out


class Predictor:
    def __init__(self,bundle,training,encoder_override):
        self.bundle=bundle; self.training=training; self.encoder=None
        self.encoder_override=encoder_override

    def predict(self,text):
        import numpy as np
        b=self.bundle
        if b['model_kind']=='tfidf':
            vector=b['estimator'].named_steps['tfidf'].transform([text])
            if vector.nnz==0:
                return {'category':None,'reason':'no_known_vocabulary','top_three':[]}
            features=[text]
        elif b['model_kind']=='embeddings':
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer
                meta=b['embedding']; path=self.encoder_override or Path(meta['encoder_path'])
                if meta.get('encoding')!=self.training.EMBEDDING_VERSION or self.training.snapshot_hash(path)!=meta['snapshot_sha256']:
                    raise ValueError('Encoder content or encoding contract mismatch')
                self.encoder=SentenceTransformer(str(path),local_files_only=True,trust_remote_code=False,device='cpu')
            features,_=self.training.encode_texts(self.encoder,[text])
        else:
            raise ValueError('Unsupported model kind')
        scores=b['estimator'].predict_proba(features)[0]
        order=np.argsort(scores)[::-1][:3]
        return {'category':str(b['estimator'].classes_[order[0]]),'reason':'provisional_model_suggestion',
                'top_three':[{'category':str(b['estimator'].classes_[i]),'score_uncalibrated':float(scores[i])} for i in order]}


def atomic_cache(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as f:
        tmp=Path(f.name); json.dump(data,f,ensure_ascii=False,allow_nan=False)
    os.replace(tmp,path)


def run(args):
    if args.out.exists(): raise ValueError('Output directory exists; choose a new --out')
    prep=import_file(args.prep_script,'category_preparation_contract')
    training=import_file(args.training_script,'category_training_contract')
    manifest=read_json(args.model_dir/'manifest.json')
    if not (args.prepared/'training_rows.csv').is_file():
        matches=[p.parent for p in args.prepared.rglob('training_rows.csv')
                 if (p.parent/'manifest.json').is_file() and sha(p)==manifest['config']['data_sha256']]
        if len(matches)!=1:
            raise ValueError('Could not uniquely locate the matching preparation run under '+str(args.prepared)+
                             '; set --prepared to its exact directory')
        args.prepared=matches[0]
        print('Matched preparation directory:',args.prepared)
    rows,taxonomy,_=training.load_prepared(args.prepared)
    if manifest['config']['data_sha256']!=sha(args.prepared/'training_rows.csv') or manifest['config']['taxonomy_sha256']!=sha(args.prepared/'taxonomy.json'):
        raise ValueError('Prepared data/taxonomy do not match the saved model run')
    if manifest['config']['text_contract_version']!=prep.TEXT_VERSION:
        raise ValueError('Training/extraction text contracts differ')
    model_path=args.model_dir/(args.model+'.joblib')
    for path in [model_path,args.model_dir/'split_manifest.csv']:
        if manifest['artifacts'].get(path.name)!=sha(path): raise ValueError('Model/split artifact hash mismatch')
    split_rows=csv_rows(args.model_dir/'split_manifest.csv')
    splits={r['source_id']:r for r in split_rows}
    if len(splits)!=len(split_rows) or set(splits)!={r['source_id'] for r in rows}:
        raise ValueError('Split IDs disagree with prepared rows')
    for r in rows:
        s=splits[r['source_id']]
        if s['group_id']!=r['group_id'] or s['category']!=r['category'] or s['partition'] not in {'development','test'}:
            raise ValueError('Split metadata mismatch')
    indexes,partitions=overlap_indexes(rows,splits,prep)
    discovery=read_json(args.discovery)
    if not isinstance(discovery.get('results'),list): raise ValueError('--discovery requires a classify.py report with results array')
    reviews=load_reviews(args.reviews,prep,taxonomy)
    selections={}
    for r in csv_rows(args.report_selections) if args.report_selections else []:
        key=identity(r['application_name'],r['endpoint'],prep)
        if key in selections: raise ValueError('Duplicate report selection')
        selections[key]=(int(r['report_index']),r['evidence_sha256'])
    keys=[identity(r['application_name'],r['endpoint'],prep) for r in discovery['results']]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate normalized discovery identities')
    if set(reviews)-set(keys) or set(selections)-set(keys): raise ValueError('Review/selection identities absent from this report')
    cache=read_json(args.cache) if args.cache.exists() else {'version':1,'entries':{}}
    if cache.get('version')!=1 or not isinstance(cache.get('entries'),dict): raise ValueError('Unsupported prediction cache')
    predictor=None; results=[]; updates=False
    for key,record in zip(keys,discovery['results']):
        evidence_hash=fingerprint(record.get('reports',[]))
        selected=None
        if key in selections:
            selected,saved_hash=selections[key]
            if saved_hash!=evidence_hash: raise ValueError('Report selection is stale; evidence changed')
        evidence=candidate(record,prep,selected)
        overlap,matches=detect_overlap(record,evidence,indexes,partitions,prep)
        review=reviews.get(key)
        validate_review(review,record,evidence_hash)
        result=dict(application_name=record['application_name'],endpoint=record['endpoint'],
            is_ai_tool=record.get('is_ai_tool'),ai_classification_status=record.get('classification_status'),
            ai_decision_source=record.get('decision_source'),identity_mode=record.get('identity_mode'),
            category_status=evidence['status'],suggested_category=None,category_source=None,
            top_three=[],cache_hit=False,overlap_status=overlap,overlap_matches=matches,
            reviewed_category=review['reviewed_category'].strip() if review else '',
            reviewer=review['reviewer'] if review else '',review_reason=review['reason'] if review else '',
            **evidence)
        if evidence['status']=='ready' and not args.extract_only:
            cache_key=fingerprint({'version':VERSION,'identity':key,'evidence':evidence_hash,
                'text':evidence['model_text'],'model':sha(model_path),'taxonomy':taxonomy,
                'extractor_sha256':sha(args.prep_script),'training_helper_sha256':sha(args.training_script),
                'detection':{k:record.get(k) for k in ['is_ai_tool','classification_status','identity_supported','identity_mode']}})
            prediction=cache['entries'].get(cache_key)
            if prediction is not None:
                result['cache_hit']=True
            else:
                if predictor is None:
                    import joblib
                    expected=manifest['config']['packages']['scikit-learn']
                    if importlib.metadata.version('scikit-learn')!=expected:
                        raise ValueError(f'Use training scikit-learn version {expected} to load this model')
                    bundle=joblib.load(model_path)
                    if bundle['run_id']!=manifest['run_id'] or bundle['taxonomy']!=taxonomy or bundle['text_contract_version']!=prep.TEXT_VERSION:
                        raise ValueError('Model bundle metadata mismatch')
                    predictor=Predictor(bundle,training,args.encoder_dir)
                prediction=predictor.predict(evidence['model_text'])
                cache['entries'][cache_key]=prediction; updates=True
            if prediction['category'] is not None and prediction['category'] not in taxonomy:
                raise ValueError('Prediction outside taxonomy')
            result.update(suggested_category=prediction['category'],top_three=prediction['top_three'],
                          category_source='model' if prediction['category'] else None,
                          category_status='provisional_prediction' if prediction['category'] else prediction['reason'])
        elif evidence['status']=='ready':
            result['category_status']='extracted_only'
        results.append(result)
        print(f'{record["application_name"]}: {result["suggested_category"] or result["category_status"]} [{overlap}]')
    report={'version':VERSION,'scope':'Diagnostic evaluation of researched products only; no catalog or enterprise report changes.',
            'summary':dict(Counter(r['category_status'] for r in results)),'results':results,
            'limitations':['Model suggestions are provisional, not accepted catalog assignments.',
                'No detected overlap is not proof of an unseen product; aliases and paraphrases may be missed.',
                'Development overlaps are seen during fitting; prior-test overlaps are previously evaluated products, not a fresh benchmark.',
                'Review labels independently, preferably using the blind review template before inspecting predictions.',
                'Scores are uncalibrated; no arbitrary acceptance threshold is applied.']}
    metrics=evaluation(results,taxonomy) if reviews and not args.extract_only else {'status':'No evaluation: supply human labels and run inference.'}
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'category_review.json',report)
    write_json(args.out/'review_metrics.json',metrics)
    flat=[dict(r,top_three=json.dumps(r['top_three'],ensure_ascii=False)) for r in results]
    write_csv(args.out/'category_predictions.csv',flat,['application_name','endpoint','is_ai_tool','ai_classification_status',
              'category_status','suggested_category','category_source','top_three','overlap_status','reviewed_category',
              'cache_hit','report_index','evidence_sha256','model_text'])
    blind=[dict(r,reason=r['review_reason']) for r in results if r['is_ai_tool'] is True and r['ai_classification_status']=='classified']
    write_csv(args.out/'human_review_template.csv',blind,['application_name','endpoint','evidence_sha256','reviewed_category','reviewer','reason','model_text'])
    write_csv(args.out/'report_selection_template.csv',[dict(r,report_index='') for r in results if r['category_status']=='needs_report_selection'],
              ['application_name','endpoint','evidence_sha256','report_index'])
    write_json(args.out/'taxonomy.json',{'categories':taxonomy})
    inputs={k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in [('discovery',args.discovery),('model',model_path),
            ('split',args.model_dir/'split_manifest.csv'),('prepared',args.prepared/'training_rows.csv'),
            ('prep_script',args.prep_script),('training_script',args.training_script),('reviews',args.reviews),('selections',args.report_selections)] if p}
    write_json(args.out/'manifest.json',{'version':VERSION,'inputs':inputs,'extract_only':args.extract_only,
                                      'model_run_id':manifest['run_id'],'script_sha256':sha(Path(__file__))})
    if updates: atomic_cache(args.cache,cache)
    print('Saved review outputs:',args.out.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--discovery',type=Path,default=Path('data/processed/ai_discovery_classification.json'))
    p.add_argument('--model-dir',type=Path,default=Path('models/categories/kaggle14_run1'))
    p.add_argument('--model',choices=['winner','tfidf','embeddings'],default='winner')
    p.add_argument('--prepared',type=Path,default=Path('data/processed/category'),help='Preparation directory or parent; matching run located by training-data hash')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reviews',type=Path)
    p.add_argument('--report-selections',type=Path)
    p.add_argument('--cache',type=Path,default=Path('data/cache/category_discovery_predictions.json'))
    p.add_argument('--prep-script',type=Path,default=Path('preprocessing_scripts/prepare_category_data.py'))
    p.add_argument('--training-script',type=Path,default=Path('category_training/train_categories.py'))
    p.add_argument('--encoder-dir',type=Path)
    p.add_argument('--extract-only',action='store_true',help='Create blind review text without loading joblib or predicting')
    args=p.parse_args()
    sources=[args.discovery,args.model_dir/'manifest.json',args.model_dir/(args.model+'.joblib'),
             args.model_dir/'split_manifest.csv',args.prepared/'training_rows.csv',args.prepared/'manifest.json',
             args.prepared/'taxonomy.json',args.prep_script,args.training_script,args.reviews,args.report_selections]
    if args.cache.resolve() in {s.resolve() for s in sources if s}:
        p.error('Prediction cache must not overwrite an input file')
    try: run(args)
    except (OSError,ValueError,KeyError,ImportError) as exc: p.exit(1,f'Error: {exc}\n')


if __name__=='__main__': main()
