"""Small synthetic tests. --guards-only runs without scikit-learn.

Creates an isolated test directory below the working directory and retains it for
inspection. Synthetic scores are not project evaluation results.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import train_categories as training

LABELS = ['Ai And Machine Learning','Business And Finance','Communication And Support',
          'Creative And Entertainment','Data And Analytics','Education And Learning',
          'Industry Specific','Marketing And Sales','Media And Content Creation',
          'Operations And Management','Personal And Lifestyle','Security And Privacy',
          'Specialized Technologies','Technology And Development']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--guards-only',action='store_true')
    args = p.parse_args()
    root = Path(tempfile.mkdtemp(prefix='category_training_test_',dir=Path.cwd())).resolve()
    prepared = root/'prepared'
    prepared.mkdir()
    rows=[]
    for i,label in enumerate(LABELS):
        for j in range(30):
            text=f'Purposeword{i} performs useful specialized tasks for example {j}.'
            rows.append(dict(source_id=f'{i}_{j}',group_id=f'group_{i}_{j}',category=label,
                model_text=text,description_only_text=text,text_origin='description_plus_primary_task',
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),text_contract_version=training.TEXT_VERSION,
                eligible='True',tool_name=f'Example {i} {j}'))
    training.write_csv(prepared/'training_rows.csv',rows,list(rows[0]))
    taxonomy={'version':'original_kaggle_14','categories':LABELS,
              'sha256':hashlib.sha256(json.dumps(LABELS).encode()).hexdigest()}
    training.write_json(prepared/'taxonomy.json',taxonomy)
    manifest={'text_contract_version':training.TEXT_VERSION,'feature_column':'model_text',
              'outputs':{n:training.sha(prepared/n) for n in ['training_rows.csv','taxonomy.json']}}
    training.write_json(prepared/'manifest.json',manifest)
    loaded,labels,_=training.load_prepared(prepared)
    assert len(loaded)==420 and labels==LABELS
    # Tampered output must be rejected, not silently used for training.
    file=prepared/'training_rows.csv'
    original=file.read_bytes()
    file.write_bytes(original+b'\n')
    try:
        training.load_prepared(prepared)
    except ValueError as exc:
        assert 'hash mismatch' in str(exc)
    else:
        raise AssertionError('Tampered data was accepted')
    file.write_bytes(original)
    cmd=[sys.executable,str(Path(training.__file__).resolve()),'--prepared',str(prepared),
         '--out',str(root/'run'),'--models','tfidf','--c-values','1.0','--description-ablation']
    subprocess.run(cmd+['--validate-only'],check=True)
    assert not (root/'run').exists()
    print('Input integrity and no-write validation checks passed.',flush=True)
    if args.guards_only:
        print('Guard-only test completed:',root)
        return
    subprocess.run(cmd,check=True)
    run=root/'run'
    metrics=training.read_json(run/'metrics.json')
    assert metrics['winner']=='tfidf' and metrics['test_rows']>0
    with (run/'split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        split=list(csv.DictReader(f))
    dev={r['group_id'] for r in split if r['partition']=='development'}
    test={r['group_id'] for r in split if r['partition']=='test'}
    assert not dev & test
    before=training.sha(run/'winner.joblib')
    rerun=subprocess.run(cmd,check=True,capture_output=True,text=True)
    assert 'no retraining' in rerun.stdout and training.sha(run/'winner.joblib')==before
    import joblib
    model=joblib.load(run/'winner.joblib')
    prediction=model['estimator'].predict(['Purposeword0 performs specialized tasks.'])
    assert prediction[0] in LABELS
    assert model['training_partition']=='development_only'
    assert training.sha(prepared/'training_rows.csv')==manifest['outputs']['training_rows.csv']
    print('TF-IDF smoke test, group isolation, saved inference and run reuse passed:',root)


if __name__=='__main__':
    main()
