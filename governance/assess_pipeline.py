"""Apply Component 4 to a saved Components 1-3 report with optional governance research."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .risk import POLICY, assess_records
from . import evidence as governance_evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-report', type=Path, required=True)
    p.add_argument('--activity', type=Path, required=True)
    p.add_argument('--company')
    p.add_argument('--policy', type=Path, default=POLICY)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--activity-provenance', choices=['unknown', 'synthetic', 'verified'], default='unknown')
    p.add_argument('--out', type=Path, required=True)
    governance_evidence.options(p)
    a = p.parse_args()
    try:
        if a.out.exists():
            raise ValueError('Output exists; choose a new file')
        report = json.loads(a.pipeline_report.read_text(encoding='utf-8'))
        policy = json.loads(a.policy.read_text(encoding='utf-8'))
        evidence = json.loads(a.evidence.read_text(encoding='utf-8')) if a.evidence else None
        evidence, details, cache = governance_evidence.prepare(report['results'], a.company, a, evidence)
        report['governance_evidence_report'] = details
        report['governance_research_cache'] = cache
        report['results'] = assess_records(report['results'], report['ingestion'], a.activity,
                                          policy, a.company, evidence, a.activity_provenance)
        report['scope'] = 'All extracted activity applications; Components 1-4.'
        report['governance_counts'] = dict(Counter(r['governance']['assessment_status'] for r in report['results']))
        import hashlib
        report['governance_inputs'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                                      for f in (a.pipeline_report, a.policy, a.evidence, *governance_evidence.input_paths(a)) if f}
        report['governance_engine_sha256'] = hashlib.sha256(Path(__file__).with_name('risk.py').read_bytes()).hexdigest()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(report['governance_counts']))
    except KeyboardInterrupt:
        p.exit(130, 'Interrupted; governance checkpoints can be reused on the next run.\n')
    except (OSError, ValueError, KeyError) as exc:
        p.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
