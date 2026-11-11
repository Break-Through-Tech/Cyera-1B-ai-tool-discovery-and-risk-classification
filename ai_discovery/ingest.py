"""Extract distinct observed applications from an enterprise activity CSV.

Python 3.10+, standard library only. Prints to the terminal; writes no output files.
No network, API keys, or AI classification.
Run from your project root:
 python ai_discovery/ingest.py --input data/raw/novatech_usage.csv

Pipeline use:
 from ai_discovery.ingest import extract_applications
 result = extract_applications('data/raw/novatech_usage.csv')
 applications = result['applications']  # pass directly to the next stage

Identity is normalized application name + host + port + path + query + fragment.
Names are case-insensitive; subdomains and case-sensitive paths are preserved.
HTTP/HTTPS and a leading www. are normalized. Root '/' equals an empty path;
other trailing slashes, queries and fragments are retained conservatively.
Aliases are NOT merged. Catalog matching belongs to the next pipeline stage.
Every accepted input record counts as an event (no event-ID deduplication).
"""
import argparse
import csv
import hashlib
import ipaddress
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

METRICS = ['session_duration_seconds', 'bytes_uploaded', 'bytes_downloaded']
APP_FIELDS = ['application_id', 'application_name', 'normalized_name', 'domain',
              'url_path', 'url_query', 'url_fragment', 'observed_endpoint',
              'event_count', 'unique_employee_count', 'departments', 'name_variants',
              'raw_domain_variants', 'events_missing_employee_id', 'events_missing_department']
for _metric in METRICS:
    APP_FIELDS += ['total_' + _metric, _metric + '_observed_events', _metric + '_invalid_events']


def clean_name(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def normalize_endpoint(value):
    """Return host/path/query/fragment and a scheme-free endpoint; raise on invalid input."""
    value = value.strip()
    if not value or re.search(r'\s', value):
        raise ValueError('missing_domain' if not value else 'whitespace_in_domain')
    try:
        parsed = urlsplit('https:' + value if value.startswith('//') else
                          value if '://' in value else 'https://' + value)
        if parsed.scheme.lower() not in ('http', 'https') or parsed.username is not None or parsed.password is not None:
            raise ValueError('unsupported_url_or_credentials')
        host = (parsed.hostname or '').lower().rstrip('.').removeprefix('www.')
        host = host.encode('idna').decode('ascii')
        try:
            ipaddress.ip_address(host)
            host_valid = True
        except ValueError:
            host_valid = bool(re.fullmatch(r'(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', host))
        if not host_valid:
            raise ValueError('invalid_hostname')
        port = parsed.port
        authority = '[' + host + ']' if ':' in host else host
        if port is not None and not ((parsed.scheme.lower() == 'http' and port == 80) or
                                     (parsed.scheme.lower() == 'https' and port == 443)):
            authority += ':' + str(port)
        path = '' if parsed.path == '/' else parsed.path
        endpoint = authority + path
        if parsed.query: endpoint += '?' + parsed.query
        if parsed.fragment: endpoint += '#' + parsed.fragment
        return host, path, parsed.query, parsed.fragment, endpoint
    except (ValueError, UnicodeError) as exc:
        raise ValueError('invalid_endpoint: ' + str(exc)) from exc


def extract_applications(input_path):
    """Read logs and return applications, event_map, rejected_rows, and summary.

    Required columns: application_name, domain. Employee, department, event_id,
    and usage metrics are optional. Missing/invalid metrics do not reject an event.
    Blank totals mean no valid observations, not a measured zero.
    """
    input_path = Path(input_path)
    applications = {}; mapping = []; rejected = []; input_count = 0
    with input_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError('Duplicate CSV column names')
        missing = {'application_name', 'domain'} - set(fields)
        if missing: raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
        for source_row, row in enumerate(reader, 1):
            input_count += 1
            reason = ''
            if None in row or any(v is None for v in row.values()):
                reason = 'malformed_csv_record'
            name = clean_name(row.get('application_name') or '')
            if not reason and not name: reason = 'missing_application_name'
            if not reason:
                try:
                    host, path, query, fragment, endpoint = normalize_endpoint(row['domain'])
                except ValueError as exc:
                    reason = str(exc)
            if reason:
                rejected.append({'source_row': source_row, 'event_id': row.get('event_id') or '',
                                 'application_name': row.get('application_name') or '',
                                 'domain': row.get('domain') or '', 'reason': reason})
                continue
            key = (name.casefold(), endpoint)
            app_id = 'app_' + hashlib.sha256(json.dumps(key, ensure_ascii=False).encode('utf-8')).hexdigest()
            if key not in applications:
                applications[key] = dict(application_id=app_id, application_name=name,
                    normalized_name=name.casefold(), domain=host, url_path=path, url_query=query,
                    url_fragment=fragment, observed_endpoint=endpoint, event_count=0,
                    employees=set(), departments=set(), name_variants=set(), raw_domain_variants=set(),
                    events_missing_employee_id=0, events_missing_department=0,
                    totals={m:0 for m in METRICS}, observed={m:0 for m in METRICS}, invalid={m:0 for m in METRICS})
            app = applications[key]; app['event_count'] += 1
            app['name_variants'].add(name); app['raw_domain_variants'].add(row['domain'])
            employee = row.get('employee_id', '').strip()
            department = row.get('department', '').strip()
            if employee: app['employees'].add(employee)
            else: app['events_missing_employee_id'] += 1
            if department: app['departments'].add(department)
            else: app['events_missing_department'] += 1
            for metric in METRICS:
                value = row.get(metric, '').strip()
                if not value: continue
                if not re.fullmatch(r'\d+', value):
                    app['invalid'][metric] += 1
                else:
                    app['totals'][metric] += int(value); app['observed'][metric] += 1
            mapping.append({'source_row':source_row, 'event_id':row.get('event_id', ''), 'application_id':app_id})
    output = []
    for app in applications.values():
        result = {k:app[k] for k in APP_FIELDS if k in app}
        result['unique_employee_count'] = len(app['employees'])
        for key in ['departments','name_variants','raw_domain_variants']:
            result[key] = json.dumps(sorted(app[key]), ensure_ascii=False)
        for metric in METRICS:
            result['total_' + metric] = app['totals'][metric] if app['observed'][metric] else ''
            result[metric + '_observed_events'] = app['observed'][metric]
            result[metric + '_invalid_events'] = app['invalid'][metric]
        output.append(result)
    output.sort(key=lambda r:(r['normalized_name'],r['observed_endpoint']))
    summary = dict(input_file=str(input_path.resolve()), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        input_rows=input_count, accepted_events=len(mapping), rejected_events=len(rejected),
        unique_applications=len(output), unique_hostnames=len({r['domain'] for r in output}),
        rejection_reasons=dict(Counter(r['reason'] for r in rejected)),
        invalid_metrics={m:sum(r[m+'_invalid_events'] for r in output) for m in METRICS},
        identity_policy='Case-folded normalized name plus normalized endpoint; product paths/subdomains preserved; aliases not merged.',
        metric_policy='Nonnegative integer metrics only; blanks are missing; invalid values excluded from totals and counted.',
        source_row_policy='1-based data-record index, excluding header; join to original records using this index.',
        event_policy='All accepted records count; no duplicate event-ID removal.',
        scope='Extraction only; no AI labels or categories assigned.')
    assert len(mapping)+len(rejected)==input_count
    assert sum(r['event_count'] for r in output)==len(mapping)
    return {'applications':output, 'event_map':mapping, 'rejected_rows':rejected, 'summary':summary}


def print_results(result):
    """Print concise results while keeping full records available in memory."""
    summary = result['summary']
    print(f"Input events: {summary['input_rows']:,}")
    print(f"Accepted: {summary['accepted_events']:,}; rejected: {summary['rejected_events']:,}")
    print(f"Distinct applications: {summary['unique_applications']}; hostnames: {summary['unique_hostnames']}")
    print('\nApplications:')
    for index, app in enumerate(result['applications'], 1):
        print(f"{index}. {app['application_name']} ({app['observed_endpoint']})")
        print(f"   Events: {app['event_count']:,}; employees: {app['unique_employee_count']}; departments: {app['departments']}")
    if summary['rejected_events']:
        print('\nRejection reasons: ' + json.dumps(summary['rejection_reasons']))
        for row in result['rejected_rows'][:10]:
            print(f"   Record {row['source_row']}: {row['reason']}")
        if len(result['rejected_rows']) > 10:
            print('   Showing first 10; all rejected records remain in the returned result.')
    if any(summary['invalid_metrics'].values()):
        print('\nInvalid metric counts: ' + json.dumps(summary['invalid_metrics']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=extract_applications(args.input)
    except (OSError,ValueError,csv.Error) as exc:
        parser.exit(1,f'Error: {exc}\n')
    print_results(result)


if __name__=='__main__':main()
