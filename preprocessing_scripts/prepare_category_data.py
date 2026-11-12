"""Prepare Kaggle-14 category data; Python 3.10+, standard library only.

No training, API calls, AI reclassification, or source-file modification.
Import build_model_text() and extract_report() in future inference code.
See README.md for the input contract, commands, and output descriptions.
"""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

VERSION = 'category-preparation-v1'
TEXT_VERSION = 'purpose-description-functionality-v1'
TAXONOMY = [
    'Ai And Machine Learning', 'Business And Finance',
    'Communication And Support', 'Creative And Entertainment',
    'Data And Analytics', 'Education And Learning', 'Industry Specific',
    'Marketing And Sales', 'Media And Content Creation',
    'Operations And Management', 'Personal And Lifestyle',
    'Security And Privacy', 'Specialized Technologies',
    'Technology And Development',
]
SHARED = {'office.com', 'microsoft.com', 'google.com', 'atlassian.com',
          'github.com', 'gitlab.com', 'huggingface.co', 'apps.apple.com',
          'play.google.com', 'chromewebstore.google.com', 'chrome.google.com',
          'gumroad.com', 'discord.com', 'discord.gg', 'youtube.com', 'youtu.be',
          'notion.site', 'notion.so', 'bit.ly', 't.co', 'linktr.ee', 'awin1.com'}
PLACEHOLDERS = {'', 'n/a', 'na', 'null', 'none', 'unknown', 'nan', '-', 'not available'}


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value or '')).split())


def key(value):
    return normalize(value).casefold()


def clean_text(value):
    text = unicodedata.normalize('NFKC', str(value or ''))
    text = re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[\d+(?:\s*[,–-]\s*\d+)*\]', '', text)
    text = re.sub(r'【[^】]*】|[^]*', '', text)
    text = normalize(text.replace('**', '').replace('__', '').replace('`', ''))
    return '' if text.casefold() in PLACEHOLDERS else text


def build_model_text(description, functionality='', mode='combined'):
    """Same text contract for training and inference; no names/labels added.

    Natural product names inside prose remain. Never pass the entire source row.
    Combined uses plain text so source-specific headings are not learned.
    """
    if mode not in {'combined', 'description_only'}:
        raise ValueError('Unsupported text mode')
    description, functionality = clean_text(description), clean_text(functionality)
    parts = [description] if description else []
    if mode == 'combined' and functionality and key(functionality) != key(description):
        parts.append(functionality)
    return '\n'.join(parts)


def endpoint(value):
    try:
        value = normalize(value)
        if not value or re.search(r'\s', value):
            return '', ''
        p = urlsplit(value if '://' in value else 'https://' + value)
        if p.scheme not in {'http', 'https'} or p.username or p.password:
            return '', ''
        host = (p.hostname or '').lower().rstrip('.').removeprefix('www.')
        host = host.encode('idna').decode('ascii')
        if '.' not in host:
            return '', ''
        port = p.port
        authority = host + (':' + str(port) if port and not
                            ((p.scheme == 'https' and port == 443) or
                             (p.scheme == 'http' and port == 80)) else '')
        return authority, p.path.rstrip('/')
    except (ValueError, UnicodeError):
        return '', ''


def shared_host(host):
    hostname = host.split(':')[0]
    return any(hostname == h or hostname.endswith('.' + h) for h in SHARED)


def read_csv(path, required):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required) <= set(fields):
            raise ValueError(f'{path}: duplicate or missing columns; need {required}')
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'{path}: malformed CSV rows')
    return rows


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def extract_report(report):
    """Extract explicit purpose sections only; return text and original offsets.

    Repeated sections are ambiguous and withheld, not concatenated blindly.
    Missing sections never fall back to full-report text.
    """
    aliases = {'product description': 'description', 'description': 'description',
               'ai functionality': 'functionality', 'functionality': 'functionality',
               'product identity': 'other', 'alternate names/domains': 'other',
               'alternate names and domains': 'other', 'uncertainty': 'other'}
    sections = defaultdict(list)
    for bi, block in enumerate(report.get('report_blocks', [])):
        text = block.get('text', '')
        if not isinstance(text, str):
            raise ValueError('Report block text must be a string')
        headings = []
        for match in re.finditer(r'^.*$', text, re.MULTILINE):
            line = match.group().strip()
            title = re.sub(r'^#{1,6}\s*', '', line)
            title = title.replace('**', '').replace('__', '')
            title = re.sub(r'^\d+[.)]\s*', '', title).strip().rstrip(':').casefold()
            if title in aliases:
                headings.append((match.start(), match.end(), aliases[title]))
            elif re.match(r'^#{1,3}\s', line) or re.match(r'^\d+[.)]\s', line):
                headings.append((match.start(), match.end(), 'other'))
        for i, (_, start, kind) in enumerate(headings):
            if kind == 'other':
                continue
            stop = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            citations = []
            for c in block.get('citations', []) or []:
                a, b = c.get('start_index'), c.get('end_index')
                if isinstance(a, int) and isinstance(b, int) and a < stop and b > start:
                    citations.append(c)
            sections[kind].append({'text': clean_text(text[start:stop]), 'block_index': bi,
                                   'start': start, 'end': stop, 'citations': citations})
    ambiguous = any(len(sections[k]) > 1 for k in ('description', 'functionality'))
    description = sections['description'][0]['text'] if len(sections['description']) == 1 else ''
    functionality = sections['functionality'][0]['text'] if len(sections['functionality']) == 1 else ''
    combined = '' if ambiguous else build_model_text(description, functionality)
    return dict(description=description, functionality=functionality, model_text=combined,
                description_only_text='' if ambiguous else build_model_text(description, mode='description_only'),
                extraction_status='ambiguous_sections' if ambiguous else 'extracted' if combined else 'missing_sections',
                sections=dict(sections), text_contract_version=TEXT_VERSION)


class Groups:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def prepare(args):
    if args.out.exists():
        raise ValueError('Output directory already exists. Choose a new --out; nothing will be overwritten.')
    raw = read_csv(args.raw, ['Name', 'Website', 'Category', 'Short Description', 'Primary Task'])
    categories = {r['Category'] for r in raw}
    if categories != set(TAXONOMY):
        raise ValueError(f'Expected exact Kaggle-14 labels. Missing: {set(TAXONOMY)-categories}; unexpected: {categories-set(TAXONOMY)}')
    catalog = read_csv(args.catalog, ['tool_id', 'tool_name', 'domain', 'category', 'is_ai_tool']) if args.catalog else []
    reference = read_csv(args.reference, ['tool_id', 'tool_name', 'domain', 'category', 'is_ai_tool']) if args.reference else []
    records = []
    for i, row in enumerate(raw, 1):
        description, functionality = clean_text(row['Short Description']), clean_text(row['Primary Task'])
        # A metadata value consisting solely of a taxonomy label is not purpose evidence.
        label_echo = key(functionality) in {key(c) for c in TAXONOMY}
        if label_echo:
            functionality = ''
        combined = build_model_text(description, functionality)
        host, path = endpoint(row['Website'])
        records.append(dict(source_row=i, source_id=f'raw_{i:05d}', tool_name=normalize(row['Name']),
            name_key=key(row['Name']), authority=host, product_path=path,
            category=row['Category'], description=description, functionality=functionality,
            model_text=combined, description_only_text=build_model_text(description, mode='description_only'),
            text_sha256=digest(combined), text_contract_version=TEXT_VERSION,
            text_origin='description_plus_primary_task' if description and functionality else
                        'description_only' if description else 'primary_task_only' if functionality else 'empty',
            word_count=len(re.findall(r'\b\w+\b', combined)),
            quality_flags=';'.join(x for x, enabled in [
                ('short_description', bool(description) and len(description) < 40),
                ('very_short_purpose_text', len(re.findall(r'\b\w+\b', combined)) < 3),
                ('missing_description', not description), ('invalid_endpoint', not host),
                ('task_is_taxonomy_label_removed', label_echo)] if enabled)))
    groups = Groups(len(records))
    indexes = {kind: defaultdict(list) for kind in ['name', 'description', 'text', 'endpoint', 'identity']}
    for i, r in enumerate(records):
        signals = {'name': r['name_key'], 'description': key(r['description']),
                   'text': key(r['model_text']),
                   'identity': (r['name_key'], r['authority'], r['product_path']) if r['authority'] and r['name_key'] else ''}
        # Conservative split grouping, not proof of product identity or a lookup rule.
        signals['endpoint'] = (r['authority'], r['product_path']) if r['authority'] and (r['product_path'] or not shared_host(r['authority'])) else ''
        for kind, signal in signals.items():
            if signal:
                indexes[kind][signal].append(i)
    edges, conflicts, conflict_rows = [], [], set()
    for kind, mapping in indexes.items():
        for members in mapping.values():
            if len(members) < 2:
                continue
            for j in members[1:]:
                groups.union(members[0], j)
                edges.append(dict(left=records[members[0]]['source_id'], right=records[j]['source_id'], reason=kind))
            labels = sorted({records[j]['category'] for j in members})
            if len(labels) > 1:
                exclude = kind in {'identity', 'text'}
                if exclude:
                    conflict_rows.update(members)
                conflicts.append(dict(signal=kind, source_ids=';'.join(records[j]['source_id'] for j in members),
                    categories=';'.join(labels), action='exclude_conflicting_training_rows' if exclude else 'group_only_review_identity'))
    memberships = defaultdict(list)
    for i in range(len(records)):
        memberships[groups.find(i)].append(i)
    for members in memberships.values():
        gid = 'group_' + digest('|'.join(records[i]['source_id'] for i in members))[:16]
        for i in members:
            records[i]['group_id'] = gid

    overlaps, known_non_ai, catalog_checks = [], set(), []
    for source, rows in [('catalog', catalog), ('reference', reference)]:
        seen_ids = set()
        for row in rows:
            if row['tool_id'] in seen_ids:
                raise ValueError(f'Duplicate {source} tool_id: {row["tool_id"]}')
            seen_ids.add(row['tool_id'])
            label = row['is_ai_tool'].strip().lower()
            if label not in {'true', 'false', '1', '0'}:
                raise ValueError(f'Unresolved {source} AI label: {row["tool_id"]}')
            n, (h, p) = key(row['tool_name']), endpoint(row['domain'])
            matches = indexes['name'].get(n, [])
            for j in matches:
                r = records[j]
                same_host = bool(h) and h == r['authority']
                exact = same_host and p == r['product_path']
                overlaps.append(dict(source=source, tool_id=row['tool_id'], source_id=r['source_id'],
                    group_id=r['group_id'], match='name_and_endpoint' if exact else 'name_and_host' if same_host else 'name_only',
                    is_ai_tool=label, reference_category=row['category'], raw_category=r['category']))
                if exact and label in {'false', '0'}:
                    known_non_ai.add(j)
            if source == 'catalog':
                proposed = re.fullmatch(r'tool_(\d+)', row['tool_id'])
                j = int(proposed.group(1))-1 if proposed else -1
                valid = 0 <= j < len(records)
                r = records[j] if valid else {}
                confirmed = valid and r['name_key'] == n and bool(h) and h == r['authority']
                catalog_checks.append(dict(tool_id=row['tool_id'], source_id=r.get('source_id', ''),
                    status='verified_name_host' if confirmed else 'unverified_do_not_join',
                    category_agrees=confirmed and row['category'] == r['category'],
                    raw_product_path=r.get('product_path', '')))

    training, excluded = [], []
    for i, r in enumerate(records):
        reasons = []
        if not r['tool_name']:
            reasons.append('missing_name')
        if r['word_count'] < args.min_words:
            reasons.append('insufficient_purpose_text')
        if i in conflict_rows:
            reasons.append('conflicting_identity_or_identical_text_labels')
        if i in known_non_ai:
            reasons.append('exact_reference_or_catalog_non_ai')
        r['eligible'] = not reasons
        r['exclusion_reason'] = ';'.join(reasons)
        (excluded if reasons else training).append(r)

    preview = []
    if args.discovery:
        data = json.loads(args.discovery.read_text(encoding='utf-8-sig'))
        if not isinstance(data.get('results'), list):
            raise ValueError('--discovery expects classify.py report with results array')
        for result in data['results']:
            reports = result.get('reports', [])
            # One candidate per report; later inference must resolve multiple evidence versions.
            for ri, report in enumerate(reports or [{}]):
                extracted = extract_report(report)
                is_ai = result.get('is_ai_tool')
                detection_ok = result.get('classification_status') == 'classified'
                ready = detection_ok and is_ai is True and result.get('identity_supported') is True
                status = ('not_applicable' if detection_ok and is_ai is False else
                          'pending_ai_detection' if not ready else
                          'needs_evidence_review' if len(reports) != 1 or extracted['extraction_status'] != 'extracted' or
                          len(re.findall(r'\b\w+\b', extracted['model_text'])) < args.min_words else 'ready_for_category_model')
                matches = indexes['name'].get(key(result.get('application_name')), [])
                h, p = endpoint(result.get('endpoint'))
                preview.append(dict(application_name=result.get('application_name'), endpoint=result.get('endpoint'),
                    report_index=ri, is_ai_tool=is_ai, ai_classification_status=result.get('classification_status'),
                    preparation_status=status, category=None, category_source=None,
                    candidate_model_text=extracted['model_text'],
                    model_input=extracted['model_text'] if status == 'ready_for_category_model' else None,
                    description_only_text=extracted['description_only_text'],
                    extraction_status=extracted['extraction_status'], sections=extracted['sections'],
                    text_contract_version=TEXT_VERSION,
                    potential_training_overlap=[dict(source_id=records[j]['source_id'], group_id=records[j]['group_id'],
                        same_endpoint=(h, p)==(records[j]['authority'], records[j]['product_path'])) for j in matches],
                    note='Preview only: catalog identity resolution/category reuse must precede model inference. No discovery labels changed.'))

    counts = Counter(r['category'] for r in training)
    group_counts = {c: len({r['group_id'] for r in training if r['category'] == c}) for c in TAXONOMY}
    audit = dict(raw_rows=len(records), eligible_rows=len(training), excluded_rows=len(excluded),
        raw_category_counts=dict(Counter(r['category'] for r in records)), eligible_category_counts=dict(counts),
        eligible_groups_by_category=group_counts, groups=len(memberships),
        blank_descriptions=sum(not r['description'] for r in records),
        short_nonempty_descriptions=sum(bool(r['description']) and len(r['description']) < 40 for r in records),
        eligible_text_origins=dict(Counter(r['text_origin'] for r in training)),
        duplicate_group_counts={k: sum(len(v)>1 for v in idx.values()) for k, idx in indexes.items()},
        conflicting_group_counts=dict(Counter(r['signal'] for r in conflicts)),
        exact_non_ai_exclusions=len(known_non_ai),
        catalog_join_status=dict(Counter(r['status'] for r in catalog_checks)),
        category_reference_rows=len(reference), discovery_previews=len(preview),
        discovery_status_counts=dict(Counter(r['preparation_status'] for r in preview)),
        warnings=['Kaggle purpose text is a proxy for web-enriched descriptions/functionality, not an identical distribution.',
                  'Group IDs prevent documented overlaps crossing splits; use group-aware splitting, never row-random splitting.',
                  'Name-only groups are conservative leakage controls, not verified aliases. Different labels in those groups are not automatically dropped.',
                  'Near-duplicate paraphrases and aliases with different names/endpoints are not automatically detected in v1. Review before final evaluation.',
                  'Task-only rows are retained when eligible; report their results separately from description-bearing products.',
                  'Reference/discovery rows are audits only, never appended as category training examples.',
                  'No split, training, model metrics, API fallback, or production integration is performed.'])
    if not training or set(counts) != set(TAXONOMY):
        raise ValueError('Preparation leaves an empty category. Review data/filters before training.')
    inputs = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in [('raw', args.raw), ('catalog', args.catalog), ('reference', args.reference), ('discovery', args.discovery)] if path}
    fields = ['source_row', 'source_id', 'group_id', 'tool_name', 'authority', 'product_path',
              'category', 'model_text', 'description_only_text', 'description', 'functionality',
              'text_origin', 'text_sha256', 'text_contract_version', 'word_count', 'quality_flags',
              'eligible', 'exclusion_reason']
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out/'training_rows.csv', training, fields)
    write_csv(args.out/'excluded_rows.csv', excluded, fields)
    write_csv(args.out/'group_links.csv', edges, ['left', 'right', 'reason'])
    write_csv(args.out/'label_conflicts.csv', conflicts, ['signal', 'source_ids', 'categories', 'action'])
    write_csv(args.out/'reference_overlap.csv', overlaps, ['source', 'tool_id', 'source_id', 'group_id', 'match', 'is_ai_tool', 'reference_category', 'raw_category'])
    write_csv(args.out/'catalog_join_audit.csv', catalog_checks, ['tool_id', 'source_id', 'status', 'category_agrees', 'raw_product_path'])
    write_json(args.out/'discovery_input_preview.json', preview)
    write_json(args.out/'taxonomy.json', {'version': 'original_kaggle_14', 'categories': TAXONOMY, 'sha256': digest(json.dumps(TAXONOMY))})
    write_json(args.out/'audit.json', audit)
    write_json(args.out/'manifest.json', dict(version=VERSION, created_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        text_contract_version=TEXT_VERSION, min_words=args.min_words,
        feature_column='model_text', comparison_feature_column='description_only_text',
        target_column='category', split_group_column='group_id',
        source_fields=['Short Description', 'Primary Task'],
        inference_fields=['Product Description', 'AI Functionality'],
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir())}))
    print(json.dumps({k: audit[k] for k in ['raw_rows', 'eligible_rows', 'excluded_rows', 'eligible_text_origins', 'discovery_status_counts']}, indent=2))
    print('Prepared:', args.out.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True, help='Original Kaggle CSV, with Short Description and Primary Task')
    parser.add_argument('--catalog', type=Path, help='Working Dataset B catalog, for join/overlap audit only')
    parser.add_argument('--reference', type=Path, help='697-row reference CSV; audit plus exact-identity non-AI exclusions')
    parser.add_argument('--discovery', type=Path, help='classify.py JSON report; optional inference text preview, never training labels')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; existing directories are rejected')
    parser.add_argument('--min-words', type=int, default=1, help='Minimum combined purpose-text word count; default 1 (short text is flagged)')
    args = parser.parse_args()
    if args.min_words < 1:
        parser.error('--min-words must be positive')
    try:
        prepare(args)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
