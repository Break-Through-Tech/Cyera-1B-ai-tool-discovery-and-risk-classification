"""Clean a Kaggle AI catalog into exactly the 14 Dataset B columns.

All AI labels are assumed true at the user's direction, not independently verified.
Run: python prepare_kaggle_dataset_b.py --input kaggle.csv --out-dir prepared
Standard library only. Inputs are never modified. Existing outputs are protected.
"""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

FIELDS = ['tool_id','tool_name','domain','description','is_ai_tool','category','vendor',
          'approved_status','enterprise_plan_available','sso_available','data_retention',
          'training_on_customer_data','admin_controls_available','risk_label']
MISSING = {'', 'nan', 'none', 'null', 'n/a', '-'}
SHARED = {'github.com','huggingface.co','apps.apple.com','play.google.com','chrome.google.com',
          'chromewebstore.google.com','gumroad.com','discord.com','discord.gg','youtube.com',
          'youtu.be','awin1.com','bit.ly','t.co','linktr.ee','notion.so','notion.site'}


def clean(value):
    text = unicodedata.normalize('NFC', value)
    # Repair only reversible encodings with recognizable mojibake markers.
    if any(x in text for x in ('â€','Ã','Â','ï¿½')):
        for encoding in ('cp1252', 'latin1'):
            try:
                fixed = text.encode(encoding).decode('utf-8')
                if sum(text.count(x) for x in ('â€','Ã','Â','ï¿½')) > sum(fixed.count(x) for x in ('â€','Ã','Â','ï¿½')):
                    text = fixed
                    break
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
    text = ' '.join(text.split())
    return '' if text.lower() in MISSING else text


def website(value):
    value = clean(value)
    if not value:
        return '', '', '', ''
    try:
        p = urlsplit(('https:'+value) if value.startswith('//') else value if '://' in value else 'https://'+value)
        if p.scheme not in ('https', 'http') or p.username or p.password:
            return '', '', '', ''
        host = (p.hostname or '').lower().rstrip('.').removeprefix('www.').encode('idna').decode()
        if not re.fullmatch(r'(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9-]{2,}',host):
            return '', '', '', ''
        port = p.port
        query = urlencode([(k,v) for k,v in parse_qsl(p.query,keep_blank_values=True)
                           if not k.lower().startswith('utm_') and k.lower() not in {'ref','fbclid','gclid'}])
        path = p.path.rstrip('/')
        url = urlunsplit((p.scheme,host+(f':{port}' if port and port not in (80,443) else ''),path,query,p.fragment))
        return host,path,query,url
    except (ValueError,UnicodeError):
        return '', '', '', ''


def write(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    names=['kaggle_dataset_b.csv','catalog_context.csv','excluded_rows.csv','summary.json']
    if any((a.out_dir/n).exists() for n in names):p.error('Output exists; choose a new directory.')
    with a.input.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'Name','Website','Short Description','Category','Website Status','Link'}
        if not required <= set(reader.fieldnames or []):p.error('Missing required Kaggle columns.')
        raw=list(reader)
    out=[];context=[];excluded=[];seen={}
    for i,r in enumerate(raw,1):
        if None in r or any(v is None for v in r.values()):raise ValueError(f'Malformed source row {i}')
        tid=f'tool_{i:05d}'
        name,desc,category=[clean(r[k]) for k in ['Name','Short Description','Category']]
        host,path,query,url=website(r['Website'])
        key=(name.casefold(),host,path,query,urlsplit(url).fragment)
        reason='missing_name' if not name else 'missing_or_invalid_domain' if not host else 'duplicate_product' if key in seen else ''
        if reason:
            excluded.append(dict(tool_id=tid,raw_row=i,tool_name=name,raw_website=r['Website'],reason=reason,
                                 duplicate_of=seen.get(key,'') if reason=='duplicate_product' else ''))
            continue
        seen[key]=tid
        row=dict.fromkeys(FIELDS,'')
        row.update(tool_id=tid,tool_name=name,domain=host,description=desc,is_ai_tool='true',category=category)
        out.append(row)
        context.append(dict(tool_id=tid,raw_row=i,raw_tool_name=r['Name'],raw_website=r['Website'],
            website_url_clean=url,url_path=path,source_url=clean(r['Link']),
            website_status_original=clean(r['Website Status']),
            missing_description=str(not bool(desc)).lower(),
            shared_host_requires_product_match=str(host in SHARED).lower(),
            label_provenance='user_assumption_all_kaggle_entries_are_ai',
            category_provenance='original_kaggle_category',independently_verified='false'))
    counts=Counter(x['domain'] for x in out)
    for r,c in zip(out,context):c['catalog_rows_at_host']=counts[r['domain']]
    assert len(out)+len(excluded)==len(raw)
    assert len({r['tool_id'] for r in out})==len(out)
    assert all(list(r)==FIELDS and r['is_ai_tool']=='true' for r in out)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    write(a.out_dir/names[0],out,FIELDS)
    write(a.out_dir/names[1],context,list(context[0]))
    write(a.out_dir/names[2],excluded,['tool_id','raw_row','tool_name','raw_website','reason','duplicate_of'])
    summary=dict(input_rows=len(raw),output_rows=len(out),excluded_rows=len(excluded),
        exclusions=dict(Counter(r['reason'] for r in excluded)),
        retained_missing_descriptions=sum(not r['description'] for r in out),
        retained_historical_status_not_active=sum(c['website_status_original'].lower()!='active' for c in context),
        shared_host_rows=sum(c['shared_host_requires_product_match']=='true' for c in context),
        schema=FIELDS,label_policy='All retained Kaggle tools assumed AI by user instruction; no verification performed.',
        category_policy='Preserved original categories; not mapped to a new taxonomy.',
        unknown_policy='Vendor and all approval/security/retention/training/admin/risk fields left blank.',
        retention_policy='Missing descriptions and historical site failures retained for lookup. Invalid domains quarantined. Exact normalized name+host+path+nontracking query+fragment duplicates removed.',
        input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest())
    (a.out_dir/names[3]).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
