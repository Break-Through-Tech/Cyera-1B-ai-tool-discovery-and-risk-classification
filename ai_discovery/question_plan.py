"""Conservative question limits and explicit evidence-control predicates."""
import re


def limits(question,company,companies):
    text=question.casefold()
    other=[c for c in companies if c.casefold()!=company.casefold() and re.search(r'(?<!\w)'+re.escape(c.casefold())+r'(?!\w)',text)]
    if other:return 'This question names another company. Ask separately with that explicit company scope; cross-company comparisons are not supported.'
    if re.search(r'\bnot approved\b',text):
        return 'Not approved is ambiguous: unapproved, under review and unknown are different states. Use an explicit approval filter.'
    if re.search(r'\bhow many\s+(?:(?:finance|engineering|hr|legal|marketing|sales|operations|unique|distinct|individual)\s+)*(?:users?|employees?|people)\b|\b(?:count|number of)\s+(?:users?|employees?|people)\b',text):
        return 'Distinct user counts across applications and department-specific user counts cannot be calculated from these application summaries. They require scoped employee-level activity; whole-application recorded counts are not interchangeable.'
    if re.search(r'\b(?:trend|trends|growing|growth|fastest.growing|last month|last week|over time|yesterday|increase|increased)\b',text):
        return 'This index contains one report snapshot per company, not a time series. It cannot establish trends or changes between dates.'
    if re.search(r'\b(?:safe|safest|breach|breached|leak|leaked|probability|probabilities|guarantee|guaranteed|compliant|compliance)\b',text):
        return 'A prototype governance priority does not establish safety, compliance, a breach, data leakage or a probability. Those conclusions require additional evidence and validation.'
    return None


def control_filters(question):
    text=question.casefold().replace("don't",'do not').replace("doesn't",'does not');result={}
    if re.search(r'\b(?:without|lack|lacking|no) sso\b|\bdo(?:es)? not (?:offer|have|support) sso\b|\bsso (?:unavailable|disabled)\b',text):result['sso_available']=False
    elif re.search(r'\b(?:with|support|supports|offer|offers|have|has) sso\b|\bsso (?:available|enabled)\b',text):result['sso_available']=True
    if re.search(r'\bunknown sso\b|\bsso (?:unknown|not documented)\b',text):result['sso_available']=None
    if re.search(r'\b(?:without|lack|no) admin(?:istrative)? controls\b',text):result['admin_controls_available']=False
    if re.search(r'\b(?:use|uses|train|trains|training on) (?:our |customer |submitted )?data (?:for (?:model )?training)?\b',text) or 'train on customer data' in text:
        result['training_on_customer_data']=False if re.search(r'\b(?:not|never|no)\b',text) else True
    return result


def requested_application(question):
    """Recognize narrow named-application forms so absent names do not get neighbors."""
    patterns=[r'^why (?:was|is) (.+?) (?:flagged|high[- ]risk|under review)\??$',
              r'^(?:tell me about|what about) (.+?)\??$',
              r'^explain (?:the )?(?:risk|governance)(?: assessment| concerns)? (?:for|of) (.+?)\??$']
    for pattern in patterns:
        match=re.match(pattern,question.strip(),re.IGNORECASE)
        if match:return match.group(1).strip().rstrip('?')
    return None
