"""Scientific correctness over completions and whole-trace process returns."""
from __future__ import annotations

import math


def _standardize(values, floor):
    mean=sum(values)/len(values)
    std=math.sqrt(sum((value-mean)**2 for value in values)/len(values))
    return [(value-mean)/max(std,floor) for value in values]


def build_advantages(scores, metadata, *, group_size=4, width=None,
                     process_weight=0.25, std_floor=0.10, process_z_clip=2.0):
    """Missing is None/NaN; a genuine zero remains a valid negative comparison.

    Columns are scientific accuracy, model, application and verification.
    Accuracy normalizes only over members with an available accuracy score
    (at least two per question). Each process dimension independently uses its
    available members. All three process dimensions credit the same original-token
    thinking span, excluding native delimiters and the final answer. These are
    whole-trace judgments, not per-step labels. An unavailable outcome never
    erases valid process returns.
    """
    if group_size<2 or len(scores)!=len(metadata) or len(scores)%group_size:
        raise ValueError('Require aligned rows in complete groups with G >= 2')
    for name,value in [('std_floor',std_floor),('process_z_clip',process_z_clip)]:
        if not math.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive and finite')
    if not math.isfinite(process_weight) or process_weight<0:
        raise ValueError('process_weight must be nonnegative and finite')
    lengths=[int(row['length']) for row in metadata]
    if any(value<1 for value in lengths):raise ValueError('Completions must contain at least one token')
    width=max(lengths,default=0) if width is None else width
    if any(value>width for value in lengths):raise ValueError('Completion exceeds advantage width')
    values=[]
    for row in scores:
        if len(row)!=4:raise ValueError('Expected exactly four score columns')
        cleaned=[]
        for value in row:
            if value is None or isinstance(value,float) and math.isnan(value):cleaned.append(None)
            elif isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Available rewards must be finite normalized scores')
            else:cleaned.append(float(value))
        values.append(cleaned)
    # Validate score/span coupling before normalization. Even a singleton cohort
    # or a tied group must not hide malformed metadata that could later assign
    # process credit to final answers or native channel delimiters.
    trace_spans=[]
    for i,row in enumerate(values):
        if not any(value is not None for value in row[1:]):
            trace_spans.append(None)
            continue
        span=metadata[i].get('reasoning_token_span')
        if (not isinstance(span,(list,tuple)) or len(span)!=2
                or any(type(value) is not int for value in span)
                or not 0<=span[0]<span[1]<lengths[i]):
            raise ValueError('Available process reward requires a valid explicit reasoning token span')
        visible_start=metadata[i].get('visible_start_token')
        if visible_start is not None and (type(visible_start) is not int or visible_start!=span[1]+1):
            raise ValueError('Reasoning token span must stop before the native closing boundary')
        spans=metadata[i].get('process_token_spans')
        if spans is not None:
            if not isinstance(spans,list) or len(spans)!=3:
                raise ValueError('Expected three whole-trace process token spans')
            for stage,value in enumerate(row[1:]):
                if value is not None and spans[stage]!=[list(span)]:
                    raise ValueError('Each available process score must credit the same whole reasoning span')
        trace_spans.append(tuple(span))
    result=[[0.0]*width for _ in scores]
    active=[False]*len(scores);outcome=[None]*len(scores);outcome_adv=[0.0]*len(scores)
    process_z=[[0.0]*3 for _ in scores];skipped=0;cohorts=[]
    for first in range(0,len(scores),group_size):
        indices=list(range(first,first+group_size))
        question_ids=[metadata[i].get('example_id') for i in indices]
        if not question_ids[0] or any(value!=question_ids[0] for value in question_ids):
            raise ValueError('Sampler group crosses question IDs; refuse incorrect relative normalization')
        members=[i for i in indices if values[i][0] is not None]
        if len(members)>=2:
            totals=[values[i][0] for i in members]
            for i,value,advantage in zip(members,totals,_standardize(totals,std_floor),strict=True):
                outcome[i]=value;outcome_adv[i]=advantage;active[i]=True
                result[i][:lengths[i]]=[advantage]*lengths[i]
            cohorts.append({'first_row':first,'columns':[0],'members':members})
        else:cohorts.append({'first_row':first,'columns':[],'members':[]})
        for stage in range(3):
            members=[i for i in indices if values[i][stage+1] is not None]
            if len(members)<2:continue
            standardized=_standardize([values[i][stage+1] for i in members],std_floor)
            for i,z in zip(members,standardized,strict=True):
                start,stop=trace_spans[i]
                active[i]=True;z=max(-process_z_clip,min(process_z_clip,z));process_z[i][stage]=z
                for token in range(start,stop):
                    result[i][token]+=process_weight/3*z
        if not any(active[i] for i in indices):skipped+=1
    return {'advantages':result,'active_rows':active,'outcome_rewards':outcome,
            'outcome_advantages':outcome_adv,'process_z':process_z,'skipped_groups':skipped,
            'reward_availability':[[value is not None for value in row] for row in values],
            'outcome_cohorts':cohorts}
