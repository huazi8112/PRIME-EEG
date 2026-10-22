"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from . import baseline_raw as raw, baseline_official as official


def union(intervals):
    out=[]
    for a,b in sorted((float(a),float(b)) for a,b in intervals if b>a):
        if out and a<=out[-1][1]+1e-9: out[-1][1]=max(b,out[-1][1])
        else: out.append([a,b])
    return out


def intersect(a,b):
    a,b=union(a),union(b);out=[];i=j=0
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if hi>lo: out.append([lo,hi])
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return union(out)


def duration(a): return sum(b-a for a,b in union(a))


def in_exposure(t,intervals):
    t=np.asarray(t,dtype=float);m=np.zeros(len(t),bool)
    for a,b in intervals: m|=(t>a+1e-9)&(t<=b+1e-9)
    return m


def clusters(times,extension=None):
    """None = exact v48 point clustering; extension = archived interval merge."""
    out=[]
    for t in sorted(float(x) for x in times):
        end=t if extension is None else t+extension
        if out and t-out[-1]['end']<=90 and end-out[-1]['start']<=300:
            out[-1]['end']=max(end,out[-1]['end']);out[-1]['points'].append(t)
        else:out.append({'start':t,'end':end,'points':[t]})
    return out


def score(subject,records,points,hours,mode,exposures=None,details=False):
    targets={e['event_id']:e for r in records for e in r['events'] if not e.get('consumed_initialization_event')}
    hit={};alarms=[];false=0
    for r in records:
        pts=np.asarray(points[r['record']],dtype=float)
        if mode in ('native','common'):
            pts=pts[in_exposure(pts,exposures[r['record']])]
        extension=r['extension'] if mode in ('legacy','match_only') else None
        refs=sorted([e for e in r['events'] if e['event_id'] in targets],key=lambda e:e['start_seconds'])
        unmatched={e['event_id'] for e in refs}
        for i,c in enumerate(clusters(pts,extension)):
            matched=None;available=None
            for e in refs:
                if e['event_id'] not in unmatched:continue
                a=e['start_seconds']-30;b=e['end_seconds']+60
                valid=[t for t in c['points'] if a<=t<=b]
                overlap=(c['end']>a and c['start']<b) if mode=='legacy' else bool(valid)
                if overlap:
                    matched=e['event_id'];available=c['start'] if mode=='legacy' else min(valid)
                    unmatched.remove(matched)
                    hit[matched]={'record':r['record'],'latency':available-e['start_seconds'],'matched_time':available,'cluster_start':c['start'],'cluster_end':c['end']}
                    break
            if matched is None:false+=1
            if details:alarms.append({'record':r['record'],'cluster':i,'start':c['start'],'end':c['end'],'points':c['points'],'matched_event_id':matched,'matched_time':available})
    tp=len(hit);n=len(targets);prec=tp/(tp+false) if tp+false else 0.
    out={'dataset':'chbmit','subject':subject,'detected':tp,'total':n,'matched_predictions':tp,'false_alarms':false,'monitoring_hours':hours,'sensitivity':tp/n if n else 0.,'precision':prec,'far_per_hour':false/hours if hours else 0.,'latencies':[hit[e]['latency'] for e in sorted(hit)],'detected_event_ids':sorted(hit),'target_event_ids':sorted(targets)}
    if details:out.update(event_details=hit,alarm_details=alarms)
    return out


def case_records(s,family,payload):
    out=[]
    for r in payload['records']:
        out.append({'record':r['record'],'events':r['events'],'extension':15. if family=='raw' else 1.})
    return out


def raw_points(payload,c,thresholds):
    rule=raw.Rule(**c);out={}
    tk=(rule.representation,rule.threshold_mode,rule.threshold_value)
    for row in payload['records']:
        k=(row['record'],tk)
        if k not in thresholds:thresholds[k]=raw.threshold_array(payload,row,rule)
        vals=raw.representation_values(row,rule.representation)
        ends=row['interval_ends'];valid=np.ones(len(vals),bool)
        for e in row['events']:
            if e.get('consumed_initialization_event'):valid&=ends-60>=e['end_seconds']
        mask=raw.persistent(vals>=thresholds[k],rule.persistence)&valid
        out[row['record']]=ends[mask]
    return out


def official_points(payload,c):
    out={};rule=official.Rule(**c);tk=official.threshold_key(rule.threshold)
    for row in payload['records']:
        a=row['score'];starts=a[f'{tk}_starts'];ends=a[f'{tk}_ends']
        starts=starts[(ends-starts)>=rule.minimum_event_seconds]
        ws=np.floor(starts/60)*60
        for e in row['events']:
            if e.get('consumed_initialization_event'):ws=ws[ws>=e['end_seconds']]
        out[row['record']]=ws+60.
    return out
