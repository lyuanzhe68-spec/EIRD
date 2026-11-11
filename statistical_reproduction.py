"""Read-only statistical reproduction. Run with --help for optional pixel replay."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import scipy
from scipy.stats import rankdata, wilcoxon

H1 = 'h1_ablation_cache_20261029/'
H2 = 'h2_analysis_final_20261029/'
V17 = 'eird_v17_1_conditional_complementarity/'
S17 = 'eird_v17_2_statistical_robustness/'
V18 = 'eird_v18_external_validation/'
V82 = 'eird_v18_2_reviewer_resolution/'
V84 = 'eird_v18_4_capacity_controls/'
DPS = 'eird_jei_revision_20261029/dps_baseline_20261029/'
ARMS = ['M0_B16', 'M1_B16_DPS', 'M2_B16_EIRD', 'M3_B16_DPS_EIRD']
PAIR = [(3, 1), (3, 2), (3, 0)]
EXT = {'KITTI2015': ('03_kitti_framewise_results.csv', 200), 'ETH3D': ('05_eth3d_framewise_results.csv', 27)}

def j(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, values):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)

def arr(rs, *cols):
    return np.array([[float(r[c]) for c in cols] for r in rs])

def fmt(v):
    return None if v is None else (f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v))

def signs(v):
    return [int((v > 0).sum()), int((v < 0).sum()), int((v == 0).sum())]

def ci(v):
    return np.quantile(v, [.025, .975], axis=0, method='linear')

def paired_draws(x, groups, draws, seed, sorted_groups=False):
    order = sorted(set(groups)) if sorted_groups else list(dict.fromkeys(groups))
    g = np.asarray(groups)
    blocks = [x[g == k] for k in order]
    sums = np.array([b.sum(axis=0) for b in blocks])
    n = np.array([len(b) for b in blocks])
    ix = np.random.default_rng(seed).integers(len(blocks), size=(draws, len(blocks)))
    return sums[ix].sum(axis=1) / n[ix].sum(axis=1)[:, None]

def holm(p):
    adj = np.empty(len(p))
    running = 0.
    for i, idx in enumerate(np.argsort(p)):
        running = max(running, min(1., (len(p)-i)*p[idx]))
        adj[idx] = running
    return adj

def safe_id(s):
    if not re.fullmatch(r'(?:\d{6}|[a-z_]+_\d[ls])', s):
        raise ValueError(f'unexpected frame identifier {s!r}')
    return s

class Audit:
    def __init__(self, data):
        self.data = data
        self.used = set()
        self.checks = []
        self.detail = {}

    def csv(self, path):
        self.used.add(path)
        return rows(self.data / path)

    def json(self, path):
        self.used.add(path)
        return j(self.data / path)

    def check(self, key, got, expected, source, level='saved observations', n=None,
              note='', mode='precision', tolerance=5e-12):
        if got is None:
            status = 'NOT_REPRODUCED'
        else:
            status = 'PASS' if (
                fmt(got) == fmt(expected) if mode == 'display6' else
                math.isclose(float(got), float(expected), rel_tol=0, abs_tol=tolerance)
                if isinstance(got, (int, float, np.number)) else got == expected
            ) else 'FAIL'
        self.checks.append(dict(check=key, status=status, reproduced=got, reference=expected,
                                reproduced_display=fmt(got), reference_display=fmt(expected),
                                evidence_level=level, n=n, source=source, note=note))

    def compare(self, key, got, expected, source, n=None, note='', level='saved observations'):
        for field, value in got.items():
            self.check(f'{key}.{field}', value, float(expected[field]), source, level, n, note)

    def core(self):
        p = H1+'framewise_ablation.csv'
        rs = self.csv(p)
        assert len(rs) == len({r['frame_id'] for r in rs}) == 32
        cols = ['B16', 'B20_R', 'B20_A', 'B24_EIRD', 'B24_N']
        x = arr(rs, *[k+'_spearman' for k in cols])
        hist = {r['arm']:r for r in self.csv(H1+'ablation_results.csv')}
        self.detail['H1'] = []
        for k, v in zip(cols, x.T):
            assert all(r[k+'_status'] == 'EVALUABLE' for r in rs)
            d = v-x[:,0]
            result = dict(mean_frame_spearman=v.mean(), paired_mean_delta_vs_B16=d.mean(),
                          **dict(zip(('positive_frames','negative_frames','zero_frames'), signs(d))))
            self.compare('H1.'+k, result, hist[k], p+' -> '+H1+'ablation_results.csv', 32)
            self.detail['H1'].append(dict(arm=k, **result))
        self.check('H1.main.B16.display6', x[:,0].mean(), .415975, p, n=32, mode='display6')
        self.check('H1.main.B24.display6', x[:,3].mean(), .469912, p, n=32, mode='display6')
        self.check('H1.main.delta.display6', (x[:,3]-x[:,0]).mean(), .053937, p, n=32, mode='display6')
        self.check('H1.main.improved', signs(x[:,3]-x[:,0])[0], 30, p, n=32)
        self.detail['H1_negative_frames'] = {r['frame_id']:float(d) for r,d in zip(rs,x[:,3]-x[:,0]) if d<0}
        ref=self.csv(H1+'framewise_contrasts.csv')
        assert [q['frame_id'] for q in rs]==[q['frame_id'] for q in ref]
        self.detail['H1_extra_contrasts']={}
        for k,a,b in [('B24_minus_B20_A',3,2),('B24_minus_B20_R',3,1),('B20_A_minus_B20_R',2,1)]:
            d=x[:,a]-x[:,b]
            self.check('H1.'+k+'.framewise_max_error',np.max(abs(d-arr(ref,k).ravel())),0,
                       H1+'framewise_ablation.csv -> framewise_contrasts.csv',n=32)
            self.detail['H1_extra_contrasts'][k]=dict(mean=float(d.mean()), signs=signs(d))
        h2 = self.csv(H2+'h2_framewise.csv')
        self.detail['H2'] = []
        for r in self.csv(H2+'h2_summary.csv'):
            t = [q for q in h2 if q['candidate']==r['candidate'] and q['status']=='EVALUABLE']
            z = arr(t,'B16_spearman','B24_spearman')
            d = z[:,1]-z[:,0]
            result = dict(B16_mean_spearman=z[:,0].mean(), B24_mean_spearman=z[:,1].mean(),
                          equal_frame_paired_delta=d.mean(), evaluable_frames=len(t),
                          valid_pixels=sum(int(q['common_pixels']) for q in t),
                          **dict(zip(('improved_frames','decreased_frames','unchanged_frames'), signs(d))))
            self.compare('H2.'+r['candidate'], result, r, H2+'h2_framewise.csv -> h2_summary.csv', len(t))
            self.detail['H2'].append(dict(candidate=r['candidate'], **result))
        pp=H2+'h2_spatial_framewise.csv'
        sf=self.csv(pp)
        self.detail['H2_spatial']={}
        for r in self.csv(H2+'h2_spatial_summary.csv'):
            sel=[q for q in sf if q['contrast']==r['contrast'] and q['region']==r['region'] and q['status']=='EVALUABLE']
            metrics=('mae','median','p90','p95','max')
            got=dict(zip(['equal_frame_mean_'+k for k in metrics],arr(sel,*metrics).mean(axis=0)))
            self.compare('H2.spatial.'+r['contrast']+'.'+r['region'],got,r,pp,len(sel))
            self.detail['H2_spatial'][r['contrast']+'/'+r['region']]=got
        valid=[q for q in self.csv(H2+'h2_validity.csv') if q['role']=='VAL32']
        self.check('H2.Val32.common_Q',sum(int(q['common_valid_pixels']) for q in valid),1029120,
                   H2+'h2_validity.csv',n=32)
        a = self.csv(DPS+'dps_framewise_results.csv')
        v = [float(r['DPS_unreliability_spearman']) for r in a if r['status']=='EVALUABLE']
        self.detail['DPS'] = float(np.mean(v))
        self.check('DPS.standalone.display6', np.mean(v), .409085, DPS+'dps_framewise_results.csv', n=len(v), mode='display6')

    def v17(self):
        source = V17+'05_conditional_model_comparison.csv'
        rs = [r for r in self.csv(source) if r['status']=='EVALUABLE']
        ids = [r['frame_id'] for r in rs]
        assert len(ids)==len(set(ids))==32
        x = arr(rs,*[a+'_spearman' for a in ARMS])
        assert {r['common_valid_pixels'] for r in rs}=={'32160'}
        meta=self.json(S17+'run_metadata.json')
        draws=paired_draws(x, ids, meta['bootstrap_resamples'], meta['bootstrap_seed'])
        defs = {k+'_equal_frame_mean':(x[:,i],draws[:,i]) for i,k in enumerate(ARMS)}
        defs.update({f'{ARMS[a]}_minus_{ARMS[b]}':(x[:,a]-x[:,b],draws[:,a]-draws[:,b])
                     for a,b in ((1,0),(2,0),(3,0),(2,1),(3,1),(3,2))})
        results={}
        for r in self.csv(S17+'bootstrap_summary.csv'):
            k=r['estimand']
            values,sim=defs[k]
            lo,hi=ci(sim)
            result=dict(estimate=values.mean(),ci_lower=lo,ci_upper=hi,n_frames=len(x))
            self.compare('V17.bootstrap.'+k,result,r,source+' -> '+S17+'bootstrap_summary.csv',32,'Paired frame bootstrap')
            results[k]=result
        self.detail['V17_bootstrap']=results
        contrasts=[x[:,3]-x[:,b] for b in (1,2)]
        tests=[wilcoxon(d, method='exact', alternative='two-sided', correction=False) for d in contrasts]
        assert all(np.all(d!=0) and len(np.unique(abs(d)))==32 for d in contrasts)
        self.detail['V17_tests']=[]
        for r,d,t,adj,target in zip(self.csv(S17+'paired_tests.csv'),contrasts,tests,
                                   holm([t.pvalue for t in tests]),(.041530,.017480)):
            result=dict(mean_difference=d.mean(),median_difference=np.median(d),
                        **dict(zip(('positive_frames','negative_frames','zero_frames'),signs(d))),
                        statistic=t.statistic,p_two_sided=t.pvalue,holm_p_two_focal_comparisons=adj)
            self.compare('V17.signed_rank.'+r['comparison'],result,r,source+' -> '+S17+'paired_tests.csv',32)
            self.check('V17.'+r['comparison']+'.display6',d.mean(),target,source,n=32,mode='display6')
            self.detail['V17_tests'].append(dict(contrast=r['comparison'],**result))
        idx=ids.index('0011896')
        self.detail['V17_difficult_0011896']=dict(zip(ARMS,x[idx].tolist()))
        self.check('V17.0011896.M3_minus_M2',x[idx,3]-x[idx,2],-.195746,source,n=1,mode='display6')
        loo=self.csv(S17+'leave_one_out.csv')
        lx=(x.sum(axis=0)-x)/31
        expected=np.column_stack([lx, lx[:,3]-lx[:,1],lx[:,3]-lx[:,2],lx[:,3]-lx[:,0]])
        assert [r['omitted_frame_id'] for r in loo]==ids
        self.check('V17.leave_one_out.all_fields.max_abs_error',
                   float(np.max(abs(expected-arr(loo,*ARMS,'M3_minus_M1','M3_minus_M2','M3_minus_M0')))),
                   0,source+' -> '+S17+'leave_one_out.csv',n=32)
        p=V17+'high_confidence_joint.csv'
        high=self.csv(p)
        assert [r['frame_id'] for r in high]==ids
        total,bad,flags=arr(high,'high_confidence_pixels','high_confidence_bad3','equal_flag_budget').sum(axis=0).astype(int)
        self.detail['V17_high_confidence']=[]
        for r in self.csv(S17+'high_confusion_summary.csv'):
            tp=sum(int(q[r['arm']+'_bad3_hits']) for q in high)
            fp,fn=flags-tp,bad-tp
            tn=total-tp-fp-fn
            result=dict(region_pixels=int(total),bad3_pixels=int(bad),flagged_pixels=int(flags),
                        tp=tp,fp=int(fp),fn=int(fn),tn=int(tn),precision=tp/flags,recall=tp/bad,
                        f1=2*tp/(2*tp+fp+fn))
            self.compare('V17.high_confidence.'+r['arm'],result,r,p+' -> '+S17+'high_confusion_summary.csv',32)
            self.detail['V17_high_confidence'].append(dict(arm=r['arm'],**result))

    def external(self, pixels=None):
        summary=self.json(V18+'07_external_paired_statistics.json')
        out={}
        for ds,(file,n) in EXT.items():
            p=V18+file
            r=self.csv(p)
            assert len(r)==len({q['frame_id'] for q in r})==n and all(q['status']=='EVALUABLE' for q in r)
            x=arr(r,*[k+'_spearman' for k in ARMS])
            if pixels is not None:
                x=np.asarray([pixels[ds]['all_positive_finite'][q['frame_id']][0] for q in r])
            groups=[q['scene_cluster'] for q in r]
            s=summary[ds]
            n_groups=len(set(groups))
            self.check('V18.'+ds+'.cluster_count',n_groups,s['cluster_count'],p,n=n)
            assert (n_groups==200 and groups==[q['frame_id'] for q in r]) if ds=='KITTI2015' else n_groups==7
            draws=paired_draws(x,groups,s['bootstrap_resamples'],s['bootstrap_seed'])
            counts=sum(int(q['common_primary_pixels']) for q in r)
            record=dict(n=n, groups=n_groups, unit=s['cluster_unit'], seed=s['bootstrap_seed'],
                        resamples=s['bootstrap_resamples'], q=counts, arms={}, contrasts={})
            for i,k in enumerate(ARMS):
                ref=s['arms'][k]
                lo,hi=ci(draws[:,i])
                res=dict(mean=x[:,i].mean(),lo=lo,hi=hi)
                self.compare('V18.'+ds+'.'+k,res,dict(mean=ref['equal_image_mean'],
                    **dict(zip(('lo','hi'),ref['cluster_bootstrap_95']))),p+' -> V18 JSON',n,s['cluster_unit'])
                record['arms'][k]=res
            ds_delta=[x[:,a]-x[:,b] for a,b in PAIR]
            tests=[wilcoxon(d,alternative='two-sided',zero_method='wilcox',correction=False,method='auto') for d in ds_delta]
            for (a,b),d,r0,t,adj in zip(PAIR,ds_delta,s['contrasts'],tests,holm([t.pvalue for t in tests])):
                k=f'M{a}-M{b}'
                assert r0['contrast']==k
                lo,hi=ci(draws[:,a]-draws[:,b])
                info=dict(equal_image_mean_delta=d.mean(),lo=lo,hi=hi,
                          **dict(zip(('improved','declined','equal'),signs(d))))
                self.compare('V18.'+ds+'.'+k,info,dict(r0,**dict(zip(('lo','hi'),r0['cluster_bootstrap_95']))),
                             p+' -> V18 JSON',n,s['cluster_unit'])
                # A rank swap in a tied difference at machine precision can change W. Use
                # the archived frame inputs for the historical nominal signed-rank test.
                record['contrasts'][k]=info
            if pixels is None:
                for r0,t,adj in zip(s['contrasts'],tests,holm([t.pvalue for t in tests])):
                    self.compare('V18.'+ds+'.nominal_test.'+r0['contrast'],
                       dict(wilcoxon_w=t.statistic,wilcoxon_two_sided_nominal_p=t.pvalue,holm_p_three_contrasts=adj),
                       r0,p+' -> V18 JSON',n,'Per-image signed-rank; not cluster-adjusted')
            self.check('V18.'+ds+'.M3-M1.display6',(x[:,3]-x[:,1]).mean(),
                       .004829 if ds=='KITTI2015' else .009204,p,n=n,mode='display6')
            out[ds]=record
        return out

    def families(self):
        p=V82+'04_eth3d_mask_and_family_sensitivity.csv'
        rr=self.csv(p)
        out={}
        for mask in ('all_positive_finite','mask_255'):
            rs=[r for r in rr if r['mask']==mask and r['row_type']=='view']
            assert len(rs)==len({r['view_id'] for r in rs})==27
            fam=sorted({r['family_or_omitted_family'] for r in rs})
            assert len(fam)==7
            out[mask]={'family':{},'leave_one_family_out':{}}
            for r in [r for r in rr if r['mask']==mask and r['row_type']!='view']:
                kind,g=r['row_type'],r['family_or_omitted_family']
                subset=[q for q in rs if kind=='equal_view_mean' or
                        (q['family_or_omitted_family']==g)==(kind=='family')]
                x=arr(subset,*ARMS)
                got=dict(zip(ARMS,x.mean(axis=0)))
                got.update({f'M{a}_minus_M{b}':(x[:,a]-x[:,b]).mean() for a,b in PAIR})
                got.update(views=len(x),common_q_pixels=sum(int(q['common_q_pixels']) for q in subset))
                self.compare(f'V18.2.ETH3D.{mask}.{kind}.{g or "all"}',got,r,p,len(x))
                got['signs_M3_M1']=signs(x[:,3]-x[:,1])
                if kind=='equal_view_mean':
                    out[mask]['total']=got
                    if mask=='mask_255':
                        self.check('V18.2.ETH3D.mask_255.M3-M1.display6',got['M3_minus_M1'],.009328,p,n=27,mode='display6')
                else:
                    out[mask][kind][g]=got
        self.detail['ETH3D_families']=out

    def sensitivity(self):
        out={}
        for fn in ('01_b16_rank_control.csv','02_sequence_overlap_sensitivity.csv','03_equal_frame_sampling.csv'):
            p=V82+fn
            rr=self.csv(p)
            for r in (q for q in rr if q['row_type']=='equal_frame_mean'):
                k=r['condition']
                rs=[q for q in rr if q['condition']==k and q['row_type']=='frame']
                x=arr(rs,*ARMS)
                got=dict(zip(ARMS,x.mean(axis=0)))
                got.update({f'M{a}_minus_M{b}':(x[:,a]-x[:,b]).mean() for a,b in PAIR+[(2,0),(1,0)]})
                got.update(val_frames=len(x),common_q_pixels=sum(int(q['common_q_pixels']) for q in rs))
                self.compare(f'V18.2.{fn[:2]}.{k}',got,r,p,len(x))
                out[k]=dict(got, train_pixels=int(r['train_pixels']))
        self.detail['V18_2_sensitivity']=out
        rank=self.csv(V82+'01_b16_rank_control.csv')
        a=arr([r for r in rank if r['condition']=='original_alpha_1' and r['row_type']=='frame'],*ARMS)
        b=arr([r for r in rank if r['condition']=='reduced_alpha_1' and r['row_type']=='frame'],*ARMS)
        self.check('V18.2.rank.alpha1.max_abs_difference_128',np.max(abs(a-b)),0,V82+'01_b16_rank_control.csv',n=128)
        s=self.json(V82+'readout_parameters.json')['svd_transform']
        self.check('V18.2.rank.retained',sum(v>s['relative_cutoff'] for v in s['singular_value_ratios']),12,
                   V82+'readout_parameters.json',n=16,level='archived singular values')
        p=V84+'03_projection_threshold_sensitivity.csv'
        rr=self.csv(p)
        out={}
        for t in ('0.5','0.75','0.9'):
            rs=[r for r in rr if r['coverage_threshold']==t and r['role']=='VAL32']
            vals={k:np.array([float(r['value']) for r in rs if r['metric']==k]) for k in ['M0','M1','M2','M3']}
            out[t]={}
            for r in (q for q in rr if q['coverage_threshold']==t and q['role']=='VAL32_mean'):
                k=r['metric']
                if '-' in k:
                    a,b=k.split('-')
                    v=vals[a]-vals[b]
                else:
                    v=vals[k]
                self.check(f'V18.4.threshold.{t}.{k}',v.mean(),float(r['value']),p,n=len(v))
                out[t][k]=float(v.mean())
        self.detail['threshold']=out

    def controls(self, cfg):
        p=V84+'01_capacity_matched_control.csv'
        rr=self.csv(p)
        out={}
        for alpha in sorted({r['alpha'] for r in rr},key=float):
            r=[q for q in rr if q['alpha']==alpha and q['row_type']=='frame']
            ids=list(dict.fromkeys(q['frame_id'] for q in r))
            assert len(ids)==32 and {q['common_q_pixels'] for q in r}=={'32160'}
            by={(q['frame_id'],q['metric']):float(q['value']) for q in r}
            keys=[k for k in ('M0','M1','M2','M3','C2','C3',*[f'C1_{s}' for s in cfg['gaussian_seeds']])
                  if (ids[0],k) in by]
            x={k:np.array([by[f,k] for f in ids]) for k in keys}
            x['C1_mean20']=np.mean([x[f'C1_{s}'] for s in cfg['gaussian_seeds']],axis=0)
            ix=np.random.default_rng(cfg['bootstrap_seed']).integers(32,size=(cfg['bootstrap_draws'],32))
            out[alpha]={}
            for q in (q for q in rr if q['alpha']==alpha and q['row_type']=='equal_frame_mean'):
                k=q['metric']
                if '-' in k:
                    a,b=k.split('-')
                    d=x[a]-x[b]
                    lo,hi=ci(d[ix].mean(axis=1))
                    info=dict(value=d.mean(),ci95_low=lo,ci95_high=hi,
                              **dict(zip(('improved','declined','tied'),signs(d))))
                else:
                    info=dict(value=x[k].mean())
                self.compare(f'V18.4.Val32.alpha{alpha}.{k}',info,q,p,32,'Paired frame bootstrap where reported')
                out[alpha][k]=info
            if alpha=='1':
                for k,target in [('M3-C2',.017279),('M3-C3',.042521)]:
                    self.check(f'V18.4.Val32.{k}.display6',out[alpha][k]['value'],target,p,n=32,mode='display6')
        self.detail['V18_4_capacity']=out
        self.detail['V18_4_external']=self.control_external(cfg)

    def control_external(self,cfg,pixels=None):
        p=V84+'04_external_control_transfer.csv'
        rr=self.csv(p)
        result={}
        for ds,mask in sorted({(r['dataset'],r['mask']) for r in rr}):
            rs=[r for r in rr if (r['dataset'],r['mask'])==(ds,mask)]
            frame=[r for r in rs if r['row_type']=='view']
            ids=list(dict.fromkeys(r['view_id'] for r in frame))
            assert len(ids)==EXT[ds][1]
            groups={r['view_id']:r['family_or_omitted_family'] for r in frame}
            ns={r['view_id']:int(r['valid_q_pixels']) for r in frame}
            keys=['M1','M3','C2','C3','M1_cached_f32','M3_cached_f32']
            by={(r['view_id'],r['metric']):float(r['value']) for r in frame}
            x={k:np.array([by[f,k] for f in ids]) for k in keys}
            if pixels is not None:
                for k,jj in [('M1',1),('M3',3),('C2',4),('C3',5)]:
                    x[k]=np.array([pixels[ds][mask][f][0][jj] for f in ids])
            result[ds+'/'+mask]={}
            if pixels is not None:
                rs=[q for q in rs if q['row_type']=='equal_view_mean' and q['metric'] in ('M3-C2','M3-C3')]
            else:
                rs=[q for q in rs if q['row_type']!='view']
            for q in rs:
                kind,g,k=q['row_type'],q['family_or_omitted_family'],q['metric']
                sel=np.array([kind=='equal_view_mean' or (groups[f]==g)==(kind=='family') for f in ids])
                if '-' in k:
                    a,b=k.split('-')
                    delta=(x[a]-x[b])[sel]
                    got=dict(value=delta.mean(),**dict(zip(('improved','declined','tied'),signs(delta))))
                    if kind=='equal_view_mean':
                        gv=[groups[f] for f in ids]
                        draws=paired_draws(delta[:,None],gv,cfg['bootstrap_draws'],cfg['external_bootstrap_seed'],True)
                        got.update(zip(('ci95_low','ci95_high'),ci(draws).ravel()))
                else:
                    got=dict(value=x[k][sel].mean())
                got.update(views=int(sel.sum()),valid_q_pixels=sum(ns[f] for f,use in zip(ids,sel) if use))
                self.compare(f'V18.4.{"pixels." if pixels else ""}{ds}.{mask}.{kind}.{g or "all"}.{k}',
                             got,q,p,int(sel.sum()),'Official image ID' if ds=='KITTI2015' else 'ETH3D scene family',
                             level='frozen readout pixel replay' if pixels else 'saved observations')
                result[ds+'/'+mask].setdefault(kind+(':'+g if g else ''),{})[k]=got
            if not pixels:
                target={('KITTI2015','all_positive_finite'):.002898,
                        ('ETH3D','all_positive_finite'):-.005913,('ETH3D','mask_255'):-.006396}[ds,mask]
                self.check(f'V18.4.{ds}.{mask}.M3-C2.display6',
                    result[ds+'/'+mask]['equal_view_mean']['M3-C2']['value'],target,p,n=len(ids),mode='display6')
        return result

    def intervention(self,cfg):
        p=V84+'02_randomized_intervention_control.csv'
        trace=self.csv(p)
        by=defaultdict(list)
        for r in trace:
            by[r['intervention'],r['frame_id'],r['row_type']].append(r)
        seeds=set(map(str,cfg['intervention']['requested_seeds']))
        assert len(seeds)==20
        self.detail['I3_I4']={}
        for site,target,expected in [('I3',.227300,73),('I4',-.214007,47)]:
            details=[]
            frames=list(dict.fromkeys(r['frame_id'] for r in trace if r['intervention']==site))
            assert len(frames)==4
            for f in frames:
                real=by[site,f,'real_norm']
                real_map={(r['recurrent_call'],r['level']):float(r['real_l2']) for r in real}
                callkeys={(str(c),str(l)) for c in range(1,33) for l in range(2)}
                rsp=by[site,f,'response']
                rand=by[site,f,'random_norm']
                initial=by[site,f,'initial_candidate']
                self.check(f'{site}.{f}.trace_complete',
                    len(real)==64 and set(real_map)==callkeys and len(rsp)==20
                    and {r['seed'] for r in rsp}==seeds and len(rand)==1280
                    and len(initial)==40 and {(r['seed'],r['level']) for r in initial}=={(s,str(l)) for s in seeds for l in range(2)},
                    True,p,n=20,level='call-level archive')
                matches=0
                differences=[]
                max_fraction_err=0.
                max_norm_err=0.
                for rr in rsp:
                    if not rr['random_spearman']:
                        continue
                    norm=[r for r in rand if r['seed']==rr['seed']]
                    assert len(norm)==64 and {(r['recurrent_call'],r['level']) for r in norm}==callkeys
                    assert all(int(r['elements_per_level'])==(293760 if site=='I3' else 8225280) for r in norm)
                    realv=np.array([real_map[r['recurrent_call'],r['level']] for r in norm])
                    ran=np.array([float(r['random_l2']) for r in norm])
                    assert np.array_equal(realv, np.array([float(r['real_l2']) for r in norm]))
                    errors=abs(ran-realv)
                    fraction=float(np.mean(errors<=.1*realv))
                    reported=float(rr['fraction_calls_levels_within_10pct'])
                    max_fraction_err=max(max_fraction_err,abs(fraction-reported))
                    archived_errors=np.array([float(r['relative_l2_error']) for r in norm])
                    max_norm_err=max(max_norm_err,float(np.max(abs(archived_errors-errors/realv))))
                    matches+=int(fraction>=.9)
                    differences.append(float(rr['real_spearman'])-float(rr['random_spearman']))
                self.check(f'{site}.{f}.fraction_max_error',max_fraction_err,0,p,n=len(differences),level='call-level archive')
                self.check(f'{site}.{f}.relative_norm_max_error',max_norm_err,0,p,n=len(rand),level='call-level archive')
                details.append(dict(frame_id=f,n=len(differences),matches=matches,delta=float(np.mean(differences)),
                                    real=float(rsp[0]['real_spearman']),random_mean=float(np.mean([float(r['random_spearman']) for r in rsp if r['random_spearman']]))))
            n=sum(q['n'] for q in details)
            m=sum(q['matches'] for q in details)
            mean=np.mean([q['delta'] for q in details])
            self.check(f'{site}.evaluable_runs',n,80,p,n=4)
            self.check(f'{site}.equal_frame_mean.display6',mean,target,p,n=4,mode='display6')
            self.check(f'{site}.L2_matches',m,expected,p,n=80,level='call-level archive')
            self.detail['I3_I4'][site]=dict(n=n,L2_matches=m,delta=mean,frames=details)


def rank_scores(predictions, target):
    y=rankdata(target)
    x=rankdata(np.asarray(predictions).T,axis=0)
    z=np.corrcoef(np.column_stack((x,y)),rowvar=False)[:-1,-1]
    assert np.isfinite(z).all()
    return z

def replay_controls(x, cfg, ds, fid, alpha='1'):
    r=cfg['rff']
    base=x[:,:17]
    z=(base-np.array(r['mean']))/np.array(r['scale'])
    fourier=np.column_stack((base,.5*np.cos(z@np.array(r['W'])+np.array(r['phase']))))
    stream=np.random.SeedSequence([cfg['shuffle_seed'],cfg['role_codes'][ds],*fid.encode('ascii')])
    perm=np.random.default_rng(stream).permutation(len(x))
    shuffled=np.column_stack((base,x[perm,17:]))
    out=[]
    for name,v in [('C2',fourier),('C3',shuffled)]:
        model=cfg['models'][name]
        w=np.array(model['weights'][alpha])
        out.append(w[0]+((v-model['mean'])/model['scale'])@w[1:])
    return out

def cached_predict(arm, model, x):
    idx=([*range(16),*range(17,25)] if arm==ARMS[2] else range(len(model['mean'])))
    w=np.array(model['weights_with_intercept_first'])
    return w[0]+((x[:,idx]-model['mean'])/model['scale'])@w[1:]

def projected(pred, gt, valid, meta):
    from scipy.ndimage import map_coordinates
    h,w=gt.shape
    top,bottom,left,right=meta['padding']
    qh,qw=meta['q_hw']
    y=(np.arange(qh)+.5)*(h+top+bottom)/qh-.5-top
    x=(np.arange(qw)+.5)*(w+left+right)/qw-.5-left
    yy,xx=np.meshgrid(y,x,indexing='ij')
    coords=np.array([yy,xx])
    def interp(a):
        return map_coordinates(a,coords,order=1,mode='constant',cval=0,prefilter=False)
    coverage=interp(valid.astype(float))
    error=np.zeros((h,w))
    error[valid]=abs(pred[valid]-gt[valid])
    inside=(yy>=0)&(yy<=h-1)&(xx>=0)&(xx<=w-1)
    eligible=inside&(coverage>=.75)
    e=np.full((qh,qw),np.nan)
    np.divide(interp(error),coverage,where=eligible,out=e)
    return e,eligible

def references(ds, fid, a):
    from PIL import Image
    if ds=='KITTI2015':
        with Image.open(a.kitti/'disp_occ_0'/f'{fid}_10.png') as im:
            gt=np.array(im).astype(float)/256
        yield 'all_positive_finite',gt,gt>0
        return
    with (a.eth/fid/'disp0GT.pfm').open('rb') as f:
        assert f.readline().strip()==b'Pf'
        w,h=map(int,f.readline().split())
        scale=float(f.readline())
        gt=np.flipud(np.frombuffer(f.read(),dtype='<f4' if scale<0 else '>f4').reshape(h,w)).astype(float)
    with Image.open(a.eth/fid/'mask0nocc.png') as im:
        mask=np.array(im)
    valid=np.isfinite(gt)&(gt>0)
    assert np.isin(mask,[0,128,255]).all() and np.array_equal(valid,mask>0)
    yield 'all_positive_finite',gt,valid
    yield 'mask_255',gt,mask==255

def pixel_run(audit,cfg,a):
    pixel={}
    for ds,(file,n) in EXT.items():
        out=defaultdict(dict)
        file_rows=audit.csv(V18+file)
        print(f'Pixel replay {ds} ({n} views)',flush=True)
        for i,r in enumerate(file_rows,1):
            fid=safe_id(r['frame_id'])
            with np.load(a.grid/V18/ds/(fid+'.npz'),allow_pickle=False) as f:
                meta=json.loads(str(f['frame_metadata']))
                assert (meta['dataset'],meta['frame_id'],meta['scene_cluster'])==(ds,fid,r['scene_cluster'])
                ordinary=f['ordinary_prediction']
                dps=f['dps_valid']
                x=np.column_stack((f['b16'],f['dps_q'].ravel(),f['eird8'])).astype(float)
                original=f['scores']
                pred=[*original,*replay_controls(x,cfg,ds,fid)]
                for mask,gt,valid in references(ds,fid,a):
                    assert ordinary.shape==gt.shape==tuple(meta['image_hw'])
                    err,available=projected(ordinary,gt,valid,meta)
                    common=available&dps
                    if mask=='all_positive_finite':
                        assert np.array_equal(common,f['primary_valid'])
                        assert np.array_equal(err.astype('float32'),f['primary_error'],equal_nan=True)
                        assert int(common.sum())==int(r['common_primary_pixels'])
                    q=common.ravel()
                    vals=rank_scores([s[q] for s in pred],err[common])
                    out[mask][fid]=(vals,int(q.sum()))
            if i%50==0:
                print(f'  {i}/{n}',flush=True)
        ref=arr(file_rows,*[k+'_spearman' for k in ARMS])
        scores=np.array([out['all_positive_finite'][q['frame_id']][0][:4] for q in file_rows])
        audit.check(f'pixels.{ds}.V18.max_frame_arm_rho_error',np.max(abs(scores-ref)),0,V18+file,
                    'reprojected pixel cache + labels',n*4)
        pixel[ds]=out
    eth=audit.csv(V82+'04_eth3d_mask_and_family_sensitivity.csv')
    rr=[r for r in eth if r['mask']=='mask_255' and r['row_type']=='view']
    ref=arr(rr,*ARMS)
    e=pixel['ETH3D']['mask_255']
    x=np.array([e[r['view_id']][0][:4] for r in rr])
    audit.check('pixels.ETH3D.strict.max_frame_arm_rho_error',np.max(abs(ref-x)),0,
                V82+'04_eth3d_mask_and_family_sensitivity.csv','reprojected pixel cache + labels',len(rr)*4)
    audit.check('pixels.ETH3D.strict.pixel_counts',all(e[r['view_id']][1]==int(r['common_q_pixels']) for r in rr),
                True,V82+'04_eth3d_mask_and_family_sensitivity.csv','reprojected pixel cache + labels',len(rr))
    audit.check('pixels.ETH3D.strict.M3-M1.display6',(x[:,3]-x[:,1]).mean(),.009328,
                V82+'04_eth3d_mask_and_family_sensitivity.csv','reprojected pixel cache + labels',27,mode='display6')
    for (ds,mask), rr in group_control_rows(audit).items():
        computed=pixel[ds][mask]
        errors=[abs(computed[r['view_id']][0][4 if r['metric']=='C2' else 5]-float(r['value'])) for r in rr]
        audit.check(f'pixels.{ds}.{mask}.C2_C3.max_frame_arm_rho_error',max(errors),0,
                    V84+'04_external_control_transfer.csv','frozen readout pixel replay',len(rr))
        audit.check(f'pixels.{ds}.{mask}.control_denominators',all(
            computed[r['view_id']][1]==int(r['valid_q_pixels']) for r in rr),True,
                    V84+'04_external_control_transfer.csv','frozen readout pixel replay',len(rr))
    audit.detail['external_pixel_replay']={ds:{mask:{fid:dict(q=n,**dict(zip(ARMS+['RFF','Shuffle'],v.tolist())))
        for fid,(v,n) in data.items()} for mask,data in masks.items()} for ds,masks in pixel.items()}
    audit.detail['V18_bootstrap_pixel_replay']=audit.external(pixel)
    audit.detail['V18_4_external_pixel_replay']=audit.control_external(cfg,pixel)

def group_control_rows(audit):
    d=defaultdict(list)
    for r in audit.csv(V84+'04_external_control_transfer.csv'):
        if r['row_type']=='view' and r['metric'] in ('C2','C3'):
            d[r['dataset'],r['mask']].append(r)
    return d

def val_pixel_run(audit,cfg,a):
    arms=audit.json(V18+'02_frozen_model_manifest.json')['models']
    historical={r['frame_id']:r for r in audit.csv(V17+'05_conditional_model_comparison.csv') if r['status']=='EVALUABLE'}
    p=V84+'01_capacity_matched_control.csv'
    archived={(r['frame_id'],r['metric']):float(r['value']) for r in audit.csv(p) if r['alpha']=='1' and r['row_type']=='frame'}
    error=defaultdict(list)
    result={}
    v82=audit.json(V82+'readout_parameters.json')
    svd=v82['svd_transform']
    model_sets=v82['fit_parameters']
    sensitivity={}
    for fn in ('01_b16_rank_control.csv','02_sequence_overlap_sensitivity.csv','03_equal_frame_sampling.csv'):
        for r in audit.csv(V82+fn):
            if r['row_type']=='frame' and r['condition'] in model_sets:
                sensitivity[r['condition'],r['frame_id']]=r
    sens_errors=defaultdict(list)
    dps_ref={r['frame_id']:r for r in audit.csv(DPS+'dps_framewise_results.csv') if r['status']=='EVALUABLE'}
    for fid,r in historical.items():
        assert re.fullmatch(r'\d{7}',fid)
        with (np.load(a.grid/H1/'cache/VAL32'/(fid+'.npz'),allow_pickle=False) as f,
              np.load(a.grid/DPS/'dps_maps'/(fid+'.npz'),allow_pickle=False) as d):
            valid=f['valid'].ravel()
            q=f['evaluation_indices']
            assert len(q)==int(r['common_valid_pixels']) and np.array_equal(q,np.flatnonzero(valid))
            assert np.array_equal(valid,d['valid_q'].ravel())
            x=np.column_stack((f['b16'],d['unreliability_q'].ravel(),f['eird8'])).astype(float)
            pred=[*[cached_predict(arm,arms[arm],x) for arm in ARMS],*replay_controls(x,cfg,'VAL32',fid)]
            vals=rank_scores([s[q] for s in pred],f['absolute_error'].ravel()[q])
            result[fid]=vals
            y=f['absolute_error'].ravel()[q]
            standalone=rank_scores([x[q,16]],y)[0]
            error['DPS_U'].append(abs(standalone-float(dps_ref[fid]['DPS_unreliability_spearman'])))
            z=(x[q]-svd['shared_input_mean'])/svd['shared_input_scale']
            reduced=np.column_stack((z[:,:16]@np.asarray(svd['b16_projection_16_by_k']),z[:,16:]))
            for condition,models in model_sets.items():
                ref=sensitivity[condition,fid]
                data=reduced if condition.startswith('reduced') else x[q]
                k=data.shape[1]-9
                columns=[range(k),range(k+1),[*range(k),*range(k+1,data.shape[1])],range(data.shape[1])]
                estimates=[]
                for arm,idx in zip(ARMS,columns):
                    m=models[arm]
                    w=np.asarray(m['weights'])
                    estimates.append(w[0]+((data[:,idx]-m['mean'])/m['scale'])@w[1:])
                sens_errors[condition].append(np.max(abs(rank_scores(estimates,y)-arr([ref],*ARMS)[0])))
        for arm,v in zip(ARMS,vals[:4]):
            error[arm].append(abs(v-float(r[arm+'_spearman'])))
        for arm,v in zip(('C2','C3'),vals[4:]):
            error[arm].append(abs(v-archived[fid,arm]))
    for k,errors in error.items():
        audit.check(f'pixels.Val32.{k}.max_frame_rho_error',max(errors),0,
                    p if k in ('C2','C3') else DPS+'dps_framewise_results.csv' if k=='DPS_U' else V17+'05_conditional_model_comparison.csv',
                    'frozen readout pixel replay',len(errors))
    for k,errors in sens_errors.items():
        audit.check(f'pixels.V18.2.{k}.max_frame_arm_rho_error',max(errors),0,
                    V82+'readout_parameters.json and sensitivity CSVs','frozen readout pixel replay',len(errors)*4)
    audit.detail['Val32_pixel_replay']={k:dict(zip(ARMS+['RFF','Shuffle'],v.tolist())) for k,v in result.items()}
    order=list(dict.fromkeys(r['frame_id'] for r in audit.csv(p) if r['alpha']=='1' and r['row_type']=='frame'))
    vals=np.array([result[fid] for fid in order])
    ix=np.random.default_rng(cfg['bootstrap_seed']).integers(32,size=(cfg['bootstrap_draws'],32))
    for idx,model in [(4,'C2'),(5,'C3')]:
        k='M3-'+model
        ref=audit.detail['V18_4_capacity']['1'][k]
        delta=vals[:,3]-vals[:,idx]
        lo,hi=ci(delta[ix].mean(axis=1))
        audit.compare('pixels.Val32.'+k,dict(value=delta.mean(),ci95_low=lo,ci95_high=hi,
                      **dict(zip(('improved','declined','tied'),signs(delta)))),ref,p,32,
                      'Paired frame bootstrap from replayed pixel correlations','frozen readout pixel replay')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    home=Path(__file__).resolve().parent
    p.add_argument('--data',type=Path,default=home/'data',help='Small CSV/JSON inputs (default: ./data beside script)')
    p.add_argument('--out',required=True,type=Path,help='Directory for new outputs; existing result files are never overwritten')
    p.add_argument('--grid',type=Path,help='Optional local protocol grid root for pixel replay; no model or images loaded')
    p.add_argument('--kitti',type=Path,help='Optional KITTI training root, used for disparity labels only')
    p.add_argument('--eth',type=Path,help='Optional ETH3D labeled training root, used for disparity labels and masks only')
    a=p.parse_args()
    if (a.kitti is None)!=(a.eth is None) or (a.kitti and not a.grid):
        p.error('Provide --kitti and --eth together with --grid for external pixel replay.')
    a.out.mkdir(parents=True,exist_ok=True)
    if any((a.out/f).exists() for f in ('reproduction_results.json','reproduction_comparison.csv')):
        p.error('Choose a fresh --out: a result file already exists.')
    start=time.perf_counter()
    q=Audit(a.data)
    cfg=q.json(V84+'control_parameters.json')
    q.core()
    q.v17()
    q.detail['V18_bootstrap']=q.external()
    q.families()
    q.sensitivity()
    q.controls(cfg)
    q.intervention(cfg)
    if a.grid:
        val_pixel_run(q,cfg,a)
    if a.kitti:
        pixel_run(q,cfg,a)
    gaps=[
        ('manuscript.V18.9.exact_baseline', 'EIRD_JEI_V18_9_Related_Work_Expanded_MathFixed.docx',
         'Exact DOCX unavailable in inspected locations. Displayed targets are from the task and archived experiments.'),
        ('H1.other_three_arms.pixel_replay',H1+'framewise_ablation.csv',
         'B20-R, B20-A and B24-N aggregate results were independently recomputed from rows; their original frozen coefficients were not located.'),
        ('H2.B_Cplus_Cminus.pixel_replay',H2+'h2_framewise.csv',
         'Candidate aggregates were independently computed from rows; candidate-specific frozen readouts were not located.'),
        ('I3_I4.response_pixel_replay',V84+'02_randomized_intervention_control.csv',
         'All response correlations and call-level norm measurements are archived; raw per-seed response arrays were not located.'),
    ]
    for key,src,note in gaps:
        q.check(key,None,None,src,'missing for this layer',note=note)
    counts={k:int(sum(r['status']==k for r in q.checks)) for k in ['PASS','FAIL','NOT_REPRODUCED']}
    exit_code=int(bool(counts['FAIL']))
    out=dict(run=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        seconds=time.perf_counter()-start,pixel_val=bool(a.grid),pixel_external=bool(a.kitti),exit_code=exit_code),
        counts=counts,sources=sorted(q.used),details=q.detail,checks=q.checks)
    write_csv(a.out/'reproduction_comparison.csv',q.checks)
    with (a.out/'reproduction_results.json').open('x',encoding='utf-8') as f:
        json.dump(out,f,indent=2,allow_nan=False,default=lambda o:o.item() if isinstance(o,np.generic) else o.tolist())
    print(json.dumps(counts),f'exit={exit_code}',flush=True)
    return exit_code

if __name__=='__main__':
    sys.exit(main())
