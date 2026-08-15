import json, numpy as np, os
S5=[42,123,2024,7777,9999]
def load(path,m):
    return np.array([json.load(open(f"coldstart_outputs/{path}_seed{s}.json"))[m]['mean'] for s in S5 if os.path.exists(f"coldstart_outputs/{path}_seed{s}.json")])
meth=[('DrugMiR','drugmirV_full_{ds}_{st}'),('MPHGNN','mphgnn_{ds}_{st}'),('DMR-PEG','dmrpeg_{ds}_{st}'),('GAM-MDR','gammdr_{ds}_{st}')]
setlabel={'S2':'S2: miRNA-cold','S3':'S3: drug-cold','S4':'S4: pair-cold'}
def cell(v, rank):
    mean,std=v.mean(),v.std()
    if rank==0: return f"\\textbf{{{mean:.4f}}}\\textsubscript{{$\\pm${std:.4f}}}"
    if rank==1: return f"\\underline{{{mean:.4f}}}\\textsubscript{{$\\pm${std:.4f}}}"
    return f"{mean:.4f}\\textsubscript{{$\\pm${std:.4f}}}"
print("==== TABLE IV BODY ====")
for st in ['S2','S3','S4']:
    cols={}
    for ds in ['D1','D2']:
        for mt in ['auc','aupr','f1']:
            vals=[load(pre.format(ds=ds,st=st),mt).mean() for _,pre in meth]
            order=sorted(range(4),key=lambda i:-vals[i]); cols[(ds,mt)]={order[0]:0,order[1]:1}
    for mi,(name,pre) in enumerate(meth):
        cells=[cell(load(pre.format(ds=ds,st=st),mt), cols[(ds,mt)].get(mi,9)) for ds in ['D1','D2'] for mt in ['auc','aupr','f1']]
        prefix=f"\\multirow{{4}}{{*}}{{{setlabel[st]}}} & {name}" if mi==0 else f" & {name}"
        print(f"{prefix} & "+" & ".join(cells)+" \\\\")
    print("\\midrule" if st!='S4' else "\\bottomrule")
print("\n==== TABLE V BODY ====")
for name,pre in [('Full DrugMiR','drugmirV_full_{ds}_S4'),('w/o Hybrid Enc.','drugmirV_no_hybrid_{ds}_S4'),('w/o Gene Bridge','drugmirV_no_bridge_{ds}_S4')]:
    cells=[f"{load(pre.format(ds=ds),mt).mean():.4f}\\textsubscript{{$\\pm${load(pre.format(ds=ds),mt).std():.3f}}}" for ds in ['D1','D2'] for mt in ['auc','aupr']]
    print(f"{name} & "+" & ".join(cells)+" \\\\")
print("\n==== §III-D NUMBERS ====")
def auc(pre,ds): return load(pre.format(ds=ds),'auc')
fa1,fa2=auc('drugmirV_full_{ds}_S4','D1'),auc('drugmirV_full_{ds}_S4','D2')
nh1,nh2=auc('drugmirV_no_hybrid_{ds}_S4','D1'),auc('drugmirV_no_hybrid_{ds}_S4','D2')
nb2=auc('drugmirV_no_bridge_{ds}_S4','D2'); dmr1,dmr2=auc('dmrpeg_{ds}_S4','D1'),auc('dmrpeg_{ds}_S4','D2')
print(f"S4 DrugMiR D1={fa1.mean():.4f} D2={fa2.mean():.4f}")
print(f"去Hybrid D1 Δ{(fa1.mean()-nh1.mean())*100:+.1f} | D2 Δ{(fa2.mean()-nh2.mean())*100:+.1f} ; 去Bridge D2 Δ{(fa2.mean()-nb2.mean())*100:+.1f}")
dif=fa2-nh2; print(f"D2去Hybrid t={dif.mean()/(dif.std(ddof=1)/len(dif)**0.5):.2f} df={len(dif)-1}")
print(f"DrugMiR-DMRPEG S4 D1 +{(fa1.mean()-dmr1.mean())*100:.1f} D2 +{(fa2.mean()-dmr2.mean())*100:.1f}")
print(f"S2 DrugMiR D1={auc('drugmirV_full_{ds}_S2','D1').mean():.4f} D2={auc('drugmirV_full_{ds}_S2','D2').mean():.4f} | DMRPEG D1={auc('dmrpeg_{ds}_S2','D1').mean():.4f} D2={auc('dmrpeg_{ds}_S2','D2').mean():.4f}")
print(f"S3 GAMMDR-D1={auc('gammdr_{ds}_S3','D1').mean():.4f} DrugMiR-D2={auc('drugmirV_full_{ds}_S3','D2').mean():.4f} MPHGNN D1={auc('mphgnn_{ds}_S3','D1').mean():.4f} D2={auc('mphgnn_{ds}_S3','D2').mean():.4f}")
