import os,json,warnings
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
warnings.filterwarnings('ignore')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}",flush=True)

def load_data(dd,km=15,kd=10):
    assoc=np.load(f"{dd}/association_matrix.npy");mf=np.load(f"{dd}/mirna_kmer_features.npy");df=np.load(f"{dd}/drug_morgan_features.npy")
    ms=np.load(f"{dd}/mirna_similarity.npy");ds=np.load(f"{dd}/drug_similarity.npy")
    d={'assoc':assoc,'mirna_feat':torch.FloatTensor(mf).to(device),'drug_feat':torch.FloatTensor(df).to(device),
       'n_mirna':assoc.shape[0],'n_drug':assoc.shape[1],
       'mirna_has_feat':torch.FloatTensor((mf.sum(1)>0).astype(float)).to(device),
       'drug_has_feat':torch.FloatTensor((df.sum(1)>0).astype(float)).to(device)}
    s1,d1=[],[]
    for i in range(d['n_mirna']):
        si=ms[i].copy();si[i]=-1;tk=np.argsort(si)[-km:]
        for j in tk:s1.extend([i,j]);d1.extend([j,i])
    d['mirna_sim_edge']=torch.LongTensor([s1,d1]).to(device)
    s2,d2=[],[]
    for i in range(d['n_drug']):
        si=ds[i].copy();si[i]=-1;tk=np.argsort(si)[-kd:]
        for j in tk:s2.extend([i,j]);d2.extend([j,i])
    d['drug_sim_edge']=torch.LongTensor([s2,d2]).to(device)
    mg=np.load(f"{dd}/mirna_gene_matrix.npy");dg=np.load(f"{dd}/drug_gene_matrix.npy")
    mg_r,mg_c=np.nonzero(mg);dg_r,dg_c=np.nonzero(dg)
    d['mg_src']=torch.LongTensor(mg_r).to(device);d['mg_dst']=torch.LongTensor(mg_c).to(device)
    d['dg_src']=torch.LongTensor(dg_r).to(device);d['dg_dst']=torch.LongTensor(dg_c).to(device)
    d['n_gene']=max(mg_c.max() if len(mg_c)>0 else 0,dg_c.max() if len(dg_c)>0 else 0)+1
    pr,pc=np.nonzero(assoc);d['pos_pairs']=list(zip(pr.tolist(),pc.tolist()))
    return d

def sn(a,p,n):
    nm,nd=a.shape;neg=[]
    while len(neg)<n:
        i=np.random.randint(0,nm);j=np.random.randint(0,nd)
        if a[i,j]==0:neg.append((i,j))
    return neg

class GG(nn.Module):
    def __init__(s,h,dr):
        super().__init__();s.gcn=GCNConv(h,h);s.gate=nn.Linear(2*h,h);s.norm=nn.BatchNorm1d(h);s.drop=nn.Dropout(dr)
    def forward(s,x,e):ht=s.drop(s.norm(F.relu(s.gcn(x,e))));g=torch.sigmoid(s.gate(torch.cat([x,ht],-1)));return x+g*ht

class GB(nn.Module):
    def __init__(s,h,dr):
        super().__init__();s.mg=nn.Linear(2*h,h);s.dg=nn.Linear(2*h,h);s.norm=nn.BatchNorm1d(h);s.drop=nn.Dropout(dr)
    def forward(s,mh,dh,gh,ms,md,ds,dd,ng):
        gm=scatter_mean(mh[ms],md,dim=0,dim_size=ng);gd=scatter_mean(dh[ds],dd,dim=0,dim_size=ng)
        ga=s.drop(s.norm(F.relu(gh+gm+gd)));mfg=scatter_mean(ga[md],ms,dim=0,dim_size=mh.size(0));dfg=scatter_mean(ga[dd],ds,dim=0,dim_size=dh.size(0))
        return mh+torch.sigmoid(s.mg(torch.cat([mh,mfg],-1)))*mfg,dh+torch.sigmoid(s.dg(torch.cat([dh,dfg],-1)))*dfg,ga

class HybridEnc(nn.Module):
    def __init__(s,n,fd,h,dr):
        super().__init__()
        s.feat=nn.Sequential(nn.Linear(fd,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(dr),nn.Linear(h,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(dr))
        s.emb=nn.Embedding(n,h);s.gate=nn.Linear(2*h,h)
    def forward(s,feat,has_feat=None):
        fh=s.feat(feat);eh=s.emb.weight;g=torch.sigmoid(s.gate(torch.cat([fh,eh],-1)))
        if has_feat is not None:
            mask=has_feat.unsqueeze(1);return mask*(g*fh+(1-g)*eh)+(1-mask)*eh
        return g*fh+(1-g)*eh

class DrugMiR(nn.Module):
    def __init__(s,nm,nd,md,dd,ng,h=256,dr=0.5):
        super().__init__()
        s.me=HybridEnc(nm,md,h,dr);s.de=HybridEnc(nd,dd,h,dr);s.ge=nn.Embedding(ng,h)
        s.mgcn=nn.ModuleList([GG(h,dr) for _ in range(2)]);s.dgcn=nn.ModuleList([GG(h,dr) for _ in range(2)])
        s.br=nn.ModuleList([GB(h,dr) for _ in range(2)])
        s.pred=nn.Sequential(nn.Linear(6*h,2*h),nn.BatchNorm1d(2*h),nn.ReLU(),nn.Dropout(dr),nn.Linear(2*h,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(dr),nn.Linear(h,1))
    def forward(s,data,mi,di):
        m0=s.me(data['mirna_feat'],data.get('mirna_has_feat'));d0=s.de(data['drug_feat'],data.get('drug_has_feat'))
        mh=m0;dh=d0
        for l in s.mgcn:mh=l(mh,data['mirna_sim_edge'])
        for l in s.dgcn:dh=l(dh,data['drug_sim_edge'])
        mb=m0;db=d0;gh=s.ge.weight
        for l in s.br:mb,db,gh=l(mb,db,gh,data['mg_src'],data['mg_dst'],data['dg_src'],data['dg_dst'],data['n_gene'])
        return s.pred(torch.cat([torch.cat([m0,mh,mb],-1)[mi],torch.cat([d0,dh,db],-1)[di]],-1)).squeeze(-1)

if __name__=='__main__':
    DD1=os.path.expanduser("~/DrugMiR/data/dataset1")
    print("Case Study: Train on full D1, predict all unknown pairs",flush=True)
    data=load_data(DD1)
    md,dd=data['mirna_feat'].shape[1],data['drug_feat'].shape[1]
    ng=data['n_gene'];nm,nd=data['n_mirna'],data['n_drug']
    assoc=data['assoc']
    pos=data['pos_pairs']
    print(f"{nm} miRNAs, {nd} drugs, {len(pos)} known pairs",flush=True)

    all_scores=[]
    for seed in [42,123,2024,7,999]:
        np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        m=DrugMiR(nm,nd,md,dd,ng).to(device)
        opt=torch.optim.Adam(m.parameters(),lr=0.0005,weight_decay=2e-4)
        for e in range(200):
            m.train()
            neg=sn(assoc,pos,len(pos));pairs=pos+neg;lab=[1.0]*len(pos)+[0.0]*len(neg)
            idx=np.random.permutation(len(pairs))
            for s in range(0,len(idx),2048):
                bi=idx[s:s+2048];bp=[pairs[i] for i in bi];bl=torch.FloatTensor([lab[i] for i in bi]).to(device)
                mi=torch.LongTensor([p[0] for p in bp]).to(device);di=torch.LongTensor([p[1] for p in bp]).to(device)
                opt.zero_grad();lo=m(data,mi,di);loss=F.binary_cross_entropy_with_logits(lo,bl);loss.backward();opt.step()
        m.eval()
        scores=np.zeros((nm,nd))
        with torch.no_grad():
            for j in range(nd):
                mi=torch.arange(nm).to(device);di=torch.full((nm,),j,dtype=torch.long).to(device)
                lo=m(data,mi,di);scores[:,j]=torch.sigmoid(lo).cpu().numpy()
        all_scores.append(scores)
        print(f"  Seed {seed} done",flush=True)

    avg_scores=np.mean(all_scores,axis=0)
    unknown_scores=avg_scores.copy()
    unknown_scores[assoc>0]=-1

    np.save(os.path.expanduser("~/DrugMiR/results/case_study_scores.npy"),avg_scores)
    np.save(os.path.expanduser("~/DrugMiR/results/case_study_unknown.npy"),unknown_scores)

    import pandas as pd
    try:
        drug_map=pd.read_csv(os.path.expanduser("~/DrugMiR/data/dataset1/drug_mapping.csv"))
        drug_names=drug_map['drug_name'].tolist()
    except:
        drug_names=[f"Drug_{i}" for i in range(nd)]
    try:
        mirna_map=pd.read_csv(os.path.expanduser("~/DrugMiR/data/dataset1/mirna_mapping.csv"))
        mirna_names=mirna_map['mirna_name'].tolist()
    except:
        mirna_names=[f"miRNA_{i}" for i in range(nm)]

    target_drugs=['Sorafenib','Doxorubicin','Cisplatin','5-Fluorouracil','Paclitaxel','Gemcitabine','Docetaxel','Tamoxifen']
    results={}
    for tgt in target_drugs:
        matches=[i for i,n in enumerate(drug_names) if tgt.lower() in n.lower()]
        if not matches:
            print(f"\n{tgt}: not found",flush=True);continue
        di=matches[0]
        known_mirnas=[i for i in range(nm) if assoc[i,di]>0]
        col=unknown_scores[:,di]
        top_idx=np.argsort(col)[::-1][:20]
        top_list=[]
        for rank,mi in enumerate(top_idx):
            if col[mi]<0:continue
            top_list.append({'rank':rank+1,'mirna':mirna_names[mi],'score':float(col[mi])})
        results[tgt]={'drug_idx':int(di),'drug_name':drug_names[di],'known_count':len(known_mirnas),'top20':top_list}
        print(f"\n{tgt} (idx={di}, {len(known_mirnas)} known):",flush=True)
        for t in top_list[:10]:
            print(f"  #{t['rank']} {t['mirna']} score={t['score']:.4f}",flush=True)

    out=os.path.expanduser("~/DrugMiR/results/case_study.json")
    with open(out,'w') as f:json.dump(results,f,indent=2)
    print(f"\nSaved to {out}",flush=True)
    print("DONE",flush=True)
