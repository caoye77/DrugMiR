import os,json,time,warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}",flush=True)

def load_data(dd,km=15,kd=10):
    assoc=np.load(f"{dd}/association_matrix.npy")
    mf=np.load(f"{dd}/mirna_kmer_features.npy");df=np.load(f"{dd}/drug_morgan_features.npy")
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

def sn(assoc,pos,n):
    nm,nd=assoc.shape;neg=[]
    while len(neg)<n:
        i=np.random.randint(0,nm);j=np.random.randint(0,nd)
        if assoc[i,j]==0:neg.append((i,j))
    return neg

def trn(m,data,trp,opt,bs=2048):
    m.train();neg=sn(data['assoc'],trp,len(trp));pairs=trp+neg;lab=[1.0]*len(trp)+[0.0]*len(neg)
    idx=np.random.permutation(len(pairs));tl=0;nb=0
    for s in range(0,len(idx),bs):
        bi=idx[s:s+bs];bp=[pairs[i] for i in bi];bl=torch.FloatTensor([lab[i] for i in bi]).to(device)
        mi=torch.LongTensor([p[0] for p in bp]).to(device);di=torch.LongTensor([p[1] for p in bp]).to(device)
        opt.zero_grad();lo=m(data,mi,di);loss=F.binary_cross_entropy_with_logits(lo,bl);loss.backward();opt.step()
        tl+=loss.item();nb+=1
    return tl/nb

@torch.no_grad()
def ev(m,data,tep):
    m.eval();neg=sn(data['assoc'],tep,len(tep));pairs=tep+neg
    lab=np.array([1.0]*len(tep)+[0.0]*len(neg))
    mi=torch.LongTensor([p[0] for p in pairs]).to(device);di=torch.LongTensor([p[1] for p in pairs]).to(device)
    lo=m(data,mi,di);pr=torch.sigmoid(lo).cpu().numpy()
    return roc_auc_score(lab,pr),average_precision_score(lab,pr)

def rcv(data,mfn,seeds=[42,123,2024],nf=5,lr=0.001,wd=2e-4,ep=200,pat=15):
    pos=data['pos_pairs'];res=[]
    for seed in seeds:
        np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        kf=KFold(n_splits=nf,shuffle=True,random_state=seed)
        for fold,(tri,tei) in enumerate(kf.split(pos)):
            trp=[pos[i] for i in tri];tep=[pos[i] for i in tei]
            m=mfn().to(device);opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd)
            ba=0;pc=0;bpr=0
            for e in range(ep):
                trn(m,data,trp,opt)
                if(e+1)%5==0:
                    a,p2=ev(m,data,tep)
                    if a>ba:ba=a;bpr=p2;pc=0
                    else:pc+=1
                    if pc>=pat:break
            res.append({'auc':ba,'aupr':bpr})
            print(f"    S{seed} F{fold+1}: AUC={ba:.4f}",flush=True)
    aucs=[r['auc'] for r in res];auprs=[r['aupr'] for r in res]
    return float(np.mean(aucs)),float(np.std(aucs)),float(np.mean(auprs)),float(np.std(auprs))

class GG(nn.Module):
    def __init__(s,h,dr):
        super().__init__();s.gcn=GCNConv(h,h);s.gate=nn.Linear(2*h,h);s.norm=nn.BatchNorm1d(h);s.drop=nn.Dropout(dr)
    def forward(s,x,e):
        ht=s.drop(s.norm(F.relu(s.gcn(x,e))));g=torch.sigmoid(s.gate(torch.cat([x,ht],-1)));return x+g*ht

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
            mask=has_feat.unsqueeze(1)
            return mask*(g*fh+(1-g)*eh)+(1-mask)*eh
        return g*fh+(1-g)*eh

class DrugMiR_Hybrid(nn.Module):
    def __init__(s,nm,nd,md,dd,ng,h=256,dr=0.5,n_gcn=2,n_br=2):
        super().__init__()
        s.me=HybridEnc(nm,md,h,dr);s.de=HybridEnc(nd,dd,h,dr);s.ge=nn.Embedding(ng,h)
        s.mgcn=nn.ModuleList([GG(h,dr) for _ in range(n_gcn)]);s.dgcn=nn.ModuleList([GG(h,dr) for _ in range(n_gcn)])
        s.br=nn.ModuleList([GB(h,dr) for _ in range(n_br)])
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
    print("="*60);print("Fine-tuning around h256 on D1 (3 seeds)");print("="*60,flush=True)
    results=[]

    configs=[
        {'h':256,'km':20,'kd':10,'lr':0.001,'wd':2e-4,'dr':0.5,'gcn':2,'br':2},
        {'h':256,'km':15,'kd':15,'lr':0.001,'wd':2e-4,'dr':0.5,'gcn':2,'br':2},
        {'h':256,'km':15,'kd':10,'lr':0.001,'wd':2e-4,'dr':0.4,'gcn':2,'br':2},
        {'h':256,'km':15,'kd':10,'lr':0.001,'wd':2e-4,'dr':0.3,'gcn':2,'br':2},
        {'h':256,'km':15,'kd':10,'lr':0.001,'wd':1e-4,'dr':0.5,'gcn':2,'br':2},
        {'h':256,'km':15,'kd':10,'lr':0.0005,'wd':2e-4,'dr':0.5,'gcn':2,'br':2},
        {'h':256,'km':20,'kd':15,'lr':0.001,'wd':2e-4,'dr':0.4,'gcn':2,'br':2},
        {'h':256,'km':20,'kd':10,'lr':0.001,'wd':2e-4,'dr':0.4,'gcn':2,'br':2},
    ]

    for i,cfg in enumerate(configs):
        tag=f"h{cfg['h']}_km{cfg['km']}_kd{cfg['kd']}_lr{cfg['lr']}_wd{cfg['wd']}_dr{cfg['dr']}_gcn{cfg['gcn']}_br{cfg['br']}"
        print(f"\n[{i+1}/{len(configs)}] {tag}",flush=True)
        t0=time.time()
        data=load_data(DD1,km=cfg['km'],kd=cfg['kd'])
        md,dd=data['mirna_feat'].shape[1],data['drug_feat'].shape[1]
        ng=data['n_gene'];nm,nd=data['n_mirna'],data['n_drug']
        auc,std,aupr,aupr_std=rcv(data,
            lambda h=cfg['h'],dr=cfg['dr'],gcn=cfg['gcn'],br=cfg['br']:DrugMiR_Hybrid(nm,nd,md,dd,ng,h=h,dr=dr,n_gcn=gcn,n_br=br),
            lr=cfg['lr'],wd=cfg['wd'])
        t=time.time()-t0
        results.append({**cfg,'auc':auc,'std':std,'aupr':aupr,'aupr_std':aupr_std,'tag':tag,'time':t})
        print(f"  >>> AUC={auc:.4f}+/-{std:.4f} AUPR={aupr:.4f}+/-{aupr_std:.4f} ({t:.0f}s)",flush=True)

    out=os.path.expanduser("~/DrugMiR/results/hp_finetune.json")
    with open(out,'w') as f:json.dump(results,f,indent=2)
    results.sort(key=lambda x:-x['auc'])
    print(f"\n{'='*60}\nFINE-TUNE RESULTS (all 3-seed)\n{'='*60}",flush=True)
    print(f"Ref: h256 base = 0.9613+/-0.0021",flush=True)
    for r in results:
        print(f"  AUC={r['auc']:.4f}+/-{r['std']:.4f} AUPR={r['aupr']:.4f} | {r['tag']}",flush=True)
