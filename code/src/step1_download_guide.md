# Step 1: 数据下载指南

## 1. ncRNADrug（核心关联数据）

**网址**: http://www.jianglab.cn/ncRNADrug/

**下载步骤**:
1. 进入 Browse → 选择 "miRNA" 类别
2. 下载所有 miRNA-Drug 关联数据
3. 或者用爬虫脚本（见下方备选方案）

**备选方案**：如果网页下载不便，尝试以下方式：
- 查找是否有 Supplementary Data 可下载（很多数据库论文会提供）
- 原始论文: "ncRNADrug: a database for validated and predicted ncRNA-drug associations"
- 检查 GitHub 是否有人整理过该数据

**需要的字段**: miRNA name, Drug name, Association type (resistance/sensitivity), Cancer type, PMID

**保存到**: `data/raw/ncRNADrug/mirna_drug_associations.csv`

---

## 2. DrugBank（药物信息）

**网址**: https://go.drugbank.com/releases/latest

**下载步骤**:
1. 注册 DrugBank 账号（Academic License，免费）
2. 下载 "DrugBank XML Database" (drugbank_all_full_database.xml.zip)
3. 解压后放到 `data/raw/DrugBank/`

**备选方案（更轻量）**:
- 下载 "Drug Target Identifiers" CSV
- 下载 "All Drug Links" CSV（含 SMILES）
- 这两个 CSV 文件足够提取我们需要的信息

**需要的字段**: DrugBank ID, Drug Name, SMILES, Target Gene (UniProt ID / Gene Symbol)

**保存到**: `data/raw/DrugBank/`

---

## 3. miRTarBase（miRNA靶基因）

**网址**: https://mirtarbase.cuhk.edu.cn/

**下载步骤**:
1. 进入 Download 页面
2. 下载 "miRTarBase" 完整数据（Excel 格式）
3. 选择最新版本（v10.0 或最新）
4. 物种选择 "Homo sapiens"

**需要的字段**: miRNA, Target Gene, Gene Symbol, Evidence (Strong/Weak)

**保存到**: `data/raw/miRTarBase/hsa_MTI.xlsx`

---

## 4. miRBase（miRNA序列）

**网址**: https://mirbase.org/ 或 FTP: https://mirbase.org/download/

**下载步骤**:
1. 下载 `mature.fa`（成熟 miRNA 序列，所有物种）
2. 我们只需要 hsa（人类）的序列

**保存到**: `data/raw/miRBase/mature.fa`

---

## 下载完成后的文件清单

请确认以下文件存在：
```
data/raw/
├── ncRNADrug/
│   └── mirna_drug_associations.csv  (或 .xlsx / .tsv)
├── DrugBank/
│   ├── drugbank_all_full_database.xml  (完整版)
│   │   或以下轻量版:
│   ├── drug_links.csv
│   └── drug_target_identifiers.csv
├── miRTarBase/
│   └── hsa_MTI.xlsx
└── miRBase/
    └── mature.fa
```

## 注意事项

1. **DrugBank 需要注册**：申请 Academic License，通常 1-2 天批准
2. **ncRNADrug 可能不稳定**：如果网站访问慢，考虑用 Wayback Machine 或联系作者
3. **miRTarBase 只要 Strong Evidence**：弱证据的靶基因关系噪声太大
4. **所有数据只取人类（Homo sapiens）**
