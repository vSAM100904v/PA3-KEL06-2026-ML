# -*- coding: utf-8 -*-
"""
================================================================
MAMACARE AI v2.0 - SISTEM PREDIKSI RISIKO KEHAMILAN MULTI-DIMENSI
================================================================
Proyek Machine Learning - Institut Teknologi Del
Mata Kuliah  : 4143104 - Pembelajaran Mesin
Dosen        : Oppir Hutapea, S.Tr.Kom., M.Kom
Domain       : Kesehatan (Maternal Health)
Topik ML     : Klasifikasi Multi-Label (Multi-Output)

PERUBAHAN v2.0 (Revisi Dosen Penguji):
  - Sistem kini mendeteksi JENIS RISIKO SPESIFIK secara bersamaan:
      * Risiko Kematian Ibu (Maternal Mortality Risk)
      * Risiko Keguguran / Persalinan Prematur
      * Risiko Preeklampsia / Hipertensi
      * Risiko Anemia Berat
      * Risiko BBLR & Potensi Stunting Bayi
      * Risiko Infeksi Menular (HIV/Sifilis/HepB)
  - Setiap jenis risiko memiliki model biner tersendiri (MultiOutputClassifier)
  - Output mencakup: tingkat risiko KESELURUHAN + breakdown per jenis risiko
    + penjelasan klinis berbasis panduan Kemenkes RI / WHO
  - Sistem tidak lagi hanya menghasilkan label, tetapi juga ALASAN KLINIS
    yang dapat dipertanggungjawabkan oleh dokter/bidan

Cara menjalankan:
    python mamacare_train_v2.py

Output:
    model_risk_overall.pkl    — model Random Forest utama (3 kelas)
    model_risk_types.pkl      — model multi-output (6 jenis risiko)
    scaler_v2.pkl
    feature_cols_v2.pkl
    label_map_v2.pkl
    risk_type_names.pkl
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, roc_auc_score, hamming_loss
)
from sklearn.utils import resample

print("=" * 70)
print("  MAMACARE AI v2.0 – MULTI-DIMENSIONAL PREGNANCY RISK DETECTION")
print("=" * 70)

# ================================================================
# 1. MEMUAT DATASET
# ================================================================
print("\n[1/9] Memuat dataset...")
df = pd.read_excel('C:/Users/samue/ML v0.2/dataset_anc_bumil_v2.xlsx')
print(f"  ✅ Dataset: {df.shape[0]:,} baris, {df.shape[1]} kolom")

# ================================================================
# 2. PREPROCESSING
# ================================================================
print("\n[2/9] Preprocessing...")

df['Tgl Lahir'] = pd.to_datetime(df['Tgl Lahir'], dayfirst=True, errors='coerce')
df['Usia_Ibu'] = df['Tgl Lahir'].apply(
    lambda x: (pd.Timestamp('2025-10-01') - x).days // 365 if pd.notnull(x) else 28
).clip(14, 55)

df['Tinggi Fundus Uteri (cm)'] = df.groupby('Trimester')['Tinggi Fundus Uteri (cm)'].transform(
    lambda x: x.fillna(x.median()))
df['Tindakan'] = df['Tindakan'].fillna('Normal')
df['Riwayat Penyakit'] = df['Riwayat Penyakit'].fillna('Tidak Ada')
df['IMT'] = df['IMT'].fillna(df['IMT'].median())
df['LiLA (cm)'] = df['LiLA (cm)'].fillna(df['LiLA (cm)'].median())
df['Hemoglobin/Hb (g/dL)'] = df['Hemoglobin/Hb (g/dL)'].fillna(df['Hemoglobin/Hb (g/dL)'].median())

print(f"  ✅ Missing values setelah preprocessing: {df.isnull().sum().sum()}")

# ================================================================
# 3. FEATURE ENGINEERING
# ================================================================
print("\n[3/9] Feature engineering...")

df['MAP'] = (df['TD Sistolik'] + 2 * df['TD Diastolik']) / 3
df['Pulse_Pressure'] = df['TD Sistolik'] - df['TD Diastolik']
df['Trimester_Num'] = df['Trimester'].map({'I': 1, 'II': 2, 'III': 3}).fillna(2)

df['Riwayat_Enc']   = (df['Riwayat Penyakit'].str.strip().str.lower() != 'tidak ada').astype(int)
df['Riwayat_Berat'] = df['Riwayat Penyakit'].str.strip().str.lower().apply(
    lambda x: 1 if x in ('hipertensi', 'jantung', 'tb') else 0)
df['Riwayat_Metabolik'] = df['Riwayat Penyakit'].str.strip().str.lower().apply(
    lambda x: 1 if x in ('diabetes', 'hepatitis b') else 0)

df['HIV_Rek']  = (df['Tripel Eliminasi - HIV'].str.strip()    == 'Reaktif').astype(int)
df['Sif_Rek']  = (df['Tripel Eliminasi - Sifilis'].str.strip() == 'Reaktif').astype(int)
df['HepB_Rek'] = (df['Tripel Eliminasi - Hep B'].str.strip()   == 'Reaktif').astype(int)
df['Tripel_Total'] = df['HIV_Rek'] + df['Sif_Rek'] + df['HepB_Rek']

df['KatIMT_Enc']    = df['Kategori IMT'].map({'Kurus': 0, 'Normal': 1, 'Gemuk': 2, 'Obesitas': 3}).fillna(1)
df['KatLiLA_Enc']   = (df['Kategori LiLA'].str.strip() == 'Risiko KEK').astype(int)
df['KatTD_Enc']     = df['Kategori TD'].map({'Normal': 0, 'Pra-Hipertensi': 1, 'Hipertensi': 2}).fillna(0)
df['KatHb_Enc']     = (df['Kategori Hb'].str.strip() == 'Anemia').astype(int)
df['Imunisasi_Enc'] = df['Status Imunisasi TD'].map({'T1':1,'T2':2,'T3':3,'T4':4,'T5':5}).fillna(3)

df['Usia_Risiko']    = ((df['Usia_Ibu'] < 19) | (df['Usia_Ibu'] > 38)).astype(int)
df['IMT_Kurus']      = (df['IMT'] < 18.5).astype(int)
df['IMT_Obese']      = (df['IMT'] >= 30.0).astype(int)
df['Anemia_Flag']    = (df['Hemoglobin/Hb (g/dL)'] < 11.0).astype(int)
df['Anemia_Berat']   = (df['Hemoglobin/Hb (g/dL)'] < 8.0).astype(int)
df['Hiper_Flag']     = (df['TD Sistolik'] >= 140).astype(int)
df['Hiper_Berat']    = (df['TD Sistolik'] >= 160).astype(int)
df['MAP_Tinggi']     = (df['MAP'] > 107).astype(int)
df['KEK_Flag']       = (df['LiLA (cm)'] < 23.5).astype(int)
df['Abortus_Tinggi'] = (df['Abortus'] >= 2).astype(int)
df['Grandemulti']    = (df['Gravida'] >= 5).astype(int)
df['Primigravida']   = (df['Gravida'] == 1).astype(int)
df['ANC_Terlambat']  = (df['Kunjungan ANC Ke-'] <= 2).astype(int)

df['Risk_Score'] = (
    df['Anemia_Flag'] * 2 + df['Anemia_Berat'] * 3 +
    df['KatTD_Enc'] * 3  + df['Hiper_Berat'] * 5 +
    df['KEK_Flag'] * 2   + df['KatIMT_Enc'].apply(lambda x: 2 if x in [0, 3] else 0) +
    df['Riwayat_Enc'] * 2 + df['Riwayat_Berat'] * 3 +
    df['Tripel_Total'] + df['Usia_Risiko'] +
    df['Abortus_Tinggi'] * 2 + df['Grandemulti'] +
    df['ANC_Terlambat'] + df['MAP_Tinggi'] * 2
)

print(f"  ✅ Total fitur yang dibangun: akan dihitung setelah definisi feature_cols")

feature_cols = [
    # Data dasar ibu
    'Usia_Ibu', 'Usia Kehamilan (minggu)', 'Trimester_Num',
    'Gravida', 'Para', 'Abortus',
    # Antropometri
    'IMT', 'LiLA (cm)', 'Tinggi Fundus Uteri (cm)',
    # Tanda vital
    'TD Sistolik', 'TD Diastolik', 'MAP', 'Pulse_Pressure',
    # Lab
    'Hemoglobin/Hb (g/dL)',
    # Riwayat & tripel eliminasi
    'Riwayat_Enc', 'Riwayat_Berat', 'Riwayat_Metabolik',
    'HIV_Rek', 'Sif_Rek', 'HepB_Rek', 'Tripel_Total',
    # Kategori encoded
    'KatIMT_Enc', 'KatLiLA_Enc', 'KatTD_Enc', 'KatHb_Enc', 'Imunisasi_Enc',
    # Binary flags klinis
    'Anemia_Flag', 'Anemia_Berat', 'Hiper_Flag', 'Hiper_Berat', 'MAP_Tinggi',
    'KEK_Flag', 'IMT_Kurus', 'IMT_Obese',
    'Usia_Risiko', 'Abortus_Tinggi', 'Grandemulti', 'Primigravida', 'ANC_Terlambat',
    # Skor komposit
    'Risk_Score',
    # Kunjungan ANC
    'Kunjungan ANC Ke-',
]

print(f"  ✅ Total fitur: {len(feature_cols)}")

X = df[feature_cols].fillna(0)

# ================================================================
# 4. TARGET ENGINEERING — LABEL RISIKO KESELURUHAN (3 KELAS)
# ================================================================
print("\n[4/9] Membentuk target label risiko keseluruhan...")

def compute_risk_overall(row):
    """
    Label risiko keseluruhan — SEPENUHNYA DITURUNKAN dari 6 jenis risiko spesifik.
    Logika revisi dosen penguji:

      2 = PERLU RUJUKAN  → jika terdeteksi salah satu dari 5 risiko berat:
            Risiko Kematian Ibu, Risiko Keguguran/Prematur,
            Risiko Preeklampsia/Hipertensi, Risiko Anemia Berat,
            Risiko Infeksi Menular (Tripel Eliminasi)

      1 = PERLU TINDAKAN → jika hanya Risiko BBLR/Stunting yang terdeteksi
            (tanpa satupun dari 5 risiko PERLU RUJUKAN di atas)

      0 = NORMAL         → tidak ada satupun dari 6 risiko yang terdeteksi

    Dengan pendekatan ini, label overall KONSISTEN dengan label jenis risiko,
    dan dokter tidak perlu menebak-nebak jenis risiko karena sudah tersedia.
    """
    # Hitung ulang risk_types untuk baris ini (reuse fungsi compute_risk_types)
    types = _compute_risk_types_raw(row)
    r_kematian, r_keguguran, r_preeklampsia, r_anemia, r_stunting, r_infeksi = types

    # 5 risiko yang mengarah ke PERLU RUJUKAN
    rujukan_flags = [r_kematian, r_keguguran, r_preeklampsia, r_anemia, r_infeksi]
    if any(rujukan_flags):
        return 2

    # Hanya risiko stunting → PERLU TINDAKAN
    if r_stunting:
        return 1

    return 0


def _compute_risk_types_raw(row):
    """Versi raw (tuple) dari compute_risk_types untuk dipakai di compute_risk_overall."""
    riwayat = str(row['Riwayat Penyakit']).strip().lower()
    hiv     = str(row['Tripel Eliminasi - HIV']).strip()
    sifilis = str(row['Tripel Eliminasi - Sifilis']).strip()
    hepb    = str(row['Tripel Eliminasi - Hep B']).strip()
    td_s    = float(row['TD Sistolik']); td_d = float(row['TD Diastolik'])
    hb      = float(row['Hemoglobin/Hb (g/dL)'])
    imt     = float(row['IMT']); lila = float(row['LiLA (cm)'])
    usia    = float(row['Usia_Ibu']); gravida = int(row['Gravida'])
    abortus = int(row['Abortus']); uk = int(row['Usia Kehamilan (minggu)'])
    map_val = (td_s + 2 * td_d) / 3

    r_kematian = 0
    if td_s >= 160 or td_d >= 110: r_kematian = 1
    elif hb < 8.0: r_kematian = 1
    elif hiv == 'Reaktif' and hb < 10.0: r_kematian = 1
    elif riwayat == 'jantung': r_kematian = 1
    elif riwayat == 'tb' and hb < 10.0: r_kematian = 1
    elif abortus >= 3: r_kematian = 1
    elif map_val > 110 and riwayat == 'hipertensi': r_kematian = 1

    r_keguguran = 0
    if abortus >= 2: r_keguguran = 1
    elif gravida >= 5 and uk < 32: r_keguguran = 1
    elif td_s >= 150 and uk < 34: r_keguguran = 1
    elif hb < 9.0: r_keguguran = 1
    elif riwayat == 'diabetes': r_keguguran = 1
    elif usia < 18 and gravida == 1: r_keguguran = 1
    elif imt < 17.0: r_keguguran = 1

    r_preeklampsia = 0
    if td_s >= 140 or td_d >= 90: r_preeklampsia = 1
    elif map_val > 107: r_preeklampsia = 1
    elif riwayat == 'hipertensi': r_preeklampsia = 1
    elif imt >= 30 and uk > 20 and td_s >= 130: r_preeklampsia = 1
    elif gravida == 1 and usia > 35 and td_s >= 120: r_preeklampsia = 1

    r_anemia = 0
    if hb < 8.0: r_anemia = 1
    elif hb < 9.5 and lila < 23.5: r_anemia = 1
    elif hb < 10.0 and imt < 18.5: r_anemia = 1
    elif hb < 11.0 and riwayat == 'anemia': r_anemia = 1

    r_stunting = 0
    if lila < 23.5: r_stunting = 1
    elif imt < 18.5: r_stunting = 1
    elif hb < 10.0 and lila < 25.0: r_stunting = 1
    elif riwayat == 'diabetes' and imt >= 30: r_stunting = 1
    elif usia < 19: r_stunting = 1
    elif abortus >= 2 and hb < 11.0: r_stunting = 1

    r_infeksi = 0
    if hiv == 'Reaktif': r_infeksi = 1
    elif sifilis == 'Reaktif': r_infeksi = 1
    elif hepb == 'Reaktif': r_infeksi = 1
    elif riwayat == 'hepatitis b': r_infeksi = 1

    return (r_kematian, r_keguguran, r_preeklampsia, r_anemia, r_stunting, r_infeksi)

df['RISIKO'] = df.apply(compute_risk_overall, axis=1)
label_map = {0: 'NORMAL', 1: 'PERLU TINDAKAN', 2: 'PERLU RUJUKAN'}
df['RISIKO_LABEL'] = df['RISIKO'].map(label_map)

print("  Distribusi kelas risiko keseluruhan:")
for lbl, cnt in df['RISIKO_LABEL'].value_counts().items():
    print(f"    {lbl:<25}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

y_overall = df['RISIKO'].values

# ================================================================
# 5. TARGET ENGINEERING — 6 JENIS RISIKO SPESIFIK (MULTI-LABEL)
# ================================================================
print("\n[5/9] Membentuk 6 label jenis risiko spesifik (multi-label)...")

"""
REFERENSI KLINIS:
- Kemenkes RI: Pedoman ANC Terpadu (2022)
- WHO ANC Guidelines (2016)  
- Buku KIA (2023)
- POGI: Tata Laksana Preeklampsia (2019)
- PMK No. 97 Tahun 2014 tentang Pelayanan Kesehatan Masa Sebelum Hamil
"""

def compute_risk_types(row):
    """
    Menghitung 6 label risiko spesifik secara independen.
    Setiap label biner (0/1).
    
    Jenis risiko:
    [0] risiko_kematian_ibu   — ancaman langsung jiwa ibu
    [1] risiko_keguguran_prematur — risiko kehilangan/prematuritas janin
    [2] risiko_preeklampsia   — risiko pre/eklampsia & hipertensi
    [3] risiko_anemia_berat   — anemia dengan dampak klinis signifikan
    [4] risiko_bblr_stunting  — risiko bayi berat lahir rendah / stunting
    [5] risiko_infeksi_menular — HIV, Sifilis, HepB aktif
    """
    return list(_compute_risk_types_raw(row))

risk_type_names = [
    'Risiko Kematian Ibu',
    'Risiko Keguguran / Persalinan Prematur',
    'Risiko Preeklampsia / Hipertensi',
    'Risiko Anemia Berat',
    'Risiko BBLR / Stunting Bayi',
    'Risiko Infeksi Menular (Tripel Eliminasi)',
]

risk_types_data = df.apply(compute_risk_types, axis=1).tolist()
y_types = np.array(risk_types_data)  # shape: (n_samples, 6)

print("  Distribusi per jenis risiko:")
for i, name in enumerate(risk_type_names):
    count = y_types[:, i].sum()
    print(f"    [{i}] {name:<45}: {count:,} ({count/len(df)*100:.1f}%)")

# ================================================================
# 6. TRAIN/TEST SPLIT & SCALING
# ================================================================
print("\n[6/9] Train/Test split & Oversampling...")

X_train, X_test, y_train_ov, y_test_ov, y_train_types, y_test_types = train_test_split(
    X, y_overall, y_types,
    test_size=0.2, random_state=42, stratify=y_overall
)

# Oversampling untuk model overall (3 kelas)
df_train = pd.DataFrame(X_train, columns=feature_cols)
df_train['__label__'] = y_train_ov
majority_cnt = df_train['__label__'].value_counts().max()
balanced_parts = []
for cls in df_train['__label__'].unique():
    part = df_train[df_train['__label__'] == cls]
    if len(part) < majority_cnt:
        part = resample(part, replace=True, n_samples=majority_cnt, random_state=42)
    balanced_parts.append(part)
df_balanced = pd.concat(balanced_parts)
X_train_bal = df_balanced[feature_cols].values
y_train_bal  = df_balanced['__label__'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled  = scaler.transform(X_test)
X_train_types_scaled = scaler.transform(X_train[feature_cols].values)

print(f"  Train (sebelum oversample): {len(X_train):,}")
print(f"  Train (setelah oversample): {len(X_train_bal):,}")
print(f"  Test: {len(X_test):,}")

# ================================================================
# 7. TRAINING MODEL KESELURUHAN (3 KELAS)
# ================================================================
print("\n[7/9] Training model risiko keseluruhan (Random Forest)...")

rf_overall = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)
rf_overall.fit(X_train_bal, y_train_bal)

y_pred_overall = rf_overall.predict(X_test)
acc_overall = accuracy_score(y_test_ov, y_pred_overall)
f1_overall  = f1_score(y_test_ov, y_pred_overall, average='weighted')

print(f"  ✅ Akurasi: {acc_overall:.4f} ({acc_overall*100:.2f}%)")
print(f"  ✅ F1 Score: {f1_overall:.4f}")
target_names = [label_map[i] for i in sorted(label_map.keys())]
print(classification_report(y_test_ov, y_pred_overall, target_names=target_names))

# ================================================================
# 8. TRAINING MODEL MULTI-OUTPUT (6 JENIS RISIKO)
# ================================================================
print("\n[8/9] Training model multi-output (6 jenis risiko spesifik)...")

"""
Menggunakan MultiOutputClassifier dengan Random Forest sebagai base estimator.
Setiap jenis risiko mendapat model Random Forest tersendiri yang dioptimalkan.
"""

base_rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)
multi_model = MultiOutputClassifier(base_rf, n_jobs=1)
multi_model.fit(X_train_types_scaled, y_train_types)

y_pred_types = multi_model.predict(X_test_scaled)
y_proba_types = np.array([est.predict_proba(X_test_scaled) for est in multi_model.estimators_])

print("\n  Performa per jenis risiko:")
print(f"  {'Jenis Risiko':<45} {'Acc':>6} {'F1':>6} {'Support':>8}")
print(f"  {'-'*70}")
for i, name in enumerate(risk_type_names):
    acc_i = accuracy_score(y_test_types[:, i], y_pred_types[:, i])
    f1_i  = f1_score(y_test_types[:, i], y_pred_types[:, i], average='binary', zero_division=0)
    sup_i = y_test_types[:, i].sum()
    print(f"  {name:<45} {acc_i:>6.3f} {f1_i:>6.3f} {int(sup_i):>8}")

hl = hamming_loss(y_test_types, y_pred_types)
print(f"\n  Hamming Loss (semakin kecil semakin baik): {hl:.4f}")

# ================================================================
# 9. VISUALISASI & SIMPAN ARTEFAK
# ================================================================
print("\n[9/9] Menyimpan visualisasi dan model...")

# Confusion Matrix - Overall
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test_ov, y_pred_overall)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=target_names, yticklabels=target_names)
ax.set_title('Confusion Matrix – Risk Overall (RF v2)', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_v2.png', dpi=150)
plt.close()

# Feature Importance
imp_df = pd.DataFrame({
    'Fitur': feature_cols,
    'Importance': rf_overall.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=imp_df.head(15), x='Importance', y='Fitur', palette='plasma')
plt.title('Top 15 Feature Importance – MamaCare AI v2.0', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_v2.png', dpi=150)
plt.close()

# Distribusi kelas
plt.figure(figsize=(8, 5))
counts = df['RISIKO_LABEL'].value_counts()
colors = {'NORMAL': '#28a745', 'PERLU TINDAKAN': '#ffc107', 'PERLU RUJUKAN': '#dc3545'}
bars = plt.bar(counts.index, counts.values,
               color=[colors.get(k, '#888') for k in counts.index])
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=9)
plt.title('Distribusi Kelas Risiko Kehamilan (v2)', fontsize=13, fontweight='bold')
plt.xlabel('Kategori Risiko'); plt.ylabel('Jumlah Pasien')
plt.xticks(rotation=15); plt.tight_layout()
plt.savefig('distribusi_kelas_v2.png', dpi=150)
plt.close()

# Bar chart: distribusi per jenis risiko
risk_counts = y_types.sum(axis=0)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(risk_type_names, risk_counts,
               color=['#dc3545','#fd7e14','#ffc107','#6f42c1','#17a2b8','#e83e8c'])
for bar, val in zip(bars, risk_counts):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f'{int(val):,} ({val/len(df)*100:.1f}%)', va='center', fontsize=9)
ax.set_xlabel('Jumlah Pasien Berisiko')
ax.set_title('Distribusi 6 Jenis Risiko Spesifik Kehamilan', fontweight='bold')
plt.tight_layout()
plt.savefig('distribusi_jenis_risiko.png', dpi=150)
plt.close()

# Simpan model
joblib.dump(rf_overall,      'model_risk_overall.pkl')
joblib.dump(multi_model,     'model_risk_types.pkl')
joblib.dump(scaler,          'scaler_v2.pkl')
joblib.dump(feature_cols,    'feature_cols_v2.pkl')
joblib.dump(label_map,       'label_map_v2.pkl')
joblib.dump(risk_type_names, 'risk_type_names.pkl')
joblib.dump(imp_df,          'feature_importances_v2.pkl')

print("\n  ✅ Semua model v2.0 tersimpan!")
print("\n" + "=" * 70)
print("  ✅ TRAINING v2.0 SELESAI")
print("=" * 70)
print(f"  Dataset     : {len(df):,} pasien")
print(f"  Fitur       : {len(feature_cols)} fitur klinis")
print(f"  Model 1     : Risiko Keseluruhan (3 kelas) → RF Accuracy: {acc_overall:.4f}")
print(f"  Model 2     : Multi-Output (6 jenis risiko) → Hamming Loss: {hl:.4f}")
print()
print("  Jenis risiko yang terdeteksi secara otomatis:")
for i, name in enumerate(risk_type_names):
    print(f"    [{i+1}] {name}")
print()
print("  Langkah selanjutnya:")
print("    streamlit run mamacare_app_v2.py")
print("=" * 70)
