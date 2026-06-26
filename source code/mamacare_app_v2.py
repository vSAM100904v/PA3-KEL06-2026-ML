# -*- coding: utf-8 -*-
"""
================================================================
MAMACARE AI v2.0 - STREAMLIT APP
================================================================
Sistem Prediksi Risiko Kehamilan Multi-Dimensi
Revisi berdasarkan masukan dosen penguji:
  - Sistem kini mendeteksi JENIS RISIKO SPESIFIK
  - Memberikan penjelasan klinis per risiko
  - Tidak bergantung pada dokter untuk menentukan jenis risiko

Cara menjalankan:
    streamlit run mamacare_app_v2.py
================================================================
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MamaCare AI v2 – Prediksi Risiko Multi-Dimensi",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
    padding: 24px; border-radius: 14px; color: white;
    text-align: center; margin-bottom: 22px;
}
.risk-card {
    padding: 16px; border-radius: 10px; margin: 8px 0;
    border-left: 5px solid;
}
.risk-high  { background: #fff5f5; border-color: #dc3545; }
.risk-med   { background: #fffbf0; border-color: #ffc107; }
.risk-low   { background: #f0fff4; border-color: #28a745; }
.risk-type-active   { background: #fee2e2; border-radius: 8px; padding: 12px; margin: 6px 0; border-left: 4px solid #dc2626; }
.risk-type-inactive { background: #f0fdf4; border-radius: 8px; padding: 12px; margin: 6px 0; border-left: 4px solid #16a34a; }
.clinical-note { background: #eff6ff; border-radius: 8px; padding: 14px; margin: 8px 0; border-left: 4px solid #2563eb; font-size: 0.92em; }
.metric-badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.85em; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD MODEL
# ================================================================
@st.cache_resource
def load_models():
    m = {}
    try:
        m['rf_overall']    = joblib.load('model_risk_overall.pkl')
        m['multi_model']   = joblib.load('model_risk_types.pkl')
        m['scaler']        = joblib.load('scaler_v2.pkl')
        m['feature_cols']  = joblib.load('feature_cols_v2.pkl')
        m['label_map']     = joblib.load('label_map_v2.pkl')
        m['risk_type_names'] = joblib.load('risk_type_names.pkl')
        try:
            m['importances'] = joblib.load('feature_importances_v2.pkl')
        except Exception:
            m['importances'] = None
        m['status'] = 'loaded'
    except FileNotFoundError as e:
        m['status'] = f'not_found: {e}'
    return m

models = load_models()

# ================================================================
# FUNGSI KOMPUTASI FITUR
# ================================================================
def compute_features(d):
    d = dict(d)
    td_s  = float(d.get('TD Sistolik', 120))
    td_d  = float(d.get('TD Diastolik', 80))
    hb    = float(d.get('Hemoglobin/Hb (g/dL)', 12))
    lila  = float(d.get('LiLA (cm)', 26))
    imt   = float(d.get('IMT', 22))
    usia  = float(d.get('Usia_Ibu', 25))
    gravida = int(d.get('Gravida', 1))
    abortus = int(d.get('Abortus', 0))
    uk    = int(d.get('Usia Kehamilan (minggu)', 20))

    d['MAP']           = (td_s + 2 * td_d) / 3
    d['Pulse_Pressure'] = td_s - td_d

    d['KatTD_Enc']   = 2 if (td_s >= 140 or td_d >= 90) else (1 if (td_s >= 120 or td_d >= 80) else 0)
    d['KatHb_Enc']   = 1 if hb < 11.0 else 0
    d['KatLiLA_Enc'] = 1 if lila < 23.5 else 0
    d['KatIMT_Enc']  = 0 if imt < 18.5 else (3 if imt >= 30 else (2 if imt >= 25 else 1))

    d['Anemia_Flag']    = 1 if hb < 11.0 else 0
    d['Anemia_Berat']   = 1 if hb < 8.0 else 0
    d['Hiper_Flag']     = 1 if td_s >= 140 else 0
    d['Hiper_Berat']    = 1 if td_s >= 160 else 0
    d['MAP_Tinggi']     = 1 if d['MAP'] > 107 else 0
    d['KEK_Flag']       = 1 if lila < 23.5 else 0
    d['IMT_Kurus']      = 1 if imt < 18.5 else 0
    d['IMT_Obese']      = 1 if imt >= 30.0 else 0
    d['Usia_Risiko']    = 1 if (usia < 19 or usia > 38) else 0
    d['Abortus_Tinggi'] = 1 if abortus >= 2 else 0
    d['Grandemulti']    = 1 if gravida >= 5 else 0
    d['Primigravida']   = 1 if gravida == 1 else 0
    d['ANC_Terlambat']  = 1 if int(d.get('Kunjungan ANC Ke-', 4)) <= 2 else 0

    hiv  = int(d.get('HIV_Rek', 0))
    sif  = int(d.get('Sif_Rek', 0))
    hepb = int(d.get('HepB_Rek', 0))
    d['Tripel_Total'] = hiv + sif + hepb

    re  = int(d.get('Riwayat_Enc', 0))
    rb  = int(d.get('Riwayat_Berat', 0))
    rm  = int(d.get('Riwayat_Metabolik', 0))
    d['Risk_Score'] = (
        d['Anemia_Flag'] * 2 + d['Anemia_Berat'] * 3 +
        d['KatTD_Enc'] * 3  + d['Hiper_Berat'] * 5 +
        d['KEK_Flag'] * 2   + (2 if d['KatIMT_Enc'] in [0, 3] else 0) +
        re * 2 + rb * 3 + d['Tripel_Total'] + d['Usia_Risiko'] +
        d['Abortus_Tinggi'] * 2 + d['Grandemulti'] +
        d['ANC_Terlambat'] + d['MAP_Tinggi'] * 2
    )
    return d

def predict_all(data_dict):
    fc = models['feature_cols']
    d  = compute_features(data_dict)
    row = {col: d.get(col, 0) for col in fc}
    X  = pd.DataFrame([row])[fc].fillna(0)
    Xs = models['scaler'].transform(X)

    # Prediksi risiko keseluruhan
    pred_overall = models['rf_overall'].predict(X)[0]
    proba_overall = models['rf_overall'].predict_proba(X)[0]
    label_overall = models['label_map'][pred_overall]

    # Prediksi 6 jenis risiko (probability)
    pred_types = models['multi_model'].predict(Xs)[0]
    proba_types = []
    for est in models['multi_model'].estimators_:
        prob = est.predict_proba(Xs)[0]
        # prob[1] = probabilitas risiko=1
        if len(prob) > 1:
            proba_types.append(float(prob[1]))
        else:
            proba_types.append(0.0)

    # ── Logika revisi dosen: override label overall dari jenis risiko ──
    # 5 risiko berat  → PERLU RUJUKAN (2)
    # Hanya stunting  → PERLU TINDAKAN (1)
    # Tidak ada       → NORMAL (0)
    rujukan_flags = [pred_types[0], pred_types[1], pred_types[2], pred_types[3], pred_types[5]]
    if any(rujukan_flags):
        final_overall = 2
    elif pred_types[4]:
        final_overall = 1
    else:
        final_overall = 0
    final_label = models['label_map'][final_overall]

    return {
        'overall_pred'   : final_overall,
        'overall_label'  : final_label,
        'overall_proba'  : proba_overall.tolist(),
        'types_pred'     : pred_types.tolist(),
        'types_proba'    : proba_types,
        'risk_score'     : float(d.get('Risk_Score', 0)),
        'features'       : d,
    }

# Penjelasan klinis per jenis risiko
CLINICAL_EXPLANATIONS = {
    'Risiko Kematian Ibu': {
        'icon': '💀',
        'color': '#dc2626',
        'alasan_umum': [
            'Tekanan darah sistolik ≥ 160 mmHg (krisis hipertensi)',
            'Hemoglobin < 8.0 g/dL (anemia berat → risiko perdarahan fatal)',
            'Riwayat penyakit jantung / TB aktif disertai anemia',
            'Riwayat abortus berulang (≥ 3 kali)',
            'HIV reaktif disertai anemia berat',
        ],
        'tindakan': [
            '🚨 RUJUK SEGERA ke FKRTL (RS dengan layanan obstetri)',
            'Stabilisasi kondisi ibu sebelum rujukan',
            'Konsultasi SpOG dan tim multidisiplin',
            'Siapkan donor darah dan akses IV line',
        ],
        'referensi': 'Kemenkes RI (2022): Tiga terlambat dan empat terlalu sebagai faktor kematian ibu',
    },
    'Risiko Keguguran / Persalinan Prematur': {
        'icon': '🫄',
        'color': '#ea580c',
        'alasan_umum': [
            'Riwayat abortus ≥ 2 kali (incompetent cervix / faktor hormonal)',
            'Grandemultipara (gravida ≥ 5) dengan usia kehamilan muda',
            'Tekanan darah tinggi pada usia kehamilan < 34 minggu',
            'Anemia berat (Hb < 9.0 g/dL) → hipoksia janin',
            'Ibu remaja (< 18 tahun) primigravida → maturitas uterus belum optimal',
            'Riwayat diabetes mellitus → instabilitas glukosa plasenta',
        ],
        'tindakan': [
            'Pemeriksaan panjang serviks (USG transvaginal)',
            'Monitoring janin lebih ketat (NST/CTG tiap 2 minggu)',
            'Terapi progesteron bila ada riwayat prematur',
            'Konseling istirahat & pembatasan aktivitas fisik berat',
            'Kontrol gula darah ketat bila DM',
        ],
        'referensi': 'POGI (2019): Panduan Pengelolaan Persalinan Preterm',
    },
    'Risiko Preeklampsia / Hipertensi': {
        'icon': '🩸',
        'color': '#d97706',
        'alasan_umum': [
            'Tekanan darah ≥ 140/90 mmHg (kriteria hipertensi gestasional)',
            'MAP (Mean Arterial Pressure) > 107 mmHg',
            'Riwayat hipertensi kronik sebelum kehamilan',
            'Obesitas (IMT ≥ 30) + tekanan darah pra-hipertensi',
            'Primigravida usia > 35 tahun (faktor risiko independen)',
        ],
        'tindakan': [
            'Pemeriksaan protein urin (dipstick / lab)',
            'Monitoring tekanan darah 2× sehari',
            'Suplementasi kalsium 1.5–2 g/hari (terbukti mengurangi risiko PE)',
            'Aspirin dosis rendah 75–150 mg/hari (bila UK < 20 minggu, risiko tinggi)',
            'Konsultasi SpOG bila TD ≥ 140/90 mmHg',
            'Waspada gejala: sakit kepala, penglihatan kabur, nyeri epigastrik',
        ],
        'referensi': 'ISSHP (2018) & POGI (2019): Kriteria dan Tata Laksana Preeklampsia',
    },
    'Risiko Anemia Berat': {
        'icon': '🩺',
        'color': '#7c3aed',
        'alasan_umum': [
            'Hemoglobin < 8.0 g/dL (anemia berat WHO)',
            'Hb < 9.5 g/dL disertai KEK (LiLA < 23.5 cm) — double burden',
            'Hb < 10.0 g/dL disertai IMT kurus (< 18.5) — triple burden',
            'Hb < 11.0 g/dL dengan riwayat anemia sebelumnya',
        ],
        'tindakan': [
            'Pemberian tablet Fe 60–120 mg + Asam Folat setiap hari',
            'Pemeriksaan darah lengkap + ferritin + TIBC',
            'Pertimbangkan transfusi bila Hb < 7.0 g/dL atau simptomatis berat',
            'Konseling gizi: makanan tinggi zat besi (hati, bayam, kacang-kacangan)',
            'Vitamin C untuk meningkatkan absorpsi Fe',
            'Evaluasi Hb ulang setelah 4 minggu terapi',
        ],
        'referensi': 'WHO (2016): Guideline on Iron Supplementation in Pregnancy; Kemenkes (2022)',
    },
    'Risiko BBLR / Stunting Bayi': {
        'icon': '👶',
        'color': '#0891b2',
        'alasan_umum': [
            'LiLA < 23.5 cm → Kurang Energi Kronik (KEK) → BBLR → stunting',
            'IMT < 18.5 → status gizi buruk ibu → pembatasan pertumbuhan janin',
            'Anemia + border KEK → suplai nutrisi janin tidak optimal',
            'Ibu remaja (< 19 tahun) → kompetisi nutrisi ibu-janin',
            'Riwayat DM + obesitas → makrosomia ATAU BBLR (paradoks metabolik)',
        ],
        'tindakan': [
            'PMT (Pemberian Makanan Tambahan) untuk ibu KEK',
            'Edukasi gizi 1000 HPK (1000 Hari Pertama Kehidupan)',
            'Pemantauan pertumbuhan janin serial (USG biometri)',
            'Suplementasi protein, zinc, omega-3',
            'Koordinasi dengan ahli gizi Puskesmas',
            'Persiapan IMD (Inisiasi Menyusu Dini) dan ASI eksklusif',
        ],
        'referensi': 'UNICEF & WHO (2021): Global Nutrition Targets; Perpres No. 72/2021 tentang Stunting',
    },
    'Risiko Infeksi Menular (Tripel Eliminasi)': {
        'icon': '🦠',
        'color': '#db2777',
        'alasan_umum': [
            'HIV Reaktif → risiko penularan vertikal ibu-bayi (tanpa ARV: 15–45%)',
            'Sifilis Reaktif → kongenital sifilis, lahir mati, BBLR',
            'Hepatitis B Reaktif → penularan perinatal ke bayi (risiko 70–90%)',
            'Riwayat Hepatitis B sebelumnya',
        ],
        'tindakan': [
            '🔴 HIV: Inisiasi ARV segera (program PMTCT), persalinan SC elektif',
            '🔴 Sifilis: Benzathine Penicillin G 2.4 juta IU IM satu dosis',
            '🔴 HepB: Imunisasi HBIg + vaksin HepB pada bayi < 12 jam setelah lahir',
            'Koordinasi dengan program eliminasi Dinas Kesehatan',
            'Konseling pasangan dan tes ulang',
            'Notifikasi kasus ke petugas surveilans',
        ],
        'referensi': 'Permenkes No. 52/2017: Eliminasi Penularan HIV, Sifilis, dan Hepatitis B',
    },
}

def generate_alasan_klinis(d, types_pred, types_proba):
    """Menghasilkan penjelasan klinis spesifik berdasarkan nilai input pasien."""
    alasan = []

    td_s  = float(d.get('TD Sistolik', 120))
    td_d  = float(d.get('TD Diastolik', 80))
    hb    = float(d.get('Hemoglobin/Hb (g/dL)', 12))
    lila  = float(d.get('LiLA (cm)', 26))
    imt   = float(d.get('IMT', 22))
    usia  = float(d.get('Usia_Ibu', 25))
    gravida = int(d.get('Gravida', 1))
    abortus = int(d.get('Abortus', 0))
    map_v = float(d.get('MAP', 90))

    if td_s >= 160 or td_d >= 110:
        alasan.append(f"⚠️ Tekanan darah {int(td_s)}/{int(td_d)} mmHg → KRISIS HIPERTENSI, risiko eklampsia")
    elif td_s >= 140 or td_d >= 90:
        alasan.append(f"⚠️ Tekanan darah {int(td_s)}/{int(td_d)} mmHg → memenuhi kriteria hipertensi gestasional")
    elif td_s >= 120 or td_d >= 80:
        alasan.append(f"📊 Tekanan darah {int(td_s)}/{int(td_d)} mmHg → pra-hipertensi, perlu monitoring")

    if hb < 8.0:
        alasan.append(f"🩸 Hemoglobin {hb:.1f} g/dL → ANEMIA BERAT, risiko perdarahan dan kematian ibu")
    elif hb < 10.0:
        alasan.append(f"🩸 Hemoglobin {hb:.1f} g/dL → anemia sedang, perlu suplementasi intensif")
    elif hb < 11.0:
        alasan.append(f"🩸 Hemoglobin {hb:.1f} g/dL → anemia ringan (batas WHO: 11 g/dL)")

    if lila < 23.5:
        alasan.append(f"📏 LiLA {lila:.1f} cm → Kurang Energi Kronik (KEK), risiko BBLR dan stunting bayi")

    if imt < 18.5:
        alasan.append(f"⚖️ IMT {imt:.1f} → status gizi kurus, asupan nutrisi janin tidak optimal")
    elif imt >= 30:
        alasan.append(f"⚖️ IMT {imt:.1f} → obesitas, meningkatkan risiko preeklampsia dan DM gestasional")

    if usia < 19:
        alasan.append(f"👤 Usia ibu {usia:.0f} tahun → remaja, organ reproduksi belum matang sempurna")
    elif usia > 38:
        alasan.append(f"👤 Usia ibu {usia:.0f} tahun → risiko komplikasi meningkat setelah usia 35 tahun")

    if abortus >= 2:
        alasan.append(f"📋 Riwayat abortus {abortus}× → risiko persalinan prematur dan keguguran berulang")

    if gravida >= 5:
        alasan.append(f"📋 Gravida ke-{gravida} → grandemultipara, risiko atonia uteri dan plasenta previa")

    if map_v > 107:
        alasan.append(f"💊 MAP {map_v:.0f} mmHg → di atas normal (>107), indikator risiko preeklampsia")

    if d.get('HIV_Rek', 0) == 1:
        alasan.append("🦠 HIV REAKTIF → segera masuk program PMTCT, ARV wajib dimulai")
    if d.get('Sif_Rek', 0) == 1:
        alasan.append("🦠 SIFILIS REAKTIF → Penicillin segera, risiko sifilis kongenital pada bayi")
    if d.get('HepB_Rek', 0) == 1:
        alasan.append("🦠 HEPATITIS B REAKTIF → persiapkan HBIg untuk bayi segera setelah lahir")
    if d.get('Riwayat_Berat', 0) == 1:
        riwayat_txt = d.get('Riwayat_Text', 'penyakit berat')
        alasan.append(f"⚕️ Riwayat {riwayat_txt} → faktor risiko komplikasi kehamilan serius")

    if not alasan:
        alasan.append("✅ Tidak ditemukan faktor risiko klinis signifikan berdasarkan data yang diinput")

    return alasan

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("## 🤰 MamaCare AI v2.0")
    if models['status'] == 'loaded':
        st.success("✅ Model v2.0 aktif")
        st.caption("6 jenis risiko terdeteksi otomatis")
    else:
        st.error(f"❌ Model tidak ditemukan\nJalankan: python mamacare_train_v2.py")

    st.markdown("---")
    page = st.radio("Navigasi", [
        "🔍 Prediksi Risiko",
        "📊 Dashboard Analisis",
        "ℹ️ Tentang Sistem",
    ])

    st.markdown("---")
    st.markdown("**6 Jenis Risiko yang Dideteksi:**")
    risk_icons = ['💀','🫄','🩸','🩺','👶','🦠']
    risk_short  = ['Kematian Ibu','Keguguran/Prematur','Preeklampsia',
                   'Anemia Berat','BBLR/Stunting','Infeksi Menular']
    for icon, name in zip(risk_icons, risk_short):
        st.caption(f"{icon} {name}")

# ================================================================
# HALAMAN PREDIKSI
# ================================================================
if page == "🔍 Prediksi Risiko":
    st.markdown("""
    <div class="main-header">
        <h2>🤰 MamaCare AI v2.0</h2>
        <p>Sistem Prediksi Risiko Kehamilan Multi-Dimensi</p>
        <p>Mendeteksi 6 jenis risiko spesifik secara otomatis</p>
    </div>
    """, unsafe_allow_html=True)

    if models['status'] != 'loaded':
        st.error("Model belum dimuat. Jalankan: `python mamacare_train_v2.py`")
        st.stop()

    st.markdown("### 📋 Data Pemeriksaan ANC")
    st.caption("Isi data hasil pemeriksaan ibu hamil. Sistem akan mengidentifikasi jenis risiko secara otomatis.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data Ibu**")
        usia_ibu     = st.number_input("Usia Ibu (tahun)", 14, 55, 28)
        uk           = st.number_input("Usia Kehamilan (minggu)", 4, 42, 20)
        trimester    = st.selectbox("Trimester", ['I','II','III'])
        gravida      = st.number_input("Gravida", 1, 15, 1)
        para         = st.number_input("Para", 0, 14, 0)
        abortus      = st.number_input("Abortus", 0, 10, 0)
        kunjungan    = st.number_input("Kunjungan ANC Ke-", 1, 10, 1)

    with col2:
        st.markdown("**Antropometri & Vital Sign**")
        bb       = st.number_input("Berat Badan (kg)", 30.0, 120.0, 55.0)
        tb       = st.number_input("Tinggi Badan (cm)", 140.0, 180.0, 155.0)
        imt      = round(bb / (tb / 100) ** 2, 2)
        st.info(f"IMT dihitung otomatis: **{imt:.2f}**")
        lila     = st.number_input("LiLA (cm)", 15.0, 40.0, 26.0)
        tfu      = st.number_input("Tinggi Fundus Uteri (cm)", 5.0, 40.0, 20.0)
        td_s     = st.number_input("TD Sistolik (mmHg)", 80, 200, 120)
        td_d     = st.number_input("TD Diastolik (mmHg)", 50, 140, 80)
        hb       = st.number_input("Hemoglobin Hb (g/dL)", 4.0, 18.0, 11.5)

    with col3:
        st.markdown("**Riwayat & Tripel Eliminasi**")
        riwayat_options = ['Tidak Ada','Hipertensi','Anemia','TB','Hepatitis B',
                           'Diabetes','Jantung','Asma']
        riwayat  = st.selectbox("Riwayat Penyakit", riwayat_options)
        hiv_rek  = st.selectbox("Tripel – HIV", ['Non Reaktif','Reaktif'])
        sif_rek  = st.selectbox("Tripel – Sifilis", ['Non Reaktif','Reaktif'])
        hepb_rek = st.selectbox("Tripel – Hepatitis B", ['Non Reaktif','Reaktif'])
        imunisasi = st.selectbox("Status Imunisasi TD", ['T1','T2','T3','T4','T5'])

    st.markdown("---")
    predict_btn = st.button("🔍 ANALISIS RISIKO SEKARANG", use_container_width=True)

    if predict_btn:
        trimester_map = {'I': 1, 'II': 2, 'III': 3}
        imunisasi_map = {'T1':1,'T2':2,'T3':3,'T4':4,'T5':5}
        riwayat_berat_set = {'hipertensi','jantung','tb'}
        riwayat_metabolik_set = {'diabetes','hepatitis b'}

        input_data = {
            'Usia_Ibu'                  : usia_ibu,
            'Usia Kehamilan (minggu)'   : uk,
            'Trimester_Num'             : trimester_map[trimester],
            'Gravida'                   : gravida,
            'Para'                      : para,
            'Abortus'                   : abortus,
            'IMT'                       : imt,
            'LiLA (cm)'                 : lila,
            'Tinggi Fundus Uteri (cm)'  : tfu,
            'TD Sistolik'               : td_s,
            'TD Diastolik'              : td_d,
            'Hemoglobin/Hb (g/dL)'     : hb,
            'Kunjungan ANC Ke-'         : kunjungan,
            'Imunisasi_Enc'             : imunisasi_map[imunisasi],
            'Riwayat_Enc'               : 0 if riwayat == 'Tidak Ada' else 1,
            'Riwayat_Berat'             : 1 if riwayat.lower() in riwayat_berat_set else 0,
            'Riwayat_Metabolik'         : 1 if riwayat.lower() in riwayat_metabolik_set else 0,
            'Riwayat_Text'              : riwayat,
            'HIV_Rek'                   : 1 if hiv_rek == 'Reaktif' else 0,
            'Sif_Rek'                   : 1 if sif_rek == 'Reaktif' else 0,
            'HepB_Rek'                  : 1 if hepb_rek == 'Reaktif' else 0,
        }

        with st.spinner("Menganalisis risiko..."):
            result = predict_all(input_data)

        # ── Hasil Utama ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis Risiko")

        overall = result['overall_label']
        proba   = result['overall_proba']
        score   = result['risk_score']

        color_map = {'NORMAL': 'risk-low', 'PERLU TINDAKAN': 'risk-med', 'PERLU RUJUKAN': 'risk-high'}
        emoji_map = {'NORMAL': '✅', 'PERLU TINDAKAN': '⚠️', 'PERLU RUJUKAN': '🚨'}

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Status Risiko Keseluruhan", f"{emoji_map[overall]} {overall}")
        with c2:
            st.metric("Risk Score Komposit", f"{score:.0f} poin")
        with c3:
            active_risks = sum(result['types_pred'])
            st.metric("Jenis Risiko Terdeteksi", f"{active_risks} dari 6")

        # ── 6 Jenis Risiko ──────────────────────────────────────────
        st.markdown("### 🔬 Breakdown 6 Jenis Risiko Spesifik")
        st.caption("Sistem mengidentifikasi setiap jenis risiko secara independen berdasarkan panduan klinis Kemenkes RI & WHO")

        risk_names = models['risk_type_names']
        types_pred  = result['types_pred']
        types_proba = result['types_proba']

        for i, (name, pred, proba_i) in enumerate(zip(risk_names, types_pred, types_proba)):
            exp = CLINICAL_EXPLANATIONS.get(name, {})
            icon = exp.get('icon', '❓')

            if pred == 1:
                css_class = 'risk-type-active'
                status_txt = f"**🔴 TERDETEKSI** (confidence: {proba_i*100:.1f}%)"
            else:
                css_class = 'risk-type-inactive'
                status_txt = f"**✅ Tidak Terdeteksi** (confidence: {(1-proba_i)*100:.1f}%)"

            with st.expander(f"{icon} {name}  —  {status_txt}", expanded=(pred == 1)):
                if pred == 1:
                    st.markdown("**📌 Kemungkinan Penyebab pada Pasien Ini:**")
                    tindakan_list = exp.get('tindakan', [])
                    for t in tindakan_list:
                        st.markdown(f"- {t}")
                    st.markdown(f"""
                    <div class="clinical-note">
                    📚 <b>Referensi Klinis:</b> {exp.get('referensi', 'Panduan Kemenkes RI')}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("Tidak ditemukan indikator risiko signifikan untuk kategori ini berdasarkan data input.")

        # ── Alasan Klinis Spesifik ───────────────────────────────────
        st.markdown("### 📝 Alasan Klinis Spesifik (Berdasarkan Data Pasien)")
        alasan_list = generate_alasan_klinis(result['features'], types_pred, types_proba)
        for a in alasan_list:
            st.markdown(f"- {a}")

        # ── Rekomendasi Tindakan ──────────────────────────────────────
        st.markdown("### 💊 Rekomendasi Tindakan")
        if overall == 'PERLU RUJUKAN':
            st.error("""
            🚨 **PERLU RUJUKAN — RUJUK KE FKRTL**

            Ibu terdeteksi mengalami satu atau lebih risiko berat:
            Risiko Kematian Ibu / Keguguran & Prematur / Preeklampsia / Anemia Berat / Infeksi Menular.

            **Tindakan segera:**
            - Stabilisasi kondisi ibu (pasang IV line jika perlu)
            - Hubungi RS / FKRTL tujuan sebelum merujuk
            - Dampingi oleh bidan atau dokter selama perjalanan
            - Informasikan keluarga & siapkan donor darah
            - Buat surat rujukan lengkap dengan hasil pemeriksaan
            """)
        elif overall == 'PERLU TINDAKAN':
            st.warning("""
            ⚠️ **PERLU TINDAKAN — INTERVENSI DI PUSKESMAS / BIDAN**

            Ibu terdeteksi berisiko BBLR / Stunting Bayi (tanpa risiko berat lain).
            Penanganan dapat dilakukan di fasilitas kesehatan primer.

            **Tindakan:**
            - PMT (Pemberian Makanan Tambahan) untuk ibu dengan KEK atau IMT kurus
            - Suplementasi: tablet Fe, asam folat, protein, zinc
            - Edukasi gizi 1000 Hari Pertama Kehidupan (1000 HPK)
            - Monitoring pertumbuhan janin lebih ketat (USG biometri)
            - Jadwalkan kunjungan ANC tiap 2 minggu
            - Koordinasi dengan ahli gizi Puskesmas
            - Dokumentasikan dalam Buku KIA
            """)
        else:
            st.success("""
            ✅ **NORMAL — LANJUTKAN ANC RUTIN**

            Tidak ditemukan risiko klinis signifikan saat ini.

            **Tetap lakukan:**
            - Kunjungan ANC sesuai jadwal Kemenkes (min. 6 kali selama kehamilan)
            - Konsumsi tablet Fe 60 mg/hari minimal 90 tablet selama kehamilan
            - Edukasi tanda bahaya kehamilan dan persiapan persalinan
            - Pemeriksaan tripel eliminasi jika belum dilakukan
            """)

        st.markdown("---")
        st.caption("⚠️ Sistem ini adalah alat bantu skrining. Keputusan klinis tetap menjadi wewenang tenaga kesehatan berlisensi.")

# ================================================================
# HALAMAN DASHBOARD
# ================================================================
elif page == "📊 Dashboard Analisis":
    st.markdown("## 📊 Dashboard Analisis Dataset")
    try:
        df = pd.read_excel('/mnt/user-data/uploads/dataset_anc_bumil.xlsx')
        df['Tgl Lahir'] = pd.to_datetime(df['Tgl Lahir'], dayfirst=True, errors='coerce')
        df['Usia_Ibu'] = df['Tgl Lahir'].apply(
            lambda x: (pd.Timestamp('2025-10-01') - x).days // 365 if pd.notnull(x) else 28).clip(14, 55)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pasien", f"{len(df):,}")
        c2.metric("Rata-rata Usia Ibu", f"{df['Usia_Ibu'].mean():.1f} thn")
        c3.metric("Rata-rata Hb", f"{df['Hemoglobin/Hb (g/dL)'].mean():.1f} g/dL")
        c4.metric("Pasien Berisiko KEK", f"{(df['Kategori LiLA']=='Risiko KEK').sum():,}")

        try:
            img = __import__('PIL.Image', fromlist=['Image'])
            import matplotlib.pyplot as plt
        except Exception:
            import matplotlib.pyplot as plt

        st.markdown("#### Distribusi Kelas Risiko")
        try:
            st.image('distribusi_kelas_v2.png')
        except Exception:
            st.info("Jalankan training v2 untuk melihat visualisasi.")

        st.markdown("#### Distribusi 6 Jenis Risiko Spesifik")
        try:
            st.image('distribusi_jenis_risiko.png')
        except Exception:
            st.info("Visualisasi belum tersedia.")

        if models.get('importances') is not None:
            st.markdown("#### Feature Importance")
            try:
                st.image('feature_importance_v2.png')
            except Exception:
                pass

        st.markdown("#### Statistik Deskriptif")
        cols_show = ['Usia_Ibu','Usia Kehamilan (minggu)','IMT','LiLA (cm)',
                     'TD Sistolik','TD Diastolik','Hemoglobin/Hb (g/dL)']
        st.dataframe(df[[c for c in cols_show if c in df.columns]].describe().round(2),
                     use_container_width=True)
    except Exception as e:
        st.error(f"Dataset tidak ditemukan: {e}")

# ================================================================
# HALAMAN TENTANG
# ================================================================
elif page == "ℹ️ Tentang Sistem":
    st.markdown("## ℹ️ Tentang MamaCare AI v2.0")
    st.markdown("""
    <div class="main-header">
        <h2>🤰 MamaCare AI v2.0</h2>
        <p>Sistem Prediksi Risiko Kehamilan Multi-Dimensi</p>
        <p>Institut Teknologi Del | Mata Kuliah 4143104</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🆕 Perubahan v2.0 (Revisi Dosen)
| Aspek | v1.0 (Lama) | v2.0 (Baru) |
|---|---|---|
| Output | 1 label (3 kelas) | 1 + 6 jenis risiko |
| Jenis Risiko | Tidak ada | 6 jenis spesifik |
| Penjelasan | Minimal | Klinis berbasis referensi |
| Peran Dokter | Menentukan jenis risiko | Memvalidasi & tindak lanjut |
| Model | 1 RF classifier | RF + MultiOutput RF |
| Referensi | Tidak disebut | Kemenkes, WHO, POGI, ISSHP |
""")
    with c2:
        st.markdown("""
### 🔬 6 Jenis Risiko yang Dideteksi
| # | Jenis Risiko | Referensi |
|---|---|---|
| 1 | 💀 Kematian Ibu | Kemenkes 2022 |
| 2 | 🫄 Keguguran/Prematur | POGI 2019 |
| 3 | 🩸 Preeklampsia/Hipertensi | ISSHP 2018 |
| 4 | 🩺 Anemia Berat | WHO 2016 |
| 5 | 👶 BBLR/Stunting Bayi | UNICEF 2021 |
| 6 | 🦠 Infeksi Menular | Permenkes 52/2017 |
""")

    st.markdown("---")
    st.markdown("""
### 📚 Referensi Klinis
1. **Kemenkes RI** (2022). Pedoman ANC Terpadu. Jakarta.
2. **WHO** (2016). WHO Recommendations on Antenatal Care for a Positive Pregnancy Experience. Geneva.
3. **POGI** (2019). Pedoman Nasional Pelayanan Kedokteran: Diagnosis dan Tatalaksana Preeklampsia.
4. **ISSHP** (2018). The classification, diagnosis and management of the hypertensive disorders of pregnancy.
5. **Permenkes No. 52 Tahun 2017** tentang Eliminasi Penularan HIV, Sifilis, dan Hepatitis B.
6. **Perpres No. 72 Tahun 2021** tentang Percepatan Penurunan Stunting.
7. **Buku KIA** (2023). Kemenkes RI.
""")

    st.warning("""
**⚠️ Disclaimer:**
MamaCare AI adalah alat bantu skrining berbasis AI dan **tidak dapat menggantikan**
pemeriksaan & diagnosa oleh tenaga kesehatan profesional.
Untuk kondisi darurat, segera hubungi **119 ext 9** (Hotline Kemenkes RI).
""")
