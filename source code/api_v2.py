# api_v2.py
"""
================================================================
MAMACARE AI v2.0 - FastAPI Backend
================================================================
"""

import joblib
import numpy as np
import pandas as pd
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

warnings.filterwarnings('ignore')

app = FastAPI(title="MamaCare ML API v2.0", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading MamaCare v2.0 models...")
rf_overall      = joblib.load('model_risk_overall.pkl')
multi_model     = joblib.load('model_risk_types.pkl')
scaler          = joblib.load('scaler_v2.pkl')
feature_cols    = joblib.load('feature_cols_v2.pkl')
label_map       = joblib.load('label_map_v2.pkl')
risk_type_names = joblib.load('risk_type_names.pkl')
print("All models loaded!")

TINDAKAN_MAP = {
    'Risiko Kematian Ibu': [
        'RUJUK SEGERA ke FKRTL (RS dengan layanan obstetri)',
        'Stabilisasi kondisi ibu sebelum rujukan',
        'Konsultasi SpOG dan tim multidisiplin',
        'Siapkan donor darah dan akses IV line',
    ],
    'Risiko Keguguran / Persalinan Prematur': [
        'Pemeriksaan panjang serviks (USG transvaginal)',
        'Monitoring janin ketat (NST/CTG tiap 2 minggu)',
        'Pertimbangkan terapi progesteron',
        'Kontrol gula darah ketat bila ada DM',
    ],
    'Risiko Preeklampsia / Hipertensi': [
        'Pemeriksaan protein urin',
        'Monitoring tekanan darah 2x sehari',
        'Suplementasi kalsium 1.5-2 g/hari',
        'Aspirin 75-150 mg/hari bila UK < 20 minggu',
        'Konsultasi SpOG bila TD >= 140/90 mmHg',
    ],
    'Risiko Anemia Berat': [
        'Tablet Fe 60-120 mg + Asam Folat setiap hari',
        'Pemeriksaan darah lengkap + ferritin',
        'Pertimbangkan transfusi bila Hb < 7.0 g/dL',
        'Konseling gizi tinggi zat besi',
        'Evaluasi Hb ulang setelah 4 minggu',
    ],
    'Risiko BBLR / Stunting Bayi': [
        'PMT (Pemberian Makanan Tambahan) untuk ibu KEK',
        'Edukasi gizi 1000 Hari Pertama Kehidupan',
        'Pemantauan pertumbuhan janin serial (USG biometri)',
        'Koordinasi dengan ahli gizi Puskesmas',
        'Persiapan IMD dan ASI eksklusif',
    ],
    'Risiko Infeksi Menular (Tripel Eliminasi)': [
        'HIV: Inisiasi ARV segera (program PMTCT)',
        'Sifilis: Benzathine Penicillin G 2.4 juta IU IM',
        'HepB: Siapkan HBIg + vaksin HepB untuk bayi < 12 jam',
        'Koordinasi dengan program eliminasi Dinas Kesehatan',
        'Konseling pasangan dan tes ulang',
    ],
}

REFERENSI_MAP = {
    'Risiko Kematian Ibu': 'Kemenkes RI (2022): Pedoman ANC Terpadu',
    'Risiko Keguguran / Persalinan Prematur': 'POGI (2019): Tatalaksana Persalinan Preterm',
    'Risiko Preeklampsia / Hipertensi': 'ISSHP (2018) & POGI (2019): Preeklampsia',
    'Risiko Anemia Berat': 'WHO (2016): Iron Supplementation in Pregnancy',
    'Risiko BBLR / Stunting Bayi': 'UNICEF (2021); Perpres No. 72/2021 Stunting',
    'Risiko Infeksi Menular (Tripel Eliminasi)': 'Permenkes No. 52/2017: Eliminasi HIV/Sifilis/HepB',
}

FEATURE_KEY_MAP = {
    'Usia_Ibu': 'usia_ibu',
    'Usia Kehamilan (minggu)': 'usia_kehamilan',
    'Trimester_Num': 'Trimester_Num',
    'Gravida': 'gravida',
    'Para': 'para',
    'Abortus': 'abortus',
    'IMT': 'imt',
    'LiLA (cm)': 'lila',
    'Tinggi Fundus Uteri (cm)': 'tinggi_fundus_uteri',
    'TD Sistolik': 'td_sistolik',
    'TD Diastolik': 'td_diastolik',
    'MAP': 'MAP',
    'Pulse_Pressure': 'Pulse_Pressure',
    'Hemoglobin/Hb (g/dL)': 'hemoglobin',
    'Riwayat_Enc': 'riwayat_enc',
    'Riwayat_Berat': 'riwayat_berat',
    'Riwayat_Metabolik': 'riwayat_metabolik',
    'HIV_Rek': 'hiv_rek',
    'Sif_Rek': 'sif_rek',
    'HepB_Rek': 'hepb_rek',
    'Tripel_Total': 'Tripel_Total',
    'KatIMT_Enc': 'KatIMT_Enc',
    'KatLiLA_Enc': 'KatLiLA_Enc',
    'KatTD_Enc': 'KatTD_Enc',
    'KatHb_Enc': 'KatHb_Enc',
    'Imunisasi_Enc': 'imunisasi_enc',
    'Anemia_Flag': 'Anemia_Flag',
    'Anemia_Berat': 'Anemia_Berat',
    'Hiper_Flag': 'Hiper_Flag',
    'Hiper_Berat': 'Hiper_Berat',
    'MAP_Tinggi': 'MAP_Tinggi',
    'KEK_Flag': 'KEK_Flag',
    'IMT_Kurus': 'IMT_Kurus',
    'IMT_Obese': 'IMT_Obese',
    'Usia_Risiko': 'Usia_Risiko',
    'Abortus_Tinggi': 'Abortus_Tinggi',
    'Grandemulti': 'Grandemulti',
    'Primigravida': 'Primigravida',
    'ANC_Terlambat': 'ANC_Terlambat',
    'Risk_Score': 'Risk_Score',
    'Kunjungan ANC Ke-': 'kunjungan_anc_ke',
}


def compute_features(d: dict) -> dict:
    d = d.copy()
    td_s = d.get('td_sistolik', 120)
    td_d = d.get('td_diastolik', 80)
    hb   = d.get('hemoglobin', 12)
    lila = d.get('lila', 26)
    imt  = d.get('imt', 22)
    usia = d.get('usia_ibu', 25)
    gravida  = d.get('gravida', 1)
    abortus  = d.get('abortus', 0)
    kunjungan = d.get('kunjungan_anc_ke', 4)

    d['MAP']            = (td_s + 2 * td_d) / 3
    d['Pulse_Pressure'] = td_s - td_d
    d['Trimester_Num']  = d.get('trimester_num', 2)

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
    d['ANC_Terlambat']  = 1 if kunjungan <= 2 else 0

    d['Tripel_Total'] = d.get('hiv_rek', 0) + d.get('sif_rek', 0) + d.get('hepb_rek', 0)

    re = d.get('riwayat_enc', 0)
    rb = d.get('riwayat_berat', 0)
    d['Risk_Score'] = (
        d['Anemia_Flag'] * 2 + d['Anemia_Berat'] * 3 +
        d['KatTD_Enc'] * 3  + d['Hiper_Berat'] * 5 +
        d['KEK_Flag'] * 2   + (2 if d['KatIMT_Enc'] in [0, 3] else 0) +
        re * 2 + rb * 3 + d['Tripel_Total'] + d['Usia_Risiko'] +
        d['Abortus_Tinggi'] * 2 + d['Grandemulti'] +
        d['ANC_Terlambat'] + d['MAP_Tinggi'] * 2
    )
    return d


def generate_alasan(d: dict) -> List[str]:
    alasan = []
    td_s = d.get('td_sistolik', 120); td_d = d.get('td_diastolik', 80)
    hb   = d.get('hemoglobin', 12);   lila = d.get('lila', 26)
    imt  = d.get('imt', 22);          usia = d.get('usia_ibu', 25)
    gravida = d.get('gravida', 1);    abortus = d.get('abortus', 0)
    map_v   = d.get('MAP', 90)

    if td_s >= 160 or td_d >= 110:
        alasan.append(f"TD {int(td_s)}/{int(td_d)} mmHg - Krisis Hipertensi")
    elif td_s >= 140 or td_d >= 90:
        alasan.append(f"TD {int(td_s)}/{int(td_d)} mmHg - Hipertensi Gestasional")
    elif td_s >= 120 or td_d >= 80:
        alasan.append(f"TD {int(td_s)}/{int(td_d)} mmHg - Pra-Hipertensi")

    if hb < 8.0:   alasan.append(f"Hb {hb:.1f} g/dL - Anemia Berat")
    elif hb < 10.0: alasan.append(f"Hb {hb:.1f} g/dL - Anemia Sedang")
    elif hb < 11.0: alasan.append(f"Hb {hb:.1f} g/dL - Anemia Ringan")

    if lila < 23.5: alasan.append(f"LiLA {lila:.1f} cm - KEK, risiko BBLR")
    if imt < 18.5:  alasan.append(f"IMT {imt:.1f} - Status gizi kurus")
    elif imt >= 30: alasan.append(f"IMT {imt:.1f} - Obesitas")
    if usia < 19:   alasan.append(f"Usia {usia:.0f} thn - Ibu remaja")
    elif usia > 38: alasan.append(f"Usia {usia:.0f} thn - Risiko usia lanjut")
    if abortus >= 2: alasan.append(f"Abortus {abortus}x - Risiko prematur berulang")
    if gravida >= 5: alasan.append(f"Gravida ke-{gravida} - Grandemultipara")
    if map_v > 107:  alasan.append(f"MAP {map_v:.0f} mmHg - Indikator preeklampsia")
    if d.get('hiv_rek', 0):   alasan.append("HIV Reaktif - Segera program PMTCT")
    if d.get('sif_rek', 0):   alasan.append("Sifilis Reaktif - Penicillin segera")
    if d.get('hepb_rek', 0):  alasan.append("HepB Reaktif - Siapkan HBIg untuk bayi")
    if d.get('riwayat_berat', 0): alasan.append("Riwayat penyakit berat (Hipertensi/Jantung/TB)")

    if not alasan:
        alasan.append("Tidak ditemukan faktor risiko klinis signifikan")
    return alasan


class PredictRequest(BaseModel):
    usia_ibu: float
    usia_kehamilan: int
    trimester_num: int
    gravida: int
    para: int
    abortus: int
    kunjungan_anc_ke: int
    imt: float
    lila: float
    tinggi_fundus_uteri: float
    td_sistolik: float
    td_diastolik: float
    hemoglobin: float
    imunisasi_enc: int = 3
    riwayat_enc: int = 0
    riwayat_berat: int = 0
    riwayat_metabolik: int = 0
    hiv_rek: int = 0
    sif_rek: int = 0
    hepb_rek: int = 0


@app.get("/")
async def root():
    return {"service": "MamaCare AI API", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/risk-types")
async def get_risk_types():
    return {"risk_types": [
        {"id": i, "name": n, "referensi": REFERENSI_MAP.get(n, "")}
        for i, n in enumerate(risk_type_names)
    ]}


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        input_dict = req.model_dump()
        features   = compute_features(input_dict)

        row = {}
        for col in feature_cols:
            mapped = FEATURE_KEY_MAP.get(col, col)
            row[col] = features.get(mapped, features.get(col, 0))
        X  = pd.DataFrame([row])[feature_cols].fillna(0)
        Xs = scaler.transform(X)

        pred_overall  = rf_overall.predict(X)[0]
        proba_overall = rf_overall.predict_proba(X)[0].tolist()
        label_overall = label_map[pred_overall]

        pred_types = multi_model.predict(Xs)[0]
        proba_types = []
        for est in multi_model.estimators_:
            prob = est.predict_proba(Xs)[0]
            proba_types.append(float(prob[1]) if len(prob) > 1 else 0.0)

        risk_details = []
        for i, name in enumerate(risk_type_names):
            risk_details.append({
                "name"       : name,
                "detected"   : bool(pred_types[i]),
                "probability": round(proba_types[i], 4),
                "tindakan"   : TINDAKAN_MAP.get(name, []) if pred_types[i] else [],
                "referensi"  : REFERENSI_MAP.get(name, ""),
            })

        # ── Override label overall dari jenis risiko (logika revisi dosen) ──
        # 5 risiko berat [0,1,2,3,5]  → PERLU RUJUKAN  (2)
        # Hanya stunting [4]           → PERLU TINDAKAN (1)
        # Tidak ada                    → NORMAL          (0)
        rujukan_flags = [pred_types[0], pred_types[1], pred_types[2], pred_types[3], pred_types[5]]
        if any(rujukan_flags):
            final_overall = 2
        elif pred_types[4]:
            final_overall = 1
        else:
            final_overall = 0
        final_label = label_map[final_overall]

        if final_label == 'PERLU RUJUKAN':
            rek = ("RUJUK SEGERA ke FKRTL. Ibu terdeteksi risiko berat: "
                   "Kematian Ibu / Keguguran / Preeklampsia / Anemia Berat / Infeksi Menular. "
                   "Stabilisasi kondisi dan hubungi RS tujuan sebelum merujuk.")
        elif final_label == 'PERLU TINDAKAN':
            rek = ("TINDAKAN DI PUSKESMAS. Ibu berisiko BBLR/Stunting Bayi. "
                   "Berikan PMT, suplementasi gizi, dan monitoring ketat. "
                   "Jadwalkan ANC tiap 2 minggu.")
        else:
            rek = ("NORMAL. Lanjutkan ANC rutin minimal 6 kali selama kehamilan. "
                   "Konsumsi tablet Fe 60 mg/hari. Edukasi tanda bahaya kehamilan.")

        return {
            "overall_prediction"   : final_overall,
            "overall_label"        : final_label,
            "overall_probabilities": proba_overall,
            "risk_score"           : float(features.get('Risk_Score', 0)),
            "risk_types"           : risk_details,
            "active_risk_count"    : int(sum(pred_types)),
            "alasan_klinis"        : generate_alasan(features),
            "rekomendasi_utama"    : rek,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/overall")
async def predict_overall_only(req: PredictRequest):
    try:
        features = compute_features(req.model_dump())
        row = {}
        for col in feature_cols:
            mapped = FEATURE_KEY_MAP.get(col, col)
            row[col] = features.get(mapped, features.get(col, 0))
        X = pd.DataFrame([row])[feature_cols].fillna(0)
        pred  = rf_overall.predict(X)[0]
        proba = rf_overall.predict_proba(X)[0].tolist()
        return {
            "prediction"   : int(pred),
            "label"        : label_map[pred],
            "probabilities": proba,
            "risk_score"   : float(features.get('Risk_Score', 0)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
