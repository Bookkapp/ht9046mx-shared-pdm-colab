# HT-9046MX Adaptive Scoring and Calibration System

## คำตอบสั้น ๆ: ต้องเริ่มบน Colab หรือไม่

ต้องรัน Google Colab **หนึ่งครั้งต่อ model version** เพื่อเทรน Shared LSTM Autoencoder และสร้าง artifact เริ่มต้น เนื่องจาก Automatic Scoring บน Windows ต้องมีไฟล์ต่อไปนี้ก่อน:

- `shared_model.keras`
- train-only scaler แยกทุก `machine_id + module_id`
- validation threshold แยกทุก group
- Golden Calibration Profile
- Frozen chronological holdout สำหรับ regression test

หลังจากนั้น Automatic Scoring ไม่ต้องเปิด Colab ค้างไว้ ระบบ Windows จะตรวจไฟล์ใหม่ตามรอบเวลาเอง Colab จะใช้อีกครั้งเมื่อสร้าง model version ใหม่หรือทำ full retraining เท่านั้น

## Safety boundary

ระบบแบ่งสิ่งที่เปลี่ยนได้กับเปลี่ยนไม่ได้อย่างชัดเจน:

| Component | Automatic change | Reason |
|---|---:|---|
| Shared model weights | No | ป้องกัน anomaly contamination และ catastrophic forgetting |
| Train-only model-input scaler | No | คงความหมายของ input ที่โมเดลเคยเรียน |
| Golden Calibration Profile | No | เก็บสภาพอ้างอิงระยะยาว |
| Operational Calibration Profile | Yes, bounded | รองรับ baseline เฉพาะเครื่องที่ขยับอย่างช้า ๆ |
| Threshold | Yes, bounded | ปรับได้ไม่เกิน guardrail ต่อ profile version |
| Fault diagnosis / maintenance approval | No | ต้องมี fault และ maintenance-linked labels |

การอนุมัติอัตโนมัติในระบบนี้หมายถึง **อนุมัติ Calibration Profile** ไม่ใช่รับรองว่าเครื่องไม่เสียหรืออนุมัติการซ่อมบำรุง

## End-to-end flow

1. Colab เทรน Shared Model จาก normalized windows ของทุกเครื่อง
2. Colab สร้าง Golden Profile และ Frozen Holdout แยกทุก machine-module
3. ดาวน์โหลด adaptive runtime ZIP กลับมายังโปรเจกต์ Windows
4. Windows ตรวจ daily logs ใหม่และข้ามไฟล์ที่เคยประมวลผลแล้ว
5. State/data-quality gate ตัด transition, Busy, sentinel, invalid pressure/temperature และ time gaps
6. Shared Model ให้ reconstruction error โดยใช้ train-only scaler เดิม
7. Adaptive layer ให้ feature-deviation, relation/covariance, Golden drift และ operational risk
8. เฉพาะ window ที่ปกติทั้ง Golden และ Operational gate ถูกเก็บเข้า calibration buffer
9. เมื่อข้อมูลครบ ระบบสร้าง Candidate Profile ด้วย bounded update
10. Candidate ต้องผ่าน frozen replay, false-alert guardrail และ synthetic perturbation regression
11. Candidate ทำ shadow observations กับข้อมูลใหม่หลายรอบ
12. ผ่านครบจึง `AUTO_APPROVED`; ไม่ผ่านเป็น `AUTO_REJECTED` หรือ `REVIEW_REQUIRED`
13. ทุก decision ถูกบันทึกและ profile ก่อนหน้าถูกเก็บสำหรับ rollback

## 1. รัน Shared Model และ Golden Calibration บน Colab

ใน `MyDrive/Data Analysis` ต้องมี:

```text
HT9046MX_Shared_Model_Colab.ipynb
ht9046mx_colab_package.zip
```

เปิด Notebook, เลือก T4 GPU และรันจากบนลงล่าง Cell 11–12 ที่เพิ่มใหม่จะ:

- สร้าง `adaptive_seed/`
- ตรวจว่า seed ครบทุก group ที่มี scaler/threshold
- ตรวจ safety contract
- สร้าง ZIP สำหรับ Windows

Smoke output:

```text
MyDrive/Data Analysis/artifacts/
├── shared_lstm_colab_smoke/
│   ├── shared_model.keras
│   ├── scalers/
│   ├── thresholds.json
│   ├── group_metrics.csv
│   ├── manifest.json
│   └── adaptive_seed/
└── ht9046mx_adaptive_runtime_smoke.zip
```

Smoke model ใช้ทดสอบ pipeline เท่านั้น ก่อนใช้ตัดสินใจ maintenance ต้องสร้าง `shared_full_v1`, เปลี่ยน `RUN_MODE='full'` และรันใหม่

## 2. นำ artifact กลับมายัง Windows

ดาวน์โหลดและแตก `ht9046mx_adaptive_runtime_smoke.zip` ให้โครงสร้างเป็น:

```text
Data Analysis/
├── artifacts/
│   └── shared_lstm_colab_smoke/
│       ├── shared_model.keras
│       ├── adaptive_seed/
│       └── ...
├── configs/
├── compressor_ml/
└── scripts/
```

จาก PowerShell ในโปรเจกต์:

```powershell
.\scripts\initialize_adaptive_runtime.ps1
```

คำสั่งจะ copy immutable seed ไปยัง writable `adaptive_runtime/` โดยไม่แก้ artifact ต้นฉบับ

## 3. ทดสอบ Automatic Scoring หนึ่งรอบ

```powershell
.\scripts\run_adaptive_cycle.ps1
```

ตรวจสถานะ:

```powershell
.\.venv\Scripts\python.exe -m compressor_ml.adaptive_runner status `
  --runtime-dir adaptive_runtime
```

รอบแรก default จะทำเครื่องหมาย historical files เก่าเป็น `baseline_skipped` และ score เฉพาะไฟล์ล่าสุดต่อเครื่อง หลังจากนั้นจะ score เฉพาะไฟล์ใหม่หรือไฟล์เดิมที่มี size/modified time เปลี่ยน เพื่อไม่ให้ผลซ้ำ

ใน smoke bundle ไฟล์ล่าสุดอาจเป็นวันเดียวกับที่ใช้สร้าง prepared dataset ดังนั้นผลรอบแรกใช้ตรวจว่า pipeline ทำงานครบเท่านั้น ห้ามนับเป็น independent model evaluation การประเมินจริงต้องใช้ไฟล์ที่เกิดหลัง training cutoff

หากต้องการ replay history ทั้งหมด ให้แก้ `process_history` เป็น `true` ใน `configs/adaptive_system.json` ก่อน initialize/run ครั้งแรก

## 4. ผลลัพธ์ที่ระบบสร้าง

```text
adaptive_runtime/
├── adaptive_config.json
├── seed_manifest.json
├── profiles/
│   └── MX007__M01/
│       ├── golden.json
│       ├── champion.json
│       ├── candidate.json        # มีเฉพาะช่วง shadow/review
│       ├── history/
│       └── rejected/
├── frozen/
├── buffers/
├── predictions/
│   └── MX007__M01.csv
├── monitoring/
│   ├── data_quality.csv
│   └── cycle_errors.csv
├── audit/
│   └── approval_log.jsonl
├── runs/
├── state/
│   └── processed_files.json
└── latest_cycle.json
```

Prediction หลักประกอบด้วย:

- `reconstruction_error`
- `golden_risk`
- `operational_risk`
- `combined_risk`
- `health_score`
- `condition_status`
- `baseline_drift`
- `eligible_for_calibration`
- feature/relation deviation
- `top_error_feature`
- model/profile version

## 5. เปิดการทำงานอัตโนมัติ

หลัง smoke cycle ผ่านและตรวจ `latest_cycle.json`, `data_quality.csv` และ prediction CSV แล้ว:

```powershell
.\scripts\install_adaptive_task.ps1 -IntervalMinutes 15
```

Scheduled Task จะเรียก cycle ทุก 15 นาทีและใช้ `MultipleInstances IgnoreNew` เพื่อป้องกันการรันซ้อน หากไฟล์ daily log มาเพียงวันละครั้ง สามารถใช้ interval 30–60 นาทีเพื่อลดการตรวจซ้ำได้

## Approval state machine

| Outcome | Meaning |
|---|---|
| `NO_CANDIDATE` | ยังมี eligible windows ใหม่ไม่พอ |
| `REVIEW_REQUIRED` | จำนวนวันไม่พอหรือผลไม่ชัดเจน |
| `AUTO_REJECTED` | Candidate ทำให้ frozen false alerts หรือ sensitivity regression |
| `SHADOW` | ผ่าน validation และกำลังรอข้อมูลใหม่เพิ่ม |
| `SHADOW_WAIT` | ยังไม่มี eligible data ใหม่หลัง shadow ครั้งล่าสุด |
| `AUTO_APPROVED` | ผ่าน validation และ shadow observations ครบ |
| `AUTO_ROLLED_BACK` | Frozen self-test ของ deployed profile regression และมี previous champion ให้คืนค่า |

ระบบไม่ใช้ live machine anomaly เป็นเหตุ rollback เพราะอาจเป็น fault จริง Rollback อัตโนมัติใช้เฉพาะ frozen calibration regression เท่านั้น

## Default guardrails

ค่าตั้งต้นอยู่ใน `configs/adaptive_calibration.json`:

- candidate อย่างน้อย 200 eligible windows
- ครอบคลุมอย่างน้อย 3 วัน
- buffer สูงสุด 5,000 windows ต่อ group
- adaptation rate 10%
- center ขยับไม่เกิน 0.1 Golden MAD ต่อ version
- scale เปลี่ยนไม่เกิน ±10%
- threshold เปลี่ยนไม่เกิน ±10%
- frozen reference alert rate ไม่เกิน guardrail
- synthetic detection rate อย่างน้อย 80%
- shadow observations อย่างน้อย 3 รอบที่มี eligible data ใหม่

ค่าเหล่านี้เป็น conservative engineering defaults ต้อง replay กับ full historical dataset และข้อมูล maintenance ก่อนถือเป็น production thresholds

## Full production checklist

- ใช้ prepared dataset หลายวันต่อเครื่อง ไม่ใช่ smoke วันเดียว
- Run chronological full training และตรวจทุก machine-module
- ยืนยันว่า Module 7 ยังควรถูก exclude
- เพิ่ม maintenance-event table พร้อม timestamp, machine, module, action และ confirmed condition
- วัด false alarms/day, precision, recall, event lead time และ missed-event rate
- ทดสอบ scheduled cycle หลายวันแบบ shadow ก่อนใช้แจ้ง maintenance
- สำรอง `adaptive_runtime/profiles`, `audit` และ `state`
- ให้ technician review `baseline_drift` ที่ต่อเนื่อง แม้ `condition_status` จะกลับเป็น Normal ภายใต้ Adaptive Profile
