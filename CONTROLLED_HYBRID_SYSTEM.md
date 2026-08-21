# HT-9046MX Controlled Hybrid v1

## 1. ข้อสรุปของแนวทางใหม่

ระบบใหม่ไม่ได้ให้ Shared LSTM เป็นผู้ตัดสินเพียงตัวเดียว และไม่ได้ปล่อยให้
ระบบปรับ baseline ที่ใช้งานอยู่เองทันที แต่แบ่งหน้าที่ชัดเจนดังนี้

- **COM2 explainable detector เป็น Primary**: เปรียบเทียบค่าของเครื่องกับ
  `machine + module + operating mode + regime` ของตัวเอง และบอกเหตุผลเป็น
  Robust Z, LP2 residual, pressure gap, temperature span, Isolation Forest
  และแนวโน้มหนึ่งชั่วโมง
- **Shared LSTM Full เป็น Shadow evidence**: ใช้โมเดลกลาง
  `shared_lstm_full_v1` ที่เทรน 30 epochs จาก 42 machine-module groups และรับ
  input `(60, 24)`; น้ำหนักโมเดลถูกล็อก ไม่เรียนต่อจากข้อมูล live
- **Automatic Bootstrap**: เมื่อเพิ่มเครื่องใหม่ ระบบอ่านข้อมูลย้อนหลังและ
  สร้าง Candidate Profile ให้เองโดยใช้ LSTM และ COM2 ช่วยกันกรองช่วงที่ผิดปกติ
- **Controlled Activation**: Candidate ต้องผ่าน Shadow Validation แล้วหยุดที่
  `APPROVAL_REQUIRED`; คนเป็นผู้ approve ครั้งแรก จากนั้นจึงคัดลอกเป็น
  `ACTIVE_FROZEN`
- **Abstention by design**: ข้อมูลไม่ครบ, เครื่องหยุด/กำลังเปลี่ยน state,
  regime ไม่รู้จัก หรือยังไม่มี active profile จะไม่ถูกเรียกว่า Normal

ผลลัพธ์จึงเป็น **condition monitoring และ anomaly review priority** ไม่ใช่
คำยืนยันว่า compressor จะเสียเมื่อใด และยังไม่ใช่ RUL/fault classifier จนกว่า
จะมี maintenance label ที่เชื่อถือได้

## 2. ลำดับการคำนวณทุก 5 นาที

```text
Raw log ตาม event time
  -> 5-minute window + Data Quality Gate
  -> State/transition/settling Gate
  -> SV + Valve operating mode
  -> GMM operating regime
  -> Frozen COM2 profile ของ machine/module/mode/regime
       |- Robust Z / MAD
       |- Ridge LP2 conditional residual
       |- Pressure gap + temperature span
       |- Isolation Forest
       `- 1-hour LP2 residual trend
  -> Shared LSTM Full shadow score
  -> Event-time persistence + hysteresis
  -> Fusion: NORMAL / SHADOW / P2_REVIEW / P1_REVIEW
  -> JSONL prediction + lifecycle/audit state
```

### 2.1 Event-time window และ quality gate

ให้ข้อมูลในหน้าต่างที่ (W_t=[t,t+300)) มีจำนวนจุด (n_t) และช่วงห่างมัธยฐาน
ระหว่างข้อมูลเป็น \(\widetilde{\Delta t}\) จำนวนจุดที่คาดหวังคือ

\[
n_{expected}=\max\left(1,\left\lfloor
\frac{300}{\widetilde{\Delta t}}\right\rfloor\right),\qquad
coverage=\frac{n_t}{n_{expected}}
\]

หน้าต่างจึงมีสิทธิ์ถูกประเมินเมื่อ coverage อย่างน้อย 0.90, มีอย่างน้อย 30
records, maximum gap ไม่เกิน 15 วินาที, timestamp ไม่ย้อนลำดับ, ไม่มี duplicate,
ไม่มี sentinel `-200`, และ pressure/temperature อยู่ในขอบเขต policy

ค่าตัวแทนของแต่ละ sensor ใช้ median ในหน้าต่าง 5 นาทีเพื่อลดผลจาก spike:

\[
\widetilde{x}_t=\operatorname{median}\{x_i:i\in W_t\}
\]

ผลของ gate ไม่ผ่านเป็น `DATA_QUALITY_REVIEW` หรือ `INCOMPLETE_WINDOW` ไม่ใช่
`NORMAL`

### 2.2 State และ settling gate

ระบบตัดช่วง `ChangeValve`, `AdjustValve`, `MValveHome`, startup และ shutdown
รวมถึงช่วง settling 120 วินาทีหลัง transition ออกก่อนเรียน baseline และก่อน
scoring เพื่อไม่ให้พฤติกรรมชั่วคราวถูกเข้าใจผิดว่าเป็นความเสียหาย

### 2.3 Deterministic operating mode

SV ถูกแบ่งเป็น `SV_ON/SV_OFF` และ valve ถูกแบ่ง bucket ที่ 20, 50, 80:

\[
mode=(SV\ state,\;B(Valve)),\quad
B(v)=\begin{cases}
B0&v<20\\
B1&20\le v<50\\
B2&50\le v<80\\
B3&v\ge80
\end{cases}
\]

การแยก mode ก่อนสร้าง baseline ป้องกันไม่ให้ distribution ของคนละ operating
condition ถูกผสมเข้าด้วยกัน

### 2.4 GMM regime

ในแต่ละ mode ใช้เวกเตอร์

\[
\mathbf{x}=[HP2,LP2,Valve,TempHi,TempLo]^T
\]

และ Gaussian Mixture Model

\[
p(\mathbf{x})=\sum_{k=1}^{K}\pi_k
\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)
\]

โดยลอง (K=1..3) และเลือกค่าที่ BIC ต่ำที่สุด

\[
BIC(K)=-2\log\hat{L}_K+p_K\log n
\]

posterior ของ component คือ

\[
\gamma_k(\mathbf{x})=
\frac{\pi_k\mathcal{N}(\mathbf{x}\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal{N}(\mathbf{x}\mid\mu_j,\Sigma_j)}
\]

ถ้า posterior สูงสุดต่ำกว่า 0.60 หรือ log likelihood ต่ำกว่า percentile 1 ของ
baseline/ต่ำกว่า policy floor ระบบคืน `UNKNOWN_REGIME` และ abstain ไม่เทียบกับ
profile ที่ไม่ตรงบริบท ถ้ามีข้อมูล mode น้อยกว่า 40 windows จะใช้ fallback R0
โดยไม่อ้างว่าได้ค้นพบ cluster ที่น่าเชื่อถือ

### 2.5 Robust baseline และ MAD

สำหรับ feature (x) ใน context เดียวกัน:

\[
m_x=\operatorname{median}(x),\qquad
MAD_x=\operatorname{median}(|x_i-m_x|)
\]

\[
s_x=\max(1.4826\,MAD_x,10^{-6}),\qquad
z_x=\frac{x-m_x}{s_x}
\]

ตัวคูณ 1.4826 ทำให้ MAD เทียบสเกลของ standard deviation ได้เมื่อ distribution
ใกล้ normal แต่ยังทน outlier มากกว่า mean/std ระบบเข้า anomaly ที่
\(|z|\ge3.5\) และออกเมื่อ \(|z|<2.5\) เพื่อลดการสั่นเข้าออกของสถานะ

feature ที่เก็บ baseline ได้แก่ `HP1`, `LP1`, `HP2`, `LP2`, `Valve`,
`TempHi`, `TempLo`, pressure gap, pressure ratio และ temperature span โดย trigger
หลักใช้ HP2, conditional LP2 residual, pressure gap และ temperature span

\[
gap=HP2-LP2,\qquad ratio=\frac{HP2}{\max(|LP2|,0.1)},\qquad
temp\_span=TempHi-TempLo
\]

### 2.6 Ridge conditional residual ของ LP2

LP2 ไม่ควรถูกเทียบกับค่าคงที่อย่างเดียว เพราะเปลี่ยนตาม load และ control state
จึงสร้าง expected LP2 จาก

\[
\widehat{LP2}=\beta_0+\beta_1HP2+\beta_2Valve+
\beta_3TempHi+\beta_4TempLo
\]

ค่าสัมประสิทธิ์หาโดย Ridge:

\[
\hat{\boldsymbol{\beta}}=
\arg\min_{\boldsymbol{\beta}}
\sum_i(LP2_i-\beta_0-\mathbf{x}_i^T\boldsymbol{\beta})^2+
\lambda\|\boldsymbol{\beta}\|_2^2,\quad\lambda=1
\]

จากนั้นคำนวณ residual และ Robust Z ของ residual:

\[
r_t=LP2_t-\widehat{LP2}_t,\qquad
z_r=\frac{r_t-\operatorname{median}(r)}{1.4826MAD(r)}
\]

ค่า (z_r\le-3.5) หมายถึง LP2 ต่ำกว่าที่ควรเป็นเมื่อเทียบกับ condition ปัจจุบัน
และให้ reason `LP2_NEGATIVE_RESIDUAL`

### 2.7 Isolation Forest

เวกเตอร์เข้า Isolation Forest คือ

\[
\mathbf{v}_t=[z_{HP2},z_{LP2\ residual},z_{gap},z_{temp\ span}]
\]

ใช้ 200 trees และ anomaly score ตามแนวคิด path length

\[
s(\mathbf{v},n)=2^{-E[h(\mathbf{v})]/c(n)}
\]

ระบบใช้ score จาก implementation ของ scikit-learn แล้ว calibrate entry ที่
quantile 0.99 และ exit ที่ quantile 0.95 ของ healthy candidate context แทนการ
ใช้ threshold เดียวกับทุกเครื่อง

### 2.8 Trend และ persistence

แนวโน้ม LP2 ใช้การเปลี่ยน Robust residual Z ระหว่างจุดแรกกับจุดล่าสุดใน
event-time lookback หนึ่งชั่วโมง:

\[
trend_t=z_r(t)-z_r(t-1h)
\]

เข้า reason `LP2_DOWNWARD_TREND` เมื่อ trend ไม่เกิน -3 และใช้ exit hysteresis
ที่ค่าสัมบูรณ์ต่ำกว่า 2

persistence นับตาม event time ไม่ใช่เวลาที่โปรแกรมรัน หาก gap เกิน 600 วินาที
หรือลำดับเวลาย้อน ระบบ reset continuity เพื่อไม่เอาช่องว่างของ logger ไปเพิ่ม
ระยะเวลา anomaly

### 2.9 Shared LSTM Full shadow

Shared LSTM เป็น sequence autoencoder ให้ encoder-decoder สร้างลำดับกลับมา:

\[
\widehat{\mathbf{X}}=f_\theta(\mathbf{X}),\qquad
e=\frac{1}{TF}\sum_{t=1}^{T}\sum_{j=1}^{F}
(X_{t,j}-\widehat{X}_{t,j})^2
\]

โดย (T=60,F=24) แต่ละ machine-module ใช้ train-only scaler และ threshold จาก
validation ของตัวเอง แม้น้ำหนัก (\theta) ใช้ร่วมกันทั้งหมด ในหนึ่ง bucket ระบบ
ใช้ percentile 95 ของ sequence reconstruction errors เป็น bucket score

สำหรับ group เดิมใช้ scaler/threshold จาก Full artifact สำหรับเครื่องใหม่
อนุญาตให้ fit **เฉพาะ local scaler และ threshold** จากข้อมูลย้อนหลังแบบ 70/15;
ไม่ update Shared LSTM weights และไม่เปลี่ยน calibration ของเครื่องอื่น

### 2.10 Fusion และระดับผลลัพธ์

| เงื่อนไข | ผลลัพธ์ |
|---|---|
| quality/state/profile/regime gate ไม่ผ่าน | สถานะ gate นั้น ไม่ใช่ Normal |
| COM2 และ LSTM ไม่ active | `NORMAL` |
| มี evidence แต่ยังไม่ persistent | `SHADOW` |
| detector ใด detector หนึ่งต่อเนื่องอย่างน้อย 30 นาที | `P2_REVIEW` |
| COM2 และ LSTM ต่อเนื่องพร้อมกันอย่างน้อย 15 นาที | `P1_REVIEW` |

`P1/P2` คือ priority ให้คนตรวจ ไม่ใช่คำสั่งหยุดเครื่องอัตโนมัติ

## 3. Automatic Bootstrap และ Frozen Profile lifecycle

```text
COLLECTING_DATA
  -> LEARNING
  -> CANDIDATE_PROFILE_READY
  -> SHADOW_VALIDATION
  -> APPROVAL_REQUIRED
  -> ACTIVE (ACTIVE_FROZEN)
                         `-> REJECTED (เมื่อผู้ตรวจไม่อนุมัติ)
```

1. รอข้อมูล eligible อย่างน้อย 7 วันและ 200 windows; แนะนำ 14 วัน
2. ถ้าเครื่องใหม่ไม่มี group calibration ให้ fit เฉพาะ local LSTM scaler และ
   validation-p99 threshold โดยไม่แก้น้ำหนักโมเดลกลาง
3. LSTM คัดเฉพาะ windows ที่ score/threshold ไม่เกิน 1.0
4. fit preliminary COM2 จากชุดนั้น แล้วเอา windows ที่ preliminary COM2 flag ออก
5. fit final Candidate Profile รอบที่สอง แยก machine/module/mode/regime
6. รัน shadow อย่างน้อย 3 วันและ 100 eligible windows
7. ผ่าน gate เมื่อ unknown regime ไม่เกิน 5%, COM2 flag ไม่เกิน 5% และ LSTM flag
   ไม่เกิน 10%
8. เปลี่ยนเป็น `APPROVAL_REQUIRED` เท่านั้น ไม่มี auto-activation
9. เมื่อคน approve จึง copy candidate เป็น versioned `ACTIVE_FROZEN`; ระบบ live
   อ่าน active version เดิมต่อไปจนกว่าจะมี approval ใหม่

## 4. Phase ของโครงการ

### Phase A — Data contract และ empirical validation (เสร็จแล้ว)

ตรวจ schema/log จริง, event time, module state, sentinel, transition, sampling gap,
feature ranges และยืนยันว่าไม่ควรเรียกข้อมูลที่ gate ไม่ผ่านว่า normal

### Phase B — Full Shared LSTM (เสร็จแล้ว)

เทรน Colab 30 epochs จาก 42 groups ได้ `shared_lstm_full_v1`; ใช้ร่วมกันเฉพาะ
weights ส่วน scaler/threshold แยก group และตอนนี้ถูกเปลี่ยนบทบาทเป็น immutable
shadow/bootstrap evidence

### Phase C — Explainable Controlled Hybrid core (เสร็จในโค้ด)

เพิ่ม 5-minute gates, mode, GMM, Robust Z/MAD, Ridge residual, Isolation Forest,
trend, persistence, fusion, reason codes และ abstention

### Phase D — Automatic onboarding + controlled activation (เสร็จในโค้ด)

เพิ่ม two-pass bootstrap, candidate repository, shadow gates, audit history และ
mandatory human approval ก่อน Active Frozen Profile

### Phase E — Production integration/validation (ขั้นถัดไป)

- รัน bootstrap จริงครบทุกเครื่องและตรวจ rejected/insufficient modules
- เก็บ shadow ต่อเนื่องอย่างน้อย 3 วันต่อเครื่องก่อน approve
- เชื่อมผล JSONL/lifecycle เข้าฐานข้อมูลและหน้า Dashboard approval
- เพิ่ม alert acknowledgement, profile comparison, rollback และ audit display
- ทำ replay จากเหตุการณ์ซ่อมจริงเพื่อวัด false alert/day และ detection lead time
- หลังมี fault/maintenance labels เพียงพอ จึงพิจารณา supervised fault classifier
  หรือ survival/RUL model แยกต่างหาก โดยไม่แทน quality gates

## 5. วิธีรันบน Windows

ติดตั้ง dependency หนึ่งครั้ง:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

ตรวจ lifecycle โดยไม่โหลด TensorFlow:

```powershell
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config configs\controlled_condition_monitoring.json status
```

สร้าง candidate เครื่องเดียวจากข้อมูลย้อนหลัง:

```powershell
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config configs\controlled_condition_monitoring.json bootstrap `
  --machine-id MX017
```

รัน scoring cycle ทุกเครื่อง:

```powershell
.\scripts\run_controlled_monitoring_cycle.ps1
```

เมื่อ state เป็น `APPROVAL_REQUIRED` และตรวจ evidence/profile แล้ว:

```powershell
.\.venv\Scripts\python.exe -m compressor_ml.controlled_monitoring.runner `
  --system-config configs\controlled_condition_monitoring.json approve `
  --machine-id MX017 --approved-by "Engineer.Name"
```

ติดตั้ง Task Scheduler ทุก 5 นาทีใน PowerShell ที่ Run as Administrator:

```powershell
.\scripts\install_controlled_monitoring_task.ps1 -IntervalMinutes 5
```

ผลลัพธ์อยู่ใน `controlled_runtime/predictions/<machine>.jsonl`, lifecycle และ
profile versions อยู่ใต้ `controlled_runtime/profiles/` และ cycle ล่าสุดอยู่ที่
`controlled_runtime/latest_cycle.json`

## 6. ข้อจำกัดสำคัญ

- baseline ที่สร้างอัตโนมัติอาศัยสมมติฐานว่า historical data ส่วนใหญ่ดี จึงใช้
  LSTM + two-pass COM2 + shadow gate ลด contamination แต่รับประกันไม่ได้ว่าไม่มี
  latent fault
- ไม่มี labeled failure จำนวนพอ จึงยังวัด sensitivity/specificity ต่อชนิดความ
  เสียจริงไม่ได้
- Full LSTM ช่วยจับ temporal/multivariate pattern แต่ไม่ควรถูกตีความเป็น failure
  probability
- profile ไม่เรียน live เองหลัง Active; การเปลี่ยน active version ต้องสร้าง
  candidate ใหม่, validate และ approve เพื่อป้องกัน baseline drift ไล่ตาม fault
