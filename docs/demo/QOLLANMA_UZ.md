# G1 demo platformasi — to'liq qo'llanma

> **Kim uchun:** G1 jamoasining barcha a'zolari. Platformani ishga tushiradigan, fayl yuklaydigan va
> professor oldida demonstratsiya qiladigan har bir kishi uchun.
> **Kod versiyasi:** `main` branch (PR #5 merge qilingandan keyin). Yangi kodni olgach, har bir recording'da
> **"Re-run pipeline on this recording"** tugmasini bir marta bosing (15-bo'lim, 0-qadam).
> Sidebar'dagi `code version` yozuvi qaysi versiya ishlayotganini ko'rsatadi.

---

## Mundarija

1. [Platforma nima va nima uchun kerak](#1-platforma-nima-va-nima-uchun-kerak)
2. [Asosiy tushunchalar lug'ati](#2-asosiy-tushunchalar-lugati)
3. [Ma'lumot qanday oqadi: katta rasm](#3-malumot-qanday-oqadi-katta-rasm)
4. [Platformani ishga tushirish](#4-platformani-ishga-tushirish)
5. [Birinchi ochilish: built-in sintetik yozuv](#5-birinchi-ochilish-built-in-sintetik-yozuv)
6. [Qaysi fayllar kerak va ularni qanday tayyorlash](#6-qaysi-fayllar-kerak-va-ularni-qanday-tayyorlash)
7. [Yuklash: tugmani bosgandan keyin ichkarida nima bo'ladi](#7-yuklash-tugmani-bosgandan-keyin-ichkarida-nima-boladi)
8. [Sidebar: barcha boshqaruv elementlari](#8-sidebar-barcha-boshqaruv-elementlari)
9. [Modul sahifalari: har bir student qismi](#9-modul-sahifalari-har-bir-student-qismi)
10. [Demonstratsiya stsenariysi (qadam-baqadam)](#10-demonstratsiya-stsenariysi-qadam-baqadam)
11. ["Nima uchun shunday natija chiqdi?" — savol-javob](#11-nima-uchun-shunday-natija-chiqdi--savol-javob)
12. [Muammolar va yechimlar](#12-muammolar-va-yechimlar)
13. [Cheklovlar: professorga halol aytilishi kerak bo'lganlar](#13-cheklovlar-professorga-halol-aytilishi-kerak-bolganlar)
14. [Ilova: fayllar, buyruqlar, sozlamalar](#14-ilova-fayllar-buyruqlar-sozlamalar)
15. [Midterm mashqi: har bir harakatni tekshirish ro'yxati](#15-midterm-mashqi-har-bir-harakatni-tekshirish-royxati)

---

## 1. Platforma nima va nima uchun kerak

Loyihamizda yetti modul bor (S1…S7). Har bir student o'z modulini alohida yozgan, modullar esa bir-biriga
**JSON fayllar** orqali ma'lumot uzatadi. Professor tizimni **bitta butun** holda ko'rmoqchi: kamera
yozgan ma'lumot kirsa, oxirida tekshirilgan javob va xavfsiz harakat (ActionGoal) chiqishi kerak.

Demo platformasi (`demo/app.py`, Streamlit veb-ilovasi) aynan shuni qiladi:

1. **S1 yozgan ZED Mini bag'ini** o'qiydi. Bu `ros2 bag play S1_bag` buyrug'ining o'rnini bosadi:
   ROS o'rnatilmagan bulutda ham, MacBook'da ham ishlaydi.
2. Har bir frame'ni **barcha modullardan ketma-ket o'tkazadi**: S3 → S2 → S6 → S4.
3. Savol berilganda **S5 → S6 (action)** bosqichlarini bajaradi.
4. Har bir modul natijasini **alohida sahifada** ko'rsatadi. Shunda har bir student xuddi shu frame
   ustida o'z ishini ko'rsata oladi.
5. **Play** tugmasi bilan frame'lar ketma-ket ijro etiladi, ya'ni bag xuddi kamera jonli ishlayotgandek
   o'ynaydi.

**Nima uchun bu muhim:** professor "yetti alohida mini-loyiha emas, bitta tizim bo'lsin" deb talab qilgan
(W6–7 feedback, 17-bo'lim). Platforma shu talabning isboti.

---

## 2. Asosiy tushunchalar lug'ati

| Atama | Ma'nosi | Platformada qayerda |
|---|---|---|
| **Bag (rosbag2)** | ROS 2 yozuv fayli. Papka ichida `metadata.yaml` (mazmuni) va `.db3` (ma'lumotlar bazasi) bo'ladi | S1 sahifasi, yuklash |
| **Topic** | Bag ichidagi ma'lumot kanali, masalan `/zed/zed_node/rgb/image_rect_color` | S1 → "Topics in the bag" |
| **Frame** | Bir lahzadagi bitta RGB rasm va unga mos depth rasm | Sidebar → Frame slider |
| **Depth** | Har bir pikselgacha bo'lgan masofa (metrda) | S1 → Depth heat-map |
| **Sinxronlash** | RGB va depth xabarlarini vaqt bo'yicha juftlash (farq ≤ 20 ms) | S1 → "synchronized pairs" |
| **Intrinsics** | Kamera ichki parametrlari: `fx, fy` (fokus), `cx, cy` (markaz). 2D pikselni 3D'ga o'tkazish uchun kerak | S1 → "Camera intrinsics" |
| **camera_info** | Bag ichidagi intrinsics topic'i. Bo'lmasa, fallback fayl ishlatiladi | S1 → "source:" |
| **TF / `T_map_camera`** | Kamera stol (map) koordinatasiga nisbatan qayerda va qanday burchakda turgani (4×4 matritsa) | S1 → Extrinsics |
| **Mask** | S3 (SAM 2) chiqargan, obyektga tegishli piksellar xaritasi | S3 sahifasi |
| **observation.json** | S3 chiqishi: har bir aniqlangan obyekt uchun box, label, score, mask fayli | S3 → JSON |
| **ObjectInstance3D** | S2 chiqishi: obyektning 3D markazi, 3D boxi, o'lchami, depth confidence | S2 sahifasi |
| **depth_confidence** | Obyekt maskasi ichida depth qanchalik ishonchli (0…1) | S2 grafigi |
| **depth_status** | `ok`, `transparent_suspected` (shisha, ichidan ko'rinadi) yoki `no_depth` | S2 jadvali |
| **Persistent ID** | Obyektning frame'lar bo'ylab saqlanadigan nomi, masalan `mug_01` | S6 Tracking |
| **Scene graph** | Obyektlar (node) va ular orasidagi munosabatlar (edge: `on`, `near`, `left_of`) | S4 sahifasi |
| **Evidence** | Edge'ni isbotlovchi raqamlar, masalan `distance_m=0.22` | S4 jadvali |
| **Intent JSON** | S5 parser savoldan chiqargan tuzilma: `target`, `reference`, `negative_constraints` | S5 → "Parsed intent" |
| **ReasoningResult** | S5 chiqishi: javob, isbot (`computed_facts`), ishonch, ogohlantirishlar | S5 sahifasi |
| **ActionGoal** | S6 chiqishi: nishon koordinatasi, exclusion zone'lar, `validated` (xavfsizmi) | S6 Action |
| **Exclusion zone** | Virtual agent kesib o'tmasligi kerak bo'lgan hudud (hozir laptop atrofida) | S6 Action, qizil maydon |
| **Virtual twin** | Stolning yuqoridan ko'rinishidagi soddalashtirilgan modeli; agent shu yerda harakat qiladi | S6 Action |
| **Contract** | Modullar orasidagi JSON format qoidasi. S7 har bir uzatishda tekshiradi | S7 sahifasi |
| **Recording / dataset** | Platformaga import qilingan bitta bag (frame'lari bilan) | Sidebar → Recording |

---

## 3. Ma'lumot qanday oqadi: katta rasm

```
                    YUKLASH (bir marta)                               HAR BIR FRAME UCHUN                SAVOL BERILGANDA
┌────────────────────────────────────────────┐   ┌──────────────────────────────────────────┐   ┌──────────────────────────┐
│ metadata.yaml + .db3 (S1)                   │   │ S3 masks  →  S2 3D  →  S6 IDs  →  S4 graph│   │ S5 parse → S5 verify     │
│ S3 zip (observation.json + masks)           │──▶│ (replay)     ObjectInstance3D    graph JSON│──▶│ → S6 ActionGoal → twin   │
│ [camera_intrinsics.yaml]                    │   │                                          │   │                          │
└────────────────────────────────────────────┘   └──────────────────────────────────────────┘   └──────────────────────────┘
          │                                                     │                                        │
          ▼                                                     ▼                                        ▼
   datasets/<nomi>/frames/frame_XXXX/                 runs/<nomi>/objects, scene_graph,          sahifalarda darhol ko'rinadi
   (rgb.png, depth, intrinsics, observation.json)     points, run_summary.json                   (diskka yozilmaydi)
```

**Muhim:** og'ir hisob-kitoblar (S2 3D tozalash, S4 graf) **import paytida bir marta** bajariladi va
diskka saqlanadi. Demonstratsiya paytida Play bosilganda platforma tayyor natijalarni ko'rsatadi, shuning
uchun replay tez ishlaydi. Faqat savol-javob (S5 → S6) har safar jonli hisoblanadi.

**Pipeline tartibi professor tomonidan "muzlatilgan":** S1 → S3 → S2 → S6 → S4 → S5 → S6 → S7.
S6 identifikatorlarni S4 grafidan **oldin** beradi, chunki graf vaqt bo'yicha bir xil obyektni tanishi kerak.

---

## 4. Platformani ishga tushirish

Uch yo'l bor. Taqdimot uchun eng ishonchlisi **MacBook**: internet uzilsa ham ishlaydi va LLM lokal.

### 4.1 MacBook (tavsiya etiladi)

Bir martalik tayyorgarlik:
1. **Ollama** o'rnating: https://ollama.com/download → Applications → bir marta oching.
2. **Homebrew** va Open3D uchun kutubxonalar:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv zsh)"
   brew install libusb libomp
   ```
3. Kodni oling:
   ```bash
   git clone https://github.com/intodakt/physical_ai_scene.git
   ```

Har safar ishga tushirish:
```bash
cd ~/physical_ai_scene
git checkout main
git pull                  # eng yangi versiya
./G1_Demo.command         # yoki Finder'da ikki marta bosing
```

`G1_Demo.command` ketma-ket quyidagilarni bajaradi:

| Qadam | Nima qiladi | Nima uchun |
|---|---|---|
| 1 | `uv` orqali Python 3.11 muhitini (`.venv`) yaratadi, birinchi marta bir necha daqiqa | Hamma kompyuterda bir xil versiyalar bo'lishi uchun |
| 2 | `requirements-demo.txt` paketlarini o'rnatadi (faqat fayl o'zgarganda) | Sinalgan versiyalar |
| 3 | Open3D yuklanmasa, `brew install libusb libomp` | Open3D Mac'da shularsiz ishlamaydi |
| 4 | Ollama'ni ishga tushiradi, `qwen2.5:3b` modeli bo'lmasa yuklab oladi (~2 GB) | S5 parser uchun lokal LLM |
| 5 | 8501-portda eski demo ishlayotgan bo'lsa, uni to'xtatadi | "Port 8501 is not available" xatosi bo'lmasligi uchun |
| 6 | Streamlit'ni ishga tushiradi va brauzerda `http://localhost:8501` ni ochadi | Demo sahifasi |

To'xtatish: Terminal oynasida **Ctrl+C**.

### 4.2 Railway (bulut)
1. Fork: https://github.com/intodakt/physical_ai_scene → **Fork** ("Copy the main branch only" belgisini
   **olib tashlang**).
2. railway.com → New Project → Deploy from GitHub → fork'ingiz.
3. **Settings → Source → Branch = `main`** (Dockerfile endi `main`da bor).
4. Settings → Networking → **Generate Domain**.
5. Tavsiya: **Volume** qo'shing, mount path `/data`. Shunda yuklangan recording'lar redeploy'dan keyin ham
   saqlanib qoladi.

Bulutda Ollama yo'q, shuning uchun S5 parser avtomatik **rules** rejimiga o'tadi (8.1-bo'limga qarang).

### 4.3 Google Cloud Run
Batafsil buyruqlar `docs/demo/README.md` faylida. Asosiy farq: Cloud Run **32 MB dan katta faylni
brauzer orqali qabul qilmaydi**. Katta bag'larni Google Drive linki orqali yuklash kerak (7.3-bo'lim).

---

## 5. Birinchi ochilish: built-in sintetik yozuv

Platforma birinchi marta ochilganda **`demo_synthetic_desk (synthetic)`** degan recording avtomatik
yaratiladi (taxminan 10 soniya). Bu nima?

- Kompyuterda **chizilgan (ray-cast qilingan)** stol sahnasi: laptop, mug, book, multimeter, glass.
- U ZED formatidagi **haqiqiy ROS 2 bag**ga yoziladi (camera_info va TF bilan) va xuddi jamoa
  ma'lumotlari kabi import qilinadi. Ya'ni bu yozuv ham S1 replay yo'lini to'liq sinaydi.
- 8 frame'dan iborat. Mug har frame'da taxminan 1,3 sm laptop tomon siljiydi (jami ~9 cm), book esa
  4–5-frame'larda olib qo'yiladi va keyin qaytariladi.

**Nima uchun kerak:**
1. Internet yoki real fayl bo'lmasa ham demo doim ishlaydi.
2. **Ground truth ma'lum:** har bir obyektning haqiqiy joyi bizga aniq. Shuning uchun xatolikni o'lchash
   mumkin (masalan, S2'ning shisha xatoligi 5,2 → 1,0 sm).
3. Tracking (mug siljishi, book yashirinishi) va safety (laptop orqali o'tuvchi yo'l rad etiladi) sinab
   ko'riladigan holatlar bunda ataylab bor.

**Demonstratsiyada:** bu ma'lumot sintetik ekanini **doim ochiq aytib turing**. Asosiy ko'rsatuv real bag
bilan bo'lishi kerak.

---

## 6. Qaysi fayllar kerak va ularni qanday tayyorlash

### 6.1 Majburiy va ixtiyoriy fayllar

| # | Fayl | Kimdan | Majburiymi | Nima uchun |
|---|---|---|---|---|
| 1 | `metadata.yaml` | S1 (bag papkasidan) | Tavsiya etiladi | Bag mazmuni. Bo'lmasa, `.db3` to'g'ridan-to'g'ri o'qiladi |
| 2 | `*.db3` (masalan `S1_static_bag_0.db3`) | S1 | **Ha** | RGB, depth, camera_info, TF shu yerda |
| 3 | S3 natijasi `.zip` (masalan `20261016T050736_395812Z.zip`) | S3 (Li Yizhe) | **Ha*** | Maskalar. S2 qaysi piksel qaysi obyektga tegishli ekanini shundan biladi |
| 4 | `camera_intrinsics.yaml` | S1 | Yo'q | Faqat bag'da `camera_info` bo'lmasa kerak |

\* Agar server YOLO-World bilan yig'ilgan bo'lsa (`WITH_S3_LIVE=1`), S3 natijasisiz ham ishlaydi. Oddiy
image'da esa S3 natijasi **shart**.

**S3 zip ichida nima bo'lishi kerak:** S3 `src/open_vocab/run.py` chiqaradigan tuzilma, ya'ni
`frame_000000/observation.json` va `frame_000000/masks/det_0000.png` va hokazo. Zip'ni **ochmasdan**
yuklang. Platforma uni o'zi ochadi.

### 6.2 Nomlash qoidasi (bir nechta bag bo'lsa)
Bitta bag va bitta S3 zip bo'lsa, ular avtomatik juftlanadi. Bir nechta bo'lsa, S3 papka nomida bag nomi
yoki **S-tegi** bo'lishi kerak:

| Bag | S3 papka nomi | Natija |
|---|---|---|
| `S2_bag` | `output/s3_semantic/S2_static/...` | ✓ juftlandi (tag `S2`) |
| `S1_static_bag` | `S1_static_bag_run/...` | ✓ juftlandi (to'liq nom) |
| `S4_bag` | `20261016T...` (tag yo'q, boshqa bag'lar ham bor) | ✗ juftlanmaydi → maskasiz qoladi |

### 6.3 Katta bag'lar: `demo/pack.py`
90 soniyalik HD720 bag bir necha GB bo'ladi. Uni bulutga brauzer orqali yuklash qiyin, Cloud Run'da esa
umuman mumkin emas. Yechim — bag'lar turgan kompyuterda kichik zip qilish:
```bash
pip install -r requirements-demo.txt
python -m demo.pack --src D:\file\zed\demo_input --out G1_S1_demo.zip --max-frames 40
```
`pack` buyrug'i:
- bag'ni S3 vaqtlarida qayta ijro etadi va faqat kerakli frame'larni oladi;
- depth'ni 16-bit mm PNG formatiga o'tkazadi (1 mm aniqlikda yo'qotishsiz, `.npy` ga qaraganda ~4 baravar
  kichik);
- natijani 50–150 MB atrofidagi zip qiladi.

Bu zip'ni sidebar'dagi **"One zip with everything"** joyiga yuklang yoki Google Drive linki qilib bering.

---

## 7. Yuklash: tugmani bosgandan keyin ichkarida nima bo'ladi

### 7.1 Sizning qadamlaringiz
1. Sidebar → **➕ Add recording** ni oching.
2. **Max frames per bag** (standart 40, oraliq 5–120): bag'dan nechta frame olinishi. Ko'p bo'lsa,
   replay uzunroq bo'ladi, lekin import sekinlashadi. Har frame S2'da taxminan 0,1–0,8 s oladi.
3. **1 · S1 bag** → `metadata.yaml` **va** `.db3` ni **birga** belgilab tanlang (Cmd/Ctrl bilan).
4. **2 · S3 output** → S3 zip.
5. **3 · Intrinsics** → kerak bo'lsa.
6. **Recording name** `.db3` nomidan o'zi to'ladi (`S1_static_bag_0.db3` → `S1_static_bag`). O'zgartirsa
   bo'ladi.
7. **Import recording** ni bosing va progress bar tugashini kuting (456 MB bag uchun bir necha daqiqa).
8. Sidebar'da yashil **"Imported: S1_static_bag"** chiqadi va recording avtomatik tanlanadi.

Tugma ostidagi ogohlantirishlar:
- **"⚠ add the .db3 file too"** — faqat `metadata.yaml` tanlangan, `.db3` ham kerak.
- **"metadata.yaml missing…"** — `.db3` to'g'ridan-to'g'ri o'qiladi (ROS 2 Humble bag'lari uchun ishlaydi).
- **"⚠ no S3 output…"** — maskalarsiz import bo'lmaydi (oddiy image'da).

### 7.2 Ichkarida qadam-baqadam (`demo/runs.py` → `src/integration/demo_data.py` → `src/sensors/bag_replay.py`)

| # | Bosqich | Nima bo'ladi | Nima uchun |
|---|---|---|---|
| 1 | Saqlash | Fayllar diskka bo'laklab yoziladi (`data/demo_store/_parts/<nomi>/`) | Katta faylni xotiraga to'liq yuklamaslik uchun |
| 2 | S3 zip'ni ochish | Xavfsiz ochiladi: zip ichidagi `../` kabi yo'llar rad etiladi | Zip orqali tizim fayllarini buzishning oldini olish (zip-slip) |
| 3 | Qidirish (discover) | Bag'lar (`metadata.yaml` yoki `.db3`), S3 run'lar (`observation.json`), intrinsics topiladi | Qaysi fayl nima ekanini avtomatik aniqlash |
| 4 | Juftlash | Har bir bag'ga o'z S3 run'i topiladi (6.2-bo'lim) | Maskalar to'g'ri bag'ga tushishi uchun |
| 5 | Bag'ni ochish | Avval `rosbags` (metadata.yaml orqali). Xato bo'lsa, `.db3` SQLite sifatida to'g'ridan-to'g'ri o'qiladi | Fayl nomi o'zgargan bo'lsa ham ishlashi uchun |
| 6 | Topic'larni aniqlash | ZED nomlari (`/zed/zed_node/...`), qisqa nomlar (`/rgb`, `/depth`) yoki istalgan Image juftligi | S1 qanday nom bergan bo'lsa ham ishlashi uchun |
| 7 | Sinxronlash | RGB va depth xabarlari header vaqti bo'yicha juftlanadi, farq ≤ 20 ms | Rasm va masofa bir lahzaga tegishli bo'lishi uchun |
| 8 | Intrinsics | `camera_info` bo'lsa, o'sha. Bo'lmasa `configs/sensors/camera_intrinsics.yaml` rasm o'lchamiga moslab ishlatiladi | 2D → 3D hisob shusiz bo'lmaydi |
| 9 | TF | `/tf_static` va `/tf` dan map → kamera zanjiri yig'iladi. Kamera "body" frame'da bo'lsa, optik o'qlarga aylantiriladi | Obyekt balandligi va "stol ustida" degan munosabat to'g'ri chiqishi uchun |
| 10 | **Vaqt bo'yicha moslash** | S3 vaqtlari uch usulda sinab ko'riladi: (a) bag yozilgan vaqt, (b) kamera header vaqti, (c) videoning boshidan hisoblangan soniyalar. Eng ko'p frame mos kelgani tanlanadi. Tolerans = max(0,05 s, S3 kadrlari orasidagi o'rtacha intervalning yarmi) | Real ZED bag'larida kamera soati bilan yozuvchi soati farq qiladi. Aynan shu sabab birinchi import "exported 0" bilan tugagan edi |
| 11 | Frame chiqarish | Har bir S3 vaqtiga eng yaqin sinxron juftlik `frames/frame_XXXX/` ga yoziladi: `rgb.png`, `depth.npy` yoki `.png`, `camera_intrinsics.yaml`, `frame_meta.json` | Pipeline o'qiydigan standart format |
| 12 | Maskalarni ko'chirish | `observation.json` va maskalar frame papkasiga ko'chiriladi. S3 boshqa o'lchamda ishlagan bo'lsa, mask va box'lar qayta masshtablanadi | Mask piksellari RGB bilan aynan ustma-ust tushishi uchun |
| 13 | `session_meta.json` | Topic'lar, xabar soni, chastota, davomiylik, intrinsics va TF manbasi, S3 moslash usuli yoziladi | Professor talabi: "har bir bag uchun metadata" |
| 14 | Pipeline | Har bir frame: S1 yuklash → S3 (replay) → S2 → S6 tracking → S4. Har bosqich vaqti o'lchanadi, har uzatishda contract tekshiriladi | Hamma sahifalar uchun natijalar |
| 15 | Standart savol | `"Go to the mug"` (`configs/s7_integration.yaml`) S5 → S6 orqali o'tkaziladi | S5/S6 sahifalari bo'sh qolmasligi uchun |
| 16 | `run_summary.json` | Latency, FPS, contract xatolari, frame xatolari, metadata | S7 sahifasi va yuklab olish uchun |

Agar maskalar vaqt bo'yicha mos kelmasa, import to'xtamaydi: maskalar **tartib bo'yicha** juftlanadi va
sariq ogohlantirish chiqadi: **"paired by order: CHECK that the masks sit on the objects (S3 page)"**.
Bunday holda S3 sahifasida maskalar obyektlar ustida turganini ko'z bilan tekshiring.

### 7.3 Boshqa yuklash yo'llari
- **One zip with everything:** `pack.py` chiqargan zip yoki bag papkasi va S3 papkalari bir zip ichida.
  Zip ichidagi zip ochilmaydi, shuning uchun S3 zip'ini avval oching.
- **Link:** Google Drive ("Anyone with the link") yoki to'g'ridan-to'g'ri `.zip`/`.db3` URL. Cloud Run
  uchun yagona yo'l.
- **Avtomatik:** serverda `DEMO_DATA_URLS=<link1>,<link2>` o'zgaruvchisi bo'lsa, start paytida bir marta
  import qilinadi.

### 7.4 Ma'lumot qayerda saqlanadi
```
data/demo_store/                      (Railway/Docker'da /data/demo_store)
├── datasets/<nomi>/frames/frame_XXXX/   rgb.png, depth, camera_intrinsics.yaml, frame_meta.json, observation.json, masks/
├── datasets/<nomi>/session_meta.json
└── runs/<nomi>/
    ├── run_summary.json
    ├── objects/frame_XXXX_objects.json        (S2 + S6 chiqishi)
    ├── scene_graph/frame_XXXX_scene_graph.json (S4 chiqishi)
    ├── tracking/frame_XXXX_track_state.json    (S6: track holatlari va hodisalar), event_log.jsonl
    └── points/frame_XXXX.npz                  (3D ko'rinish uchun nuqtalar)
```

---

## 8. Sidebar: barcha boshqaruv elementlari

| Element | Nima qiladi | Qachon ishlatiladi |
|---|---|---|
| **code version** | Ishlayotgan kod commit'i | Yangi kod ishlayotganini tekshirish (`git pull` dan keyin) |
| **S5 parser: auto / rules / ollama** | Savolni JSON'ga kim aylantirishini tanlaydi (8.1) | Odatda `auto` |
| **S5 parser in use** | Haqiqatda qaysi biri ishlayotgani | Mac'da `ollama` bo'lishi kerak |
| **Recording** | Qaysi yozuv ko'rsatilishi | Sintetik va real yozuvlar orasida almashish |
| **➕ Add recording** | Yuklash (7-bo'lim) | Yangi bag |
| **⚙️ Manage → Re-run pipeline** | Shu recording'ni pipeline'dan qayta o'tkazadi | Kod yoki parser o'zgargandan keyin |
| **⚙️ Manage → Delete** | Recording'ni o'chiradi (built-in'ni o'chirib bo'lmaydi) | Noto'g'ri import'dan keyin |
| **▶ Play / ⏸ Pause** | Frame'larni avtomatik almashtiradi (`ros2 bag play` analogi) | Jonli kamera effekti |
| **⏮ Start** | Birinchi frame'ga qaytadi | Ko'rsatuvni boshidan boshlash |
| **Seconds per frame** | Play tezligi (0,5 / 1 / 1,5 / 2,5 / 4 s) | Tushuntirishga vaqt kerak bo'lsa, sekinroq |
| **Frame** | Qo'lda frame tanlash (Play paytida bloklangan) | Aniq bir lahzani ko'rsatish |

### 8.1 S5 parser rejimlari
- **auto** — Ollama serveri javob bersa `ollama`, aks holda `rules`.
- **ollama** — lokal LLM (`qwen2.5:3b`). Savolni erkin tilda tushunadi.
- **rules** — oddiy qoidalarga asoslangan parser (`src/integration/rule_parser.py`). U faqat S5 prompt
  misollaridagi gap tuzilmalarini tushunadi: "Where is X", "What is closest to X", "Which X is closest to
  Y", "X that is not near Y", "far from Y". Murakkab gaplarni noto'g'ri tushunishi mumkin.

**Ikkala rejimda ham verifier bir xil.** Parser faqat savolni JSON'ga aylantiradi, javobni esa doim Python
geometriyasi beradi. Shuning uchun parser xato qilsa ham tizim yolg'on javob bermaydi, faqat "Cannot
determine" deydi.

---

## 9. Modul sahifalari: har bir student qismi

Yuqoridagi tugmalar qatori (Overview · S1 · S3 · S2 · S6 Tracking · S4 · S5 · S6 Action · S7) sahifalarni
almashtiradi. Bir vaqtda **faqat tanlangan sahifa** chiziladi, shuning uchun Play tez ishlaydi.

> **Barcha sahifalar joriy frame'ni ko'rsatadi.** Frame almashtirilsa (slider yoki Play), hamma sahifa
> yangi frame bo'yicha o'zgaradi.

### 9.1 Overview — butun tizim bir qarashda
**Nimani ko'rsatadi:**
- **8 ta karta (S1→S7):** har biri joriy frame'da shu modul nima qilganini ko'rsatadi. Masalan
  "S2: 5 ObjectInstance3D · 1 low-depth, 409 ms".
- **Chapda:** kamera rasmi, ustida S3 maskalari va S6 ID'lari (masalan `glass_01 0.71 (transparent_suspected)`).
- **O'ngda:** virtual twin (yuqoridan ko'rinish), oxirgi savolning yo'li va exclusion zone.
- **Pastda:** frame soni, latency, FPS, contract xatolari, parser.

**Nima deyish kerak:** "Bitta frame yetti moduldan o'tdi. Har bir karta o'z modulining chiqishi, JSON
orqali keyingisiga uzatilgan."

**S5/S6 kartalari** oxirgi berilgan savolga bog'liq. Savol berilmagan bo'lsa, standart "Go to the mug"
ko'rsatiladi.

### 9.2 S1 Sensors — Muzaffar
| Element | Ma'nosi | Kutiladigan natija |
|---|---|---|
| RGB | Bag'dan olingan rang rasmi | Real stol |
| Depth heat-map | Masofa (m). Kulrang joylar = depth yo'q | "valid depth: 97% of pixels" kabi |
| Camera intrinsics | `fx, fy, cx, cy, width, height, depth_scale` | Yashil "source: camera_info" yoki sariq "source: fallback" |
| Extrinsics | `T_map_camera` matritsasi, **kamera balandligi** va **pitch burchagi** | Sintetikda 0,600 m va 45,0° |
| TF source | `tf: map <- camera_link` yoki "reaches no map → fallback" | TF bo'lmasa `configs/fusion_3d.yaml` ishlatiladi |
| Recording | Bag nomi, davomiyligi, RGB/depth xabarlari, sinxron juftliklar, tashlab yuborilganlar | `synced` ≈ `rgb` bo'lishi kerak |
| Topics in the bag | Bag'dagi barcha topic'lar, chastotasi, ishlatilgani (✓) | ZED topic'lari ✓ bilan |

**Nima uchun muhim:** professor "har bir bag'ga metadata, calibration sifati" talab qilgan. Bu sahifa
shuni ko'rsatadi. **Sariq "fallback"** chiqsa, haqiqiy kamera parametrlari ishlatilmagan, ya'ni masofalar
taxminiy (13-bo'lim).

### 9.3 S3 Perception — Li Yizhe
| Element | Ma'nosi |
|---|---|
| Rasm | S3'ning **o'z labellari** (S6 ID emas), box va maskalar |
| Jadval | `id`, `label`, `candidates` (muqobil nomlar), `score` (YOLO-World ishonchi), `mask px`, `mask` holati, `split` |
| `split` | `seen` = `desk.json` promptlarida bor (laptop, coffee mug, keyboard, mouse, book, bottle); `unseen` = `unseen_objects.txt` da (soldering iron, digital multimeter, vernier caliper, hot glue gun, portable lidar scanner); `open-vocab` = boshqa nomlar |
| Score grafigi | Har bir aniqlash ishonchi |
| Masks | S2'ga uzatilgan kesimlar |

**Kutiladigan natija (real bag):** 8 ta aniqlash (apple, black smartphone ×2, game controller, white mug ×2,
white handheld device, egg), scorelar 0,80…0,17.

**Muhim:** platforma S3'ni **qayta ishlatmaydi**, Li'ning tayyor natijasini ko'rsatadi (replay rejimi).
YOLO-World serverda yo'q.

### 9.4 S2 3D Fusion — Sarvar
| Element | Ma'nosi | Nimaga qarash kerak |
|---|---|---|
| 3D view: "Scene point cloud + boxes" | Butun depth bulut (rangli) va har bir obyekt 3D boxi | Boxlar obyektlar ustida |
| 3D view: "Object points only" | Faqat har obyekt maskasidan olingan nuqtalar | Obyekt shaklini ko'rish |
| Jadval | `x,y,z` (map, metr), `W×H×D` (sm), `depth conf`, `depth status`, `geometry`, `fused conf`, `points` | |
| `geometry` | `depth` = depth nuqtalaridan; `mask_plane` = depth yaroqsiz, mask va stol tekisligidan | Shisha uchun `mask_plane` |
| Depth confidence grafigi | Yashil = ok, qizil = `transparent_suspected` yoki `no_depth` | Sintetikda glass qizil (0,23) |
| Adaptive fusion weights | `q_*` (sifat) va `w_*` (og'irlik) rgb/depth/semantic uchun. LiDAR = 0 (bizda yo'q) | Shishada `w_depth` kichik |
| ObjectInstance3D JSON | S6/S4'ga uzatilgan to'liq JSON | |

**Ichkarida nima bo'ladi (har frame):**
1. Depth tozalash: 0,15–2,0 m oralig'idan tashqarisi va "flying pixel"lar olib tashlanadi.
2. **Stol tekisligi RANSAC:** stol topiladi, undan 8 mm yaqin nuqtalar obyekt hisoblanmaydi (mask stolga
   chiqib ketgan bo'lsa).
3. Mask ichidagi piksellar 3D'ga: `X=(u−cx)·Z/fx`, `Y=(v−cy)·Z/fy`.
4. Voxel 3 mm → outlier olib tashlash → eng katta DBSCAN klasteri.
5. Centroid va gravitatsiyaga tekislangan box (map frame, z yuqoriga).
6. Ishonchlilik: mask ichida depth bormi, u stol tekisligiga "tushib ketyaptimi" (shishadan o'tib ko'rinish).
   Yaroqli depth 10% dan kam bo'lsa → `mask_plane` fallback.
7. 30 nuqtadan kam bo'lsa, obyekt **rad etiladi** va sababi ko'rsatiladi. NaN geometriya hech qachon chiqmaydi.

### 9.5 S6 Tracking — Erkinov Shakhzodjon
| Element | Ma'nosi |
|---|---|
| Timeline | Qatorlar = persistent ID, ustunlar = frame. Yashil = ko'rindi, to'q sariq = yashirin, lekin xotirada |
| Ko'k chiziq | Joriy frame |
| Trajectories | Har bir ID'ning shu paytgacha bo'lgan yo'li (yuqoridan) |
| Moved since last frame | Har ID oldingi frame'dan qancha siljigani (sm) |
| **Events up to this frame** | APPEARED / MOVED / OCCLUDED / REAPPEARED / PICKED_UP / PLACED_ON / REMOVED (eng yangisi tepada) |

**Qoida (`src/tracking/tracker.py`, MultiTermTracker):** har frame'da tracker'lar va detection'lar
Hungarian usuli bilan juftlanadi. Narx = pozitsiya (tezlik bo'yicha oldindan aytilgan joyga, 15 sm gate) +
label (+ umumiy label nomzodlari) + o'lcham. Yashirin obyekt 10 frame xotirada turadi va **60 sm** gacha
uzoqda qaytsa ham o'sha ID'ni oladi (qo'lda olib, boshqa joyga qo'yish). Ishonch har yashirin frame'da ×0,85.

**Kutiladigan natija (sintetik):** 0-frame'da 5 ta APPEARED. `book_01` 4-frame'da OCCLUDED (sariq), 6-frame'da
REAPPEARED va **o'sha ID** bilan qaytadi. `mug_01` har frame ~1,3 sm siljiydi, ID o'zgarmaydi (u to'xtamagani
uchun MOVED chiqmaydi). Jami 0 ta ID almashinuvi.

**Halol eslatma:** bu kod Claude Code (AI) yozgan reference implementation; S6 uni ko'rib chiqadi
(`docs/s6/README.md`). S6 W5 da ko'rsatgan baseline o'zgarmagan holda `src/integration/s6_placeholder.py` da
turadi va `scripts/s6_tracking_eval.py` ikkalasini solishtiradi.

**Muhim:** yangi kodni olgandan keyin eski recording'lar eski tracker bilan ishlangan bo'ladi va Events jadvali
bo'sh chiqadi. Sidebar → **Re-run pipeline on this recording**.

### 9.6 S4 Scene Graph — Jaloliddin
| Element | Ma'nosi |
|---|---|
| Graf | Tugunlar obyektlar stolda qayerda tursa, o'sha joyda chiziladi (yuqoridan). Kvadrat = virtual `desk_01`, kulrang = yashirin obyekt |
| Relations shown | Ko'rsatiladigan munosabatlar. `left_of` sukut bo'yicha yashirin (juda ko'p) |
| Sichqoncha bilan | Munosabat nomi ustiga olib borilsa, isboti chiqadi: `distance_m`, `vertical_gap_m`, `xy_overlap`, `dx_m` |
| Changes since previous frame | Yashil `+` = yangi munosabat, sariq `−` = tugagan munosabat |
| Edges jadvali | Barcha munosabatlar, isboti, ishonchi, `start (s)` / `end (s)`, holati (active/ended) |

**Qoidalar (`src/scene_graph/rules.py`):**
| Munosabat | Shart | Isbot |
|---|---|---|
| `near` | markazlar orasi < 0,30 m | `distance_m` |
| `on` (A on B) | A pastki qismi B ustki qismiga 3 sm dan yaqin, A yuqorida, A izining ≥ 30% B ustida | `vertical_gap_m`, `xy_overlap` |
| `left_of` | kamera bo'yicha A, B dan 5 sm dan ko'proq chapda | `dx_m` |

Munosabat ishonchi = ikkala obyekt ishonchining **kichigi**. Munosabat tugasa o'chirilmaydi:
`end_time` qo'yiladi va tarix sifatida saqlanadi.

**Kutiladigan natija:** sintetikda `frame_0003` da `laptop_01 near mug_01` paydo bo'ladi (mug yaqinlashdi),
`frame_0007` da `book_01 near mug_01` va `glass_01 near multimeter_01` tugaydi. Real bag'da 9 tugun va 45 faol munosabat (`left_of` 27, `near` 10, `on` 8).

### 9.7 S5 Reasoning — Kurbonaliyev Sanjarbek
**Ish tartibi:** savol → parser (LLM yoki rules) → intent JSON → verifier grafda hisoblaydi → javob, isbot
va ishonch.

Sahifada:
- **Tayyor tugmalar** (grafdan avtomatik tuziladi):
  1. `Where is the <1-obyekt>?` — joylashuv;
  2. `What is closest to the <1-obyekt>?` — eng yaqin;
  3. `Go to the <2-obyekt>` — harakat buyrug'i;
  4. `Find the <obyekt> that is not near the <1-obyekt>` — inkor (negative);
  5. `Where is the portable lidar scanner?` — sahnada **yo'q** obyekt, tizim rad etishi kerak.
- **Chat** — istalgan savol.
- **1 · Parsed intent** — parser chiqargan JSON.
- **2 · Geometric proof** — `verified yes/no`, ishonch, `computed_facts` (masofalar va qoidalar).
- **ReasoningResult JSON** — S6'ga uzatiladi.
- **Query suite** — tayyor savollarni birdaniga sinab, jadval chiqaradi.
- **Earlier questions** — shu frame'dagi oldingi savollar.

**Verifier qaysi savollarni tushunadi** (`src/reasoning/main.py`):
| Savol turi | Shart | Javob | Misol |
|---|---|---|---|
| Joylashuv | faqat `target` | eng ishonchli mos tugun, koordinata, ustida turgan joy | "Where is the mug?" |
| Eng yaqin | `reference` bor | `target` (yoki barcha real obyektlar) orasidan reference'ga eng yaqini | "Which cup is closest to the book?" |
| Inkor | `target` va `negative_constraints` | qoidani buzmaydigan nomzod | "Find the book that is not near the laptop" |
| Inkor + eng yaqin | ikkalasi | qoidadan keyin eng yaqini | "What is closest to the book that is not near the laptop" — sintetikda **Cannot determine**, chunki book'dan boshqa hamma obyekt laptopga yaqin |

**Nom bo'yicha moslash** so'z darajasida: "book" so'zi "notebook computer" ga mos kelmaydi. "red mug" topilmasa,
oxirgi so'z "mug" bilan qidiriladi.

**Kutiladigan natijalar (rules parser):**

| Savol | Sintetik yozuv | Real bag (S1_static_bag) |
|---|---|---|
| Where is the laptop / apple? | `laptop_01`, x=0.72 y=0.17 z=0.10, on desk_01 | `apple_0`, x=0.78 y=−0.25 z=0.04, on desk_01 |
| What is closest to the laptop / apple? | `glass_01`, 0,22 m | `white_mug_1`, 0,28 m + ⚠ low confidence (0,22) |
| Go to the mug | `mug_01` | `white_mug_0` + ⚠ "2 objects match 'mug'", ⚠ low confidence (0,56) |
| Find the book that is not near the laptop | `book_01` | — |
| Find the black smartphone that is not near the apple | — | `black_smartphone_0` |
| Where is the portable lidar scanner? | **Cannot determine** | **Cannot determine** |
| Which chair is closest to the laptop? | **Cannot determine** (stul yo'q) | **Cannot determine** |
| Where is the white mug? | `mug_01` + ⚠ "Could not verify: white" | `white_mug_0` + ⚠ |

Ogohlantirishlar (⚠) nimani bildiradi:
- **"has low confidence (x)"** — tugun ishonchi 0,6 dan past. Javob bor, lekin ehtiyot bo'lish kerak.
- **"N objects match 'X', showing the most confident"** — bir nechta mos obyekt bor, eng ishonchlisi tanlandi.
- **"Could not verify: white"** — rang va shunga o'xshash sifatlar grafda saqlanmaydi, shuning uchun tekshirilmadi.
- **"is occluded"** — obyekt hozir ko'rinmayapti, oxirgi ma'lum joyi ishlatildi.
- **"Could not verify relation: …"** — `on`/`under` kabi munosabat so'ralgan, javob esa faqat masofaga asoslangan.

**Ishonch** = javobda qatnashgan tugunlar ishonchining kichigi.

### 9.8 S6 Action — Erkinov Shakhzodjon
**Oldin S5 sahifasida savol bering.** Bu sahifa oxirgi savolni ishlatadi.

| Element | Ma'nosi |
|---|---|
| Virtual twin | Yuqoridan ko'rinish. Qora kvadrat = agent (start x=0,25 m, y=0), nuqtalar = obyektlar, qizil maydon = exclusion zone |
| Yashil chiziq | Qabul qilingan yo'l (`validated ✓`) |
| Qizil uzuq chiziq | Rad etilgan yo'l (`rejected ✗`) |
| ActionGoal | `target_object_id`, `target_pose [x, y, yaw]`, `validated`, `reason` |
| Virtual-twin simulation | `executed`, `success`, `collisions`, `final_error_m`, `steps` |

**Qoidalar (`src/planning/action.py` — ActionGoal v2, `configs/s7_integration.yaml`):**
1. S5 javobi tasdiqlanmagan bo'lsa → `TARGET_UNVERIFIED`, goal yuborilmaydi.
2. Nishon hozir **yashirin** (occluded) bo'lsa → `TARGET_OCCLUDED`: joyi faqat xotirada, hozir tasdiqlanmagan.
3. Exclusion zone faqat **`laptop`** labelli obyektlar atrofida quriladi (`exclusion_labels: [laptop]`),
   3 sm zaxira bilan. Yashirin obyektlar uchun zone qurilmaydi.
4. Nishon zone **ichida** bo'lsa (masalan, laptopga 3 sm dan yaqin stakan) → `TARGET_IN_EXCLUSION_ZONE`.
5. Agent nishonga **to'g'ri chiziq** bo'ylab boradi (`planner: straight`). Chiziq zone'ni kesib o'tsa →
   `PATH_BLOCKED`. `planner: detour` qilinsa, A* bilan zone'larni aylanib o'tuvchi yo'l qidiriladi (`OK_DETOUR`).
6. Nishon ishonchi 0,3 dan past bo'lsa, goal yuboriladi, lekin `LOW_TARGET_CONFIDENCE` ogohlantirishi chiqadi.
7. Simulatsiya 1 sm qadam bilan yuradi. `success` = zone'ga hech kirmadi va nishonga 5 sm dan yaqin keldi.

**Kutiladigan natijalar (sintetik):** "Go to the mug" → ✓ `OK`. "Go to the glass" va "What is closest to the
laptop?" (javob glass) → ✗ `TARGET_IN_EXCLUSION_ZONE`, chunki stakan laptopning xavfsizlik zonasida turibdi.
**4- yoki 5-frame'da** "Go to the book" → ✗ `TARGET_OCCLUDED` (kitob qo'l ostida). Real bag'da **laptop yo'q**,
shuning uchun zone yo'q va har bir tasdiqlangan, ko'rinib turgan nishon ✓ bo'ladi.

### 9.9 S7 Integration — Mirzokhidbek
| Element | Ma'nosi |
|---|---|
| mean / p95 latency, FPS | Frame boshiga o'rtacha va 95-persentil vaqt, sekundiga frame |
| Mean latency per stage | Qaysi modul qancha vaqt oladi (odatda S2 ~97%) |
| Per-frame latency | Har frame bo'yicha bosqichlar |
| Contract status | Yashil = barcha uzatishlar formatga mos. Qizil = qaysi modul qaysi maydonni buzgani |
| Frame errors / skipped | Xato bergan yoki S3 maskasiz qolgan frame'lar |
| S3 ↔ bag alignment | Maskalar qaysi usulda moslangani (7.2-bo'lim, 10-bosqich) |
| Download | `run_summary.json` va scene graph'ni yuklab olish |
| Reproduce | Xuddi shu natijani buyruq satridan olish buyruqlari |

**Tekshiriladigan 5 ta contract:** S3→S2 (observation), S2→S6/S4 (objects), S4→S5 (scene graph),
S5→S6 (reasoning result), S6→twin (action goal).

---

## 10. Demonstratsiya stsenariysi (qadam-baqadam)

**Tayyorgarlik (taqdimotdan 30 daqiqa oldin):**
1. MacBook'da `git checkout main && git pull` → `./G1_Demo.command`. Yangi kod kelgan bo'lsa, har bir recording'da
   **Re-run pipeline on this recording**.
2. Sidebar'da **code version** va **S5 parser in use: ollama** ni tekshiring.
3. Real bag allaqachon import qilingan bo'lsin (7-bo'lim). Import'ni taqdimot paytida qilmang, bir necha
   daqiqa oladi.
4. S3 sahifasida maskalar real obyektlar ustida turganini tekshiring.
5. Internet bo'lmasa ham ishlaydi. Wi-Fi'ga ishonmang.

**Ko'rsatuv (8–10 daqiqa):**

| # | Kim | Qayerda | Nima qilinadi | Nima deyiladi / kutiladigan natija |
|---|---|---|---|---|
| 1 | S7 | Recording = real bag, **Overview** | ⏮ Start, keyin ▶ Play, 3–4 frame | "Bu ros2 bag play bilan bir xil: S1 yozgan bag frame-ma-frame ijro etilyapti, har frame yetti moduldan o'tadi." Kartalar o'zgaradi |
| 2 | S7 | | ⏸ Pause | Frame to'xtaydi |
| 3 | S1 | **S1 Sensors** | Depth va intrinsics manbasini ko'rsatish | "camera_info'dan olingan intrinsics, TF'dan kamera balandligi" |
| 4 | S3 | **S3 Perception** | Jadval va maskalar | "Training'siz, matn bo'yicha 8 ta obyekt. Scorelar S2/S4 ga ehtiyotkorlik uchun uzatiladi" |
| 5 | S2 | **S2 3D Fusion** | 3D'ni aylantirish, depth confidence grafigi | "Mask + depth → 3D box. Past ishonchli obyektlar (white mug) belgilanadi" |
| 6 | S6 | Recording = **sintetik**, **S6 Tracking** | Timeline va Events jadvali | "Book 2 frame yashirildi (OCCLUDED), o'sha ID bilan qaytdi (REAPPEARED); mug 9 sm siljidi, ID o'zgarmadi" |
| 7 | S4 | Real bag, **S4 Scene Graph** | Munosabat ustiga sichqoncha olib borish | "Har bir edge o'z isbotiga ega: distance_m, vertical_gap_m" |
| 8 | S4 | Sintetik, Frame slider 2 → 3 | "Changes since previous frame" | `+ laptop_01 near mug_01` paydo bo'ladi |
| 9 | S5 | Real bag, **S5 Reasoning** | "What is closest to the apple?" | Javob, masofa isboti, past ishonch ogohlantirishi |
| 10 | S5 | | "Where is the portable lidar scanner?" | **Cannot determine**: "LLM taxmin qilmaydi, geometriya isbotlay olmasa rad etadi" |
| 11 | S6 | Sintetik, S5'da "Go to the glass" → **S6 Action** | Twin | **rejected ✗** `TARGET_IN_EXCLUSION_ZONE` — stakan laptopning xavfsizlik zonasida |
| 12 | S6 | S5'da "Go to the mug" → S6 Action | | **validated ✓** `OK`, `success: true` |
| 12b | S6 | Sintetik, **Frame = 4** (⏸), S5'da "Go to the book" → S6 Action | | **rejected ✗** `TARGET_OCCLUDED` — "kitob hozir ko'rinmayapti, joyi faqat xotirada; robotni xotiradagi joyga yubormaymiz" |
| 13 | S7 | **S7 Integration** | Latency va contract | "0 contract xatosi; S2 vaqtning ~97% ini oladi, optimizatsiya rejada" |

**Muhim nozik joylar:**
- **Savol tarixi har bir frame uchun alohida.** Frame almashtirsangiz (yoki Play bossangiz), S5/S6 standart
  "Go to the mug" savoliga qaytadi. Avval frame'ni tanlang, keyin savol bering.
- Play paytida Frame slider bloklanadi. Avval ⏸ bosing.
- S6 Action sahifasi oxirgi savolni ko'rsatadi, shuning uchun avval S5'da savol bering.

---

## 11. "Nima uchun shunday natija chiqdi?" — savol-javob

**"Cannot determine" chiqdi, bu xatomi?**
Yo'q, bu dizayn. Verifier faqat isbotlay oladigan javobni beradi. Sabablari: obyekt grafda yo'q ("lidar
scanner"); nishon aytilmagan ("Find an object that…"); hech bir nomzod qoidaga mos emas ("Find a mug that is
not near the laptop": sintetikda mug laptopga yaqin); nishon va reference bir xil obyekt.

**Nega "white" tekshirilmadi?**
Graf ranglarni saqlamaydi. Parser "white" so'zini ajratadi, verifier esa uni tekshira olmasligini ochiq
aytadi. Professorga aynan shu ko'rsatiladi: tizim bilmagan narsasini yashirmaydi.

**Nega real bag'da hamma yo'l "validated"?**
Exclusion zone faqat laptop atrofida, real bag'da esa laptop yo'q. Rad etishni ko'rsatish uchun sintetik
yozuvdan foydalaning ("Go to the glass").

**Nega sintetikda glass `transparent_suspected` va depth confidence 0,23?**
Shisha orqali stereo kamera stolni ko'radi, ya'ni depth shishaga emas, uning orqasiga tegishli. S2 buni
aniqlaydi va geometriyani mask + stol tekisligidan quradi (`geometry: mask_plane`).

**Nega `left_of` sukut bo'yicha yashirin?**
Har juft obyekt uchun chiqadi (real bag'da 27 ta), grafni to'ldirib yuboradi. "Relations shown" orqali yoqiladi.

**Nega Overview'dagi latency 0,4–0,8 s/frame, S2'da esa real bag'da ~0,1 s?**
Sintetik sahna 640×480, zich nuqtalar va 5 obyektdan iborat. Bundan tashqari, Overview'dagi raqam butun
frame vaqti (S1+S3+S2+S6+S4). Har ikki raqam o'lchangan va to'g'ri.

**Nega `desk_01` "closest" javobida chiqmaydi?**
Desk virtual tugun (S2 stolni obyekt sifatida olib tashlaydi). Savolda aniq "desk" so'ralmasa, u nomzod
bo'lmaydi.

**Nega S3 sahifasidagi labellar va Overview'dagi nomlar farq qiladi?**
S3 sahifasi S3'ning **xom labellari**ni ko'rsatadi (`white mug`). Overview esa S6 bergan **persistent
ID**larni (`white_mug_0`).

**Ollama va rules natijalari farq qiladimi?**
Parser JSON'i farq qilishi mumkin: LLM murakkab gaplarni yaxshiroq tushunadi. Javobni esa ikkala holatda
bir xil verifier hisoblaydi, shuning uchun bir xil intent JSON → bir xil javob.

---

## 12. Muammolar va yechimlar

| Belgi | Sabab | Yechim |
|---|---|---|
| `Library not loaded: …libusb-1.0.0.dylib` | Mac'da Homebrew libusb yo'q | `brew install libusb libomp` |
| `brew: command not found` (o'rnatilgandan keyin) | PATH yangilanmagan | `eval "$(/opt/homebrew/bin/brew shellenv zsh)"` |
| `Port 8501 is not available` | Eski demo ishlab turibdi | `lsof -ti tcp:8501 \| xargs kill`, keyin qayta. Yangi launcher buni o'zi qiladi |
| Tuzatishdan keyin ham eski xato | Eski kod ishlayapti | Ctrl+C → `git pull` → qayta ishga tushirish. Sidebar'dagi **code version** ni tekshiring |
| `no synchronized RGB/depth pair … exported: 0` | Eski kod (vaqt moslash bug'i) | 7.2-bo'lim, 10-bosqich. `69cf76e` yoki undan keyingi commit kerak |
| "paired by order: CHECK…" | S3 vaqtlari bag'ga mos kelmadi | S3 sahifasida maskalarni tekshiring. Siljigan bo'lsa, Li'dan qaysi frame'larda ishlaganini so'rang |
| "no frame has S3 masks…" | S3 zip yuklanmagan yoki juftlanmagan | S3 zip'ni yuklang. Bir nechta bag bo'lsa, nomlash qoidasi (6.2) |
| "Railpack could not determine how to build" | Railway eski (Dockerfile'siz) commit'ni build qilyapti | Settings → Source → Branch = `main`, keyin Redeploy |
| Railway'da repo ko'rinmaydi | Repo `intodakt` akkauntida | Fork qiling yoki `intodakt` Railway'ga ruxsat bersin |
| Cloud Run'da yuklash xatosi | 32 MB cheklovi | Google Drive linki |
| `S5 parser in use: rules` (Mac'da) | Ollama ishlamayapti | Ollama ilovasini oching, launcher'ni qayta ishga tushiring |
| S5'da "the question could not be parsed" | Ollama xato qaytardi | Parser'ni **rules** ga o'tkazing |

---

## 13. Cheklovlar: professorga halol aytilishi kerak bo'lganlar

1. **S3 replay.** Platforma YOLO-World'ni jonli ishlatmaydi, S3'ning saqlangan natijasini ko'rsatadi.
   Jonli rejim `WITH_S3_LIVE=1` bilan mumkin, lekin sinalmagan.
2. **S1 / S3 / S6 dagi yangi kod — AI yozgan reference implementation.** S6 tracker va ActionGoal v2, S1
   kalibrlash / static TF / session report, S3 alias va baholash vositalari Claude Code bilan yozilgan; egalari
   ko'rib chiqadi. S1 va S6 kodi faqat sintetik ma'lumotda tekshirilgan (`docs/s1/`, `docs/s6/`).
3. **Exclusion zone faqat laptop.** Konfiguratsiyada o'zgartiriladi (`exclusion_labels`).
4. **Rules parser oddiy.** Bulutda Ollama yo'q, murakkab gaplar uchun Mac'dagi LLM kerak.
5. **Temporal savollar yo'q.** S6 endi hodisalarni (`MOVED`, `OCCLUDED`, ...) va har bir ID tarixini yozadi,
   lekin S5 hali "where was the mug before?" kabi savollarni qabul qilmaydi (W11).
6. **Intrinsics fallback nominal.** Bag'da `camera_info` bo'lmasa, ZED Mini datasheet qiymatlari ishlatiladi.
   S1 haqiqiy qiymatlarni qo'yishi kerak.
7. **Ba'zi metrikalar sintetik ma'lumotda o'lchangan** (11/11, 7/7, 2/2, 0 ID switch). Real bag'dagi
   metrikalar alohida: 8/8 obyekt, 0,95 sm drift, ~96 ms.
8. **Ground truth faqat sintetikda.** Real bag uchun lenta bilan o'lchangan joylar W12 rejasida.

---

## 14. Ilova: fayllar, buyruqlar, sozlamalar

### 14.1 Asosiy fayllar
| Fayl | Vazifasi |
|---|---|
| `demo/app.py` | Veb-ilova (barcha sahifalar) |
| `demo/runs.py` | Recording'lar ombori, import, pipeline ishga tushirish, built-in yozuv, URL/Drive |
| `demo/pack.py` | Katta bag + S3 → kichik zip |
| `G1_Demo.command` | MacBook launcher |
| `src/sensors/bag_replay.py` | S1 replay: bag o'qish, topic aniqlash, intrinsics, TF, vaqt bo'yicha frame olish |
| `src/integration/demo_data.py` | Yuklangan fayllarni topish, juftlash, S3 vaqtini moslash, maskalarni ko'chirish |
| `src/integration/pipeline.py` | S1→S3→S2→S6→S4 pipeline, S5→S6 savol, vaqt o'lchash, contract tekshiruvi |
| `src/integration/contracts.py` | 5 ta JSON contract tekshiruvi |
| `src/integration/s5_adapter.py`, `rule_parser.py` | S5 kodini jonli grafda chaqirish, offline parser |
| `src/tracking/tracker.py` | S6: MultiTermTracker (ID, holat, hodisalar) |
| `src/planning/action.py` | S6: ActionGoal v2, reason code'lar, detour planner, virtual twin simulatsiyasi |
| `src/integration/s6_placeholder.py` | S6 Week-5 baseline (solishtirish uchun) |
| `src/sensors/calibration.py`, `static_tf.py`, `session_report.py` | S1: kalibrlash, o'lchangan TF, FPS/sync/depth hisoboti |
| `src/open_vocab/evaluate.py`, `aliases.py` | S3: seen/unseen P/R/F1, alias jadvali |
| `src/fusion_3d/` | S2: 3D, depth ishonchliligi, fusion |
| `src/scene_graph/` | S4: graf va qoidalar |
| `src/reasoning/` | S5: parser (Ollama) va verifier |
| `configs/s7_integration.yaml` | Pipeline, tracking, action sozlamalari |
| `configs/fusion_3d.yaml` | S2 sozlamalari (depth oralig'i, RANSAC, tozalash, TF fallback) |
| `configs/sensors/camera_intrinsics.yaml` | S1 fallback intrinsics |

### 14.2 Buyruqlar (brauzersiz, xuddi shu kod)
```bash
python -m demo.runs                              # recording'lar ro'yxati
python -m demo.runs --builtin                    # sintetik yozuvni qayta yaratish
python -m demo.runs --import G1_bags.zip         # import + pipeline
python -m demo.runs --url <Drive link>           # linkdan import
python -m demo.runs --rerun S1_static_bag --parser ollama
python -m demo.pack --src recordings/ --out G1_bags.zip
python -m pytest -q && python tests/smoke_test.py   # testlar
python main_pipeline.py --bag <bag> --camera-tf auto  # S1: TF'ni stol tekisligidan o'lchab ishlatish
python -m src.sensors.session_report --bag <bag> --out reports/<bag>/   # S1: FPS, sync, depth sifati
python scripts/s6_tracking_eval.py                    # S6: baseline vs yangi tracker
python scripts/s2_week9_ablation.py                   # S2: F1 vs F2 ablation
```
Mac'da `python` o'rniga `.venv/bin/python` yozing (launcher yaratgan muhit).

### 14.3 Asosiy sozlamalar
| Parametr | Qiymat | Fayl |
|---|---|---|
| RGB/depth sinxron farqi | ≤ 20 ms | `bag_replay.py` (`max_dt`) |
| Max frames per bag | 40 (5–120) | sidebar |
| Depth oralig'i | 0,15–2,0 m | `configs/fusion_3d.yaml` |
| Stol tekisligidan zaxira | 8 mm | `configs/fusion_3d.yaml` |
| Obyekt uchun min nuqta | 30 | `configs/fusion_3d.yaml` |
| `near` | < 0,30 m | `src/scene_graph/rules.py` |
| `on` | gap < 3 sm, overlap ≥ 30% | `src/scene_graph/rules.py` |
| `left_of` | > 5 sm (kamera) | `src/scene_graph/rules.py` |
| Tracking gate / xotira / qayta tanish | 15 sm / 10 frame / 60 sm | `configs/s7_integration.yaml` |
| Planner | `straight` (yoki `detour`) | `configs/s7_integration.yaml` |
| Past ishonch ogohlantirishi (S6) | 0,3 | `configs/s7_integration.yaml` |
| Past ishonch chegarasi (S5) | 0,6 | `src/reasoning/verifier.py` |
| Agent start | (0,25; 0,00) m | `configs/s7_integration.yaml` |
| Exclusion | laptop, +3 sm | `configs/s7_integration.yaml` |
| Nishonga yetish toleransi | 5 sm | `configs/s7_integration.yaml` |
| Standart savol | "Go to the mug" | `configs/s7_integration.yaml` |
| LLM modeli | `qwen2.5:3b` (`S5_MODEL`) | `src/reasoning/query_parser.py` |

---

## 15. Midterm mashqi: har bir harakatni tekshirish ro'yxati

Midtermdan oldin shu ro'yxatni **boshidan oxirigacha bir marta** bajaring. Har bir qatorda: nima qilinadi,
nima chiqishi kerak (✓ mezon). Biror qator ✓ bo'lmasa, 12-bo'limga qarang yoki Terminal matnini Claude'ga yuboring.
"Sintetik" qiymatlar aniq (kod bir xil natija beradi); "real bag" qiymatlari taxminiy, sizning bag'ingizga bog'liq.

### 15.0 Tayyorgarlik
| # | Harakat | ✓ mezon |
|---|---|---|
| 0.1 | `cd ~/physical_ai_scene && git checkout main && git pull` | `Already up to date` yoki yangi fayllar ro'yxati, xatosiz |
| 0.2 | `./G1_Demo.command` | Brauzerda `http://localhost:8501` ochiladi |
| 0.3 | Sidebar: **code version** | `main` dagi oxirgi commit (GitHub'dagi bilan bir xil) |
| 0.4 | Sidebar: **S5 parser in use** | `ollama` (Ollama ilovasi ochiq bo'lsa) |
| 0.5 | Har bir recording'ni tanlab **Re-run pipeline on this recording** | Progress tugaydi, xato yo'q. Shundan keyin S6 Events jadvali to'ladi |
| 0.6 | Wi-Fi'ni o'chirib, sahifani yangilang | Hammasi ishlayveradi (internet kerak emas) |

### 15.1 Sintetik yozuv (`demo_synthetic_desk`) — aniq natijalar
| # | Sahifa | Harakat | ✓ mezon |
|---|---|---|---|
| 1.1 | Sidebar | ⏮ Start → ▶ Play → ⏸ | Frame 0 → 7 gacha o'tadi va to'xtaydi |
| 1.2 | Overview | Frame 0 | 7 ta karta, har birida qiymat; contract xatosi yo'q |
| 1.3 | S1 Sensors | Frame 0 | intrinsics `source: camera_info`; kamera balandligi **0,600 m**, pitch **45,0°** |
| 1.4 | S3 Perception | Frame 0 | 5 ta detection: laptop, mug, book, multimeter, glass; maskalar obyektlar ustida |
| 1.5 | S2 3D Fusion | Frame 0 | 5 ta 3D box; glass qizil (past depth ishonchi) |
| 1.6 | S6 Tracking | Frame 7 | Timeline: `book_01` 4–5 da sariq; Events: 5 × APPEARED, `book_01` OCCLUDED (4), REAPPEARED (6); `_02` ID yo'q |
| 1.7 | S4 Scene Graph | Frame 2 → 3 | "Changes since previous frame": `+ laptop_01 near mug_01` |
| 1.8 | S5 Reasoning | Frame 2, "Go to the mug" | `mug_01`, verified |
| 1.9 | S5 | Frame 2, "Find a cup that is not near the laptop" | `mug_01 is a cup that is NOT near the laptop` |
| 1.10 | S5 | Frame 7, xuddi shu savol | **Cannot determine** (mug endi laptop yonida) |
| 1.11 | S5 | "Where is the portable lidar scanner?" | **Cannot determine**: not in the scene graph |
| 1.12 | S6 Action | Frame 2, "Go to the mug" | ✓ `OK`, `success: true`, yashil yo'l |
| 1.13 | S6 Action | Frame 2, "Go to the glass" | ✗ `TARGET_IN_EXCLUSION_ZONE`, qizil zone laptop atrofida |
| 1.14 | S6 Action | **Frame 4**, "Go to the book" | ✗ `TARGET_OCCLUDED` |
| 1.15 | S7 Integration | — | contract status yashil, 0 xato; S2 eng ko'p vaqt oladi |

### 15.2 Real bag (masalan `S1_static_bag`)
| # | Sahifa | Harakat | ✓ mezon |
|---|---|---|---|
| 2.1 | Sidebar | Recording = real bag | Ro'yxatda bor (oldin import qilingan) |
| 2.2 | Overview | ▶ Play | Har frame'da ~8 obyekt; "paired by order" ogohlantirishi **yo'q** |
| 2.3 | S1 Sensors | — | intrinsics manbasi (`camera_info` yoki sariq `fallback`) va TF manbasi ko'rinadi. Fallback bo'lsa, videoda shuni ayting |
| 2.4 | S3 Perception | 2–3 frame | Maskalar real obyektlar ustida (siljimagan) |
| 2.5 | S2 3D Fusion | — | 8 ta box; white mug va controller past depth ishonchi (0,3–0,5) |
| 2.6 | S6 Tracking | oxirgi frame | Har obyektga **bitta** ID (8 ta), `_02` yo'q; Events'da faqat APPEARED (sahna statik) |
| 2.7 | S4 Scene Graph | — | `on desk_01` va `near` edge'lar, har birida isbot |
| 2.8 | S5 | preset savollar (har birini bosing) | Har biri javob yoki "Cannot determine"; past ishonchli obyektda ogohlantirish |
| 2.9 | S6 Action | "Go to the …" (har qanday real obyekt) | ✓ `OK` (laptop yo'q → zone yo'q) |
| 2.10 | S7 | — | 0 contract xatosi, frame errors bo'sh |

### 15.3 Buyruq satridan (S1, S2, S6 yangi vositalar) — Terminal'da, `~/physical_ai_scene` ichida
Avval bir marta: `~/.local/bin/uv pip install --python .venv/bin/python matplotlib pytest`

| # | Buyruq | ✓ mezon |
|---|---|---|
| 3.1 | `.venv/bin/python -m pytest -q` | `110 passed` (yoki ko'proq), `failed` yo'q |
| 3.2 | `.venv/bin/python tests/smoke_test.py` | `SMOKE TEST PASSED` |
| 3.3 | `.venv/bin/python main_pipeline.py` | oxirida `contract errors: 0`, `validated=True` |
| 3.4 | `.venv/bin/python -m src.sensors.static_tf --frames data/demo_store/datasets/<real bag>/frames --out reports/<real bag>/` | **S1 uchun birinchi real o'lchov:** kamera balandligi va pitch chiqadi. S2 avval ≈ 0,32 m va ≈ 52° ni ko'rgan; shunga yaqin bo'lsa ✓. `vs_config` placeholder 0,80 m bilan farqni ko'rsatadi |
| 3.5 | `.venv/bin/python -m src.sensors.session_report --bag <bag yo'li> --frames data/demo_store/datasets/<real bag>/frames --out reports/<real bag>/` | `session_meta.json` (FPS ≈ 30?, RGB↔depth farqi ms) va `depth_quality.csv` (har obyekt uchun valid depth %) |
| 3.6 | `.venv/bin/python scripts/s6_tracking_eval.py` | jadval: relocation'da baseline 1 ID switch, yangi tracker 0 |
| 3.7 | `.venv/bin/python scripts/s2_week9_ablation.py && .venv/bin/python scripts/s2_week9_ablation.py --plot` | p95: F1 7,74 / depth only 5,26 / adaptive 1,04 sm; `docs/s2/week9/` da 2 ta rasm |

3.4 va 3.5 natijalarini S1 (Muzaffar) o'z hujjati va videosiga **o'zining real o'lchovi** sifatida qo'shishi
mumkin — buyruqni o'zi ishga tushirsin. `reports/` papkasi git'ga qo'shilmaydi.

### 15.4 Video yozish mashqi
| # | Harakat | ✓ mezon |
|---|---|---|
| 4.1 | OBS / Cmd+Shift+5 bilan 1 daqiqa sinov | Ovoz aniq, ekran o'qiladi |
| 4.2 | Skript (2-monitor) bo'yicha birinchi 3 blokni o'qing | Vaqt rejaga yaqin (blok 1–3 ≈ 2:45) |
| 4.3 | Demo ko'rsatiladigan blokda (skriptda "demo platforma" deb yozilgan) platformaga o'tish | Kerakli sahifa oldindan ochiq, Alt/Cmd+Tab bilan o'tiladi |

