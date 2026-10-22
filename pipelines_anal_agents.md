> **Инструкция для AI-агента:**
> Реализуй описанные ниже модули строго по шагам. Твоя задача — добавить поддержку трёх мультимодальных пайплайнов (Сцена 1: Лицо+Звук, Сцена 2: Лицо+Звук+Тело, Сцена 3: Звук+Тело) с 3-уровневым распознаванием эмоций по языку тела и голосу, **не сломав ни один из существующих проверочных скриптов**:
> * `python scripts/emotion_cache_check.py`[cite: 20]
> * `QT_QPA_PLATFORM=offscreen python scripts/presets_check.py`[cite: 21]
> * `QT_QPA_PLATFORM=offscreen python scripts/streams_ui_check.py`[cite: 22]
> * `python scripts/e2e_check.py`[cite: 19]

---

## 1. Жёсткие архитектурные инварианты (НЕ НАРУШАТЬ!)

1. **Неприкосновенность `CurrentYoloVitModel` (`mood_base/models/current_yolo_vit.py`):**
   * Класс `CurrentYoloVitModel`, функцию `_get_emotion_pipeline`, кэш `_EMOTION_PIPELINE_CACHE` и ключ пайплайна `"yolo_vit"` в `app/core/pipeline_registry.py` оставляем без изменений[cite: 18, 20, 21].
2. **Совместимость с UI-рендером и `scripts/try.py` через `FrameResult.detections`:**
   * UI-плеер (`VideoWorker`[cite: 19]) и `scripts/try.py`[cite: 23] рисуют рамки в цикле `for det in result.detections:`, обращаясь к `det.bbox`, `det.track_id`, `det.emotion.value` и `f"{det.emotion_confidence:.2f}"`[cite: 23].
   * В `FaceDetection` поля `emotion` и `emotion_confidence` **НИКОГДА не должны быть `None`**: дефолт при отсутствии эмоции — `Emotion.UNKNOWN` и `0.0`[cite: 12, 18].
   * В **Сцене 2 (когда человек отвернулся и лица не видно)** и в **Сцене 3 (где детектор лиц отключён)** каждый найденный трек тела должен возвращаться в `FrameResult.detections` как прокси-`FaceDetection` (с `bbox` тела, `track_id` тела и распознанной по языку тела и голосу эмоцией `Emotion`), чтобы штатный плеер приложения отображал рамки и эмоции без переписывания UI[cite: 12, 19, 23].
3. **Совместимость с `CreatePresetDialog` и `PresetManager` (`scripts/presets_check.py`):**
   * В `app/core/pipeline_registry.py` словарь `defaults` и параметры `__init__` всех новых моделей должны содержать **только примитивные типы** (`str`, `int`, `float`, `bool`)[cite: 21].
4. **Паттерн загрузки HuggingFace для `transformers 5.x` и многопоточности:**
   * Все HF-модели (мимика, эмоции по телу Уровня 3, эмоции по голосу) обязаны загружаться через потокобезопасный процессный кэш с `threading.Lock()`, как в `current_yolo_vit.py`[cite: 18]:
     * Загружать модель и процессор явно через `from_pretrained(..., local_files_only=offline)` и передавать готовые объекты в `pipeline(...)` (в `transformers 5.19.0` передача `local_files_only` напрямую в `pipeline()` вызывает `TypeError`)[cite: 17, 18].
     * Сначала пробовать `offline=True`, и только при исключении делать `offline=False`[cite: 18].
5. **Паттерн кадровой оптимизации:**
   * Сжимать кадр перед YOLO через `cv2.resize` на `frame_scale` (`0.5`), пересчитывая координаты боксов и скелетных точек обратно в исходное разрешение[cite: 18].
   * Тяжёлые нейросетевые классификаторы (ViT для лица, классификатор тела Уровня 3 и аудио-модель) вызывать только при `frame_id % analyze_every_n_frames == 0` (по умолчанию раз в 5 кадров), кэшируя результаты по `track_id` и очищая исчезнувшие `stale_tid`[cite: 18].

---

## 2. Пошаговый план разработки

### Шаг 1. Расширение `contracts/schemas.py`

1. Сохранить существующие классы `Emotion`, `BoundingBox`, `FaceDetection`, `FrameResult`[cite: 12].
2. В `BoundingBox` добавить `@property` методы: `width`, `height`, `center_x`, `center_y`[cite: 12].
3. Добавить `BodyPostureTag(str, Enum)` — биомеханические маркеры позы клиента (Уровень 1 и Уровень 2):
   * `NEUTRAL_POSTURE = "neutral_posture"` (спокойная открытая поза)
   * `AGITATED_GESTURES = "agitated_gestures"` (активное разведение/взмахи руками при споре)
   * `HANDS_ON_HIPS = "hands_on_hips"` (руки в боки — претензия, агрессивное ожидание)
   * `ARMS_CROSSED = "arms_crossed"` (скрещенные руки на груди — закрытость, недовольство)
   * `HANDS_TO_HEAD = "hands_to_head"` (руки у головы/лица — стресс, возмущение, фрустрация)
   * `SLOUCHED_HEAD_DOWN = "slouched_head_down"` (опущенная голова и сведённые плечи — подавленность, грусть)
   * `LEANING_FORWARD = "leaning_forward"` (резкий наклон корпуса вперёд через стойку к сотруднику)
4. Добавить `Keypoint2D(BaseModel)` с полями `x: float`, `y: float`, `conf: float`.
5. Добавить `BodyDetection(BaseModel)`:
   * `track_id: Optional[int] = None`
   * `bbox: BoundingBox`
   * `body_confidence: float = Field(ge=0.0, le=1.0)`
   * `keypoints: List[Keypoint2D] = Field(default_factory=list)`
   * `kinetic_energy: float = Field(default=0.0, ge=0.0, le=1.0)` (Уровень 1: скорость рук/корпуса)
   * `posture_tag: BodyPostureTag = BodyPostureTag.NEUTRAL_POSTURE` (Уровень 2: биомеханика позы)
   * `body_emotion: Emotion = Emotion.UNKNOWN` (Уровень 3: итоговая эмоция по языку тела)
   * `body_emotion_confidence: float = Field(default=0.0, ge=0.0, le=1.0)`
   * `face: Optional[FaceDetection] = None` (привязанное лицо для Сцены 2)
6. Добавить `AudioSegmentResult(BaseModel)`:
   * `speech_detected: bool = False`[cite: 4]
   * `rms_energy: float = 0.0`
   * `is_loud_spike: bool = False` (крик, удар по стойке, хлопок)[cite: 1, 10]
   * `audio_emotion: Emotion = Emotion.UNKNOWN`
   * `audio_confidence: float = 0.0`
7. Добавить `TrackAffectState(BaseModel)`:
   * `track_id: int`[cite: 4]
   * `final_emotion: Emotion = Emotion.UNKNOWN`
   * `negative_score: float = Field(ge=0.0, le=1.0)`
   * `dominant_modality: str = "face"` (`"face"`, `"body"`, `"audio"`)
   * `is_negative_alert: bool = False`
8. В `FrameResult` добавить опциональные поля с дефолтами:
   * `bodies: List[BodyDetection] = Field(default_factory=list)`
   * `audio: Optional[AudioSegmentResult] = None`
   * `track_events: List[TrackAffectState] = Field(default_factory=list)`

---

### Шаг 2. Трёхуровневый модуль распознавания языка тела (`mood_base/body_detect.py`)

Создать класс `YoloBodyPoseDetector`, реализующий все 3 уровня анализа языка тела клиента:

1. **Базовая детекция и трекинг скелета:**
   * Использует `YOLO(model_path)` с весами `yolov8n-pose.pt` в режиме `.track(persist=True)`[cite: 7, 14, 18].
   * Возвращает рамку тела `bbox`, `track_id` и 17 ключевых точек COCO (индексы: `0` — нос, `1,2` — глаза, `3,4` — уши, `5,6` — плечи, `7,8` — локти, `9,10` — запястья, `11,12` — бёдра).

2. **Уровень 1 — Кинематика жестов (`kinetic_energy`):**
   * Хранит очередь `deque(maxlen=10)` координат запястий `(lw_x, lw_y, rw_x, rw_y)` и центра плеч по каждому `track_id` (с удалением ушедших `stale_tid`[cite: 18]).
   * Вычисляет среднее межкадровое смещение запястий относительно корпуса, делённое на высоту рамки тела `bbox.height` (нормировка на дистанцию), и приводит к диапазону `[0.0, 1.0]`.

3. **Уровень 2 — Биомеханика позы и постуральные маркеры (`BodyPostureTag`):**
   На основе взаимного расположения 17 точек скелета (нормированных на ширину плеч $W_{\text{shoulders}} = \Vert{}P_5 - P_6\Vert{}$ и длину торса $H_{\text{torso}} = \Vert{}\frac{P_5+P_6}{2} - \frac{P_{11}+P_{12}}{2}\Vert{}$) вычисляет позу клиента:
   * `HANDS_TO_HEAD` (стресс/фрустрация/испуг — `FEAR`/`ANGRY`): расстояние от левого или правого запястья (`9, 10`) до носа/ушей (`0, 3, 4`) меньше $0.6 \cdot W_{\text{shoulders}}$.
   * `ARMS_CROSSED` (закрытое недовольство — `DISGUST`/`ANGRY`): оба запястья находятся в зоне груди между линией плеч и талией, причём левое запястье ближе к правому локтю (или центру груди), а правое — к левому локтю.
   * `HANDS_ON_HIPS` (вызов/претензия в очереди — `ANGRY`): запястья находятся вблизи бёдер (`11, 12`), а локти (`7, 8`) разведены в стороны шире линии плеч ($\vert{}x_7 - x_8\vert{} > 1.3 \cdot W_{\text{shoulders}}$).
   * `SLOUCHED_HEAD_DOWN` (подавленность/усталость/грусть — `SAD`): вертикальная дистанция от носа (`0`) до линии плеч (`5, 6`) сокращена ниже $0.25 \cdot H_{\text{torso}}$ при низкой скорости жестов (`kinetic_energy < 0.15`) и опущенных вниз запястьях.
   * `LEANING_FORWARD` (агрессивное нависание над стойкой — `ANGRY`): резкое сокращение проекции торса $H_{\text{torso}}$ относительно ширины плеч в сочетании с выдвижением плеч/рук вперёд и `kinetic_energy > 0.25`.
   * `AGITATED_GESTURES` (возмущённая жестикуляция — `ANGRY`): `kinetic_energy > 0.38` или взмахи руками выше уровня плеч.

4. **Уровень 3 — Нейросетевой классификатор эмоций по кропу тела и вектору скелета (`BodyEmotionClassifier`):**
   * Вызывается раз в `analyze_every_n_frames` кадров (по умолчанию каждые 5 кадров)[cite: 18] для каждого подтверждённого `track_id` тела.
   * **Архитектура Уровня 3 (двухконтурная):**
     * *Визуальный контур (Image/Context Branch):* вырезает из полного кадра кроп человека `frame[y1:y2, x1:x2]` и подаёт в кэшированный в памяти процесса (`_BODY_PIPELINE_CACHE` + `threading.Lock()`, аналогично `_get_emotion_pipeline` в `current_yolo_vit.py`[cite: 18]) классификатор из Hugging Face (по умолчанию `model_id` задаётся в параметрах пайплайна; поддерживает любую `AutoModelForImageClassification` для кропа фигуры/контекста)[cite: 18].
     * *Постурально-векторный контур (Geometric Pose Head):* преобразует 17 точек скелета в инвариантный к масштабу 12-мерный вектор углов суставов и пропорций (углы в локтях, углы подъёма плеч, наклон головы, симметрия рук, `kinetic_energy`) и комбинирует его логиты с визуальным контуром для получения финального распределения по 7 классам `Emotion` (`angry`, `sad`, `fear`, `disgust`, `surprise`, `happy`, `neutral`)[cite: 12] и `body_emotion_confidence`.
   * Результат кэшируется в `self._last_body_emotions[track_id]` на промежуточных кадрах[cite: 18].

---

### Шаг 3. Модуль аудио-аналитики (`mood_base/audio_branch.py`)

Создать класс `AudioAffectAnalyzer` для распознавания эмоций по голосу:
1. **Процессный кэш `_AUDIO_PIPELINE_CACHE` + `_AUDIO_PIPELINE_LOCK`:**
   * Реализовать `_get_audio_pipeline(model_id, device, dtype)` через явный вызов `AutoModelForAudioClassification.from_pretrained(..., local_files_only=offline)` и `AutoFeatureExtractor.from_pretrained(..., local_files_only=offline)` с фоллбеком `offline=True` $\rightarrow$ `offline=False`[cite: 18].
2. **Чтение аудио и фильтр тишины (VAD + Peak Gate):**
   * Если источник видео содержит аудиодорожку (или передан `audio_chunk`), извлекать окно звука длительностью 1.5 с (16 кГц, моно)[cite: 1, 9]. Если видео без звука — возвращать `AudioSegmentResult(speech_detected=False)` без ошибок.
   * Вычислять громкость `rms` и пик `peak`. Если `rms < vad_rms_threshold` (`0.015`) и `peak < loud_spike_threshold` (`0.18`) — нейросеть не вызывать[cite: 7].
   * При наличии голоса вызывать HF-пайплайн раз в `analyze_every_n_frames` кадров[cite: 18] и приводить ответ к перечислению `Emotion` (`angry`, `sad`, `fear`, `disgust`, `happy`, `surprise`, `neutral`)[cite: 12].

---

### Шаг 4. Модуль слияния модальностей и временного окна (`mood_base/fusion_engine.py`)

Создать класс `MultimodalFusionEngine`:
1. **Привязка Лица к Телу (для Сцены 2):**
   * Сопоставляет `FaceDetection` (от `yolov8n-face` без трекера) с `BodyDetection` (от `yolov8n-pose` с трекером) по попаданию центра лица в верхнюю половину рамки тела (`y_min .. y_min + 0.55 * height`).
   * Присваивает лицу `face.track_id = body.track_id` и сохраняет его в `body.face`.
2. **Фильтр качества лица (`Quality Gate`):**
   * Если высота лица `< min_face_px` (`64 px`)[cite: 8] или кроп размыт (дисперсия Лапласиана $< 35.0$)[cite: 2, 7], классификатор мимики лица не вызывается, выставляется `passed_quality_gate = False` и `emotion = Emotion.UNKNOWN`[cite: 1, 12].
3. **Ассоциация «Звук $\leftrightarrow$ Трек человека»:**
   * Если в кадре несколько людей, голосовой результат `AudioSegmentResult` привязывается к тому `track_id`, у которого в текущем окне максимальная жестикуляция (`kinetic_energy`) или выраженная поза недовольства (`posture_tag != NEUTRAL_POSTURE`)[cite: 10].
   * Если зафиксирован резкий шум (`is_loud_spike=True`), но все люди в кадре стоят в абсолютно спокойной позе (`kinetic_energy < 0.12` и `NEUTRAL_POSTURE`), звуковой скор умножается на понижающий коэффициент `0.3` (защита от бытового шума в магазине).
4. **Вычисление объединённой эмоции и скора негатива (`negative_score` в `[0.0, 1.0]`):**
   * Для каждой модальности (`face`, `body`, `audio`) негативный скор равен её `confidence`, если распознанная эмоция входит в `{ANGRY, SAD, FEAR, DISGUST}`[cite: 4, 12], иначе `0.0`.
   * *Сцена 1 (`face_audio`):* `0.65 * face_score + 0.35 * audio_score` (при речи) или `face_score` (в тишине).
   * *Сцена 2 (`face_body_audio` — Quality-Gated Fusion)[cite: 10]:*
     * Если лицо видно и прошло `Quality Gate`: `0.50 * face_score + 0.25 * body_score + 0.25 * audio_score`.
     * Если человек отвернулся или лицо $< 64\text{ px}$: вес лица обнуляется, трек непрерывно ведётся по формуле `0.55 * body_score + 0.45 * audio_score` (используя эмоцию тела Уровня 3 `body_emotion` и голос)[cite: 10].
   * *Сцена 3 (`body_audio`):* `0.55 * body_score + 0.45 * audio_score` (итоговая эмоция трека выбирается по максимуму взвешенной уверенности между `body_emotion` Уровня 3 и `audio_emotion`).
5. **Временная агрегация (окно 1.5–3 секунды по `track_id`)[cite: 1, 4]:**
   * Хранить скользящее окно `deque(maxlen=temporal_window)` последних скоров и эмоций для каждого `track_id` (очищая ушедшие `stale_tid`[cite: 18]).
   * Выставлять `is_negative_alert = True`, только если сглаженный за окно скор $\ge \text{alert\_threshold}$ (`0.65`).

---

### Шаг 5. Классы пайплайнов и регистрация в `app/core/pipeline_registry.py`

1. Создать файл `mood_base/models/multimodal_scenes.py` с тремя классами-наследниками `BaseFaceEmotionModel`[cite: 18]:
   * `Scene1FaceAudioModel`
   * `Scene2FaceBodyAudioModel`
   * `Scene3BodyAudioModel`
2. В `Scene2FaceBodyAudioModel` (для тел без видимого лица) и в `Scene3BodyAudioModel` (для всех тел) добавлять в `FrameResult.detections` прокси-объект `FaceDetection(track_id=body.track_id, bbox=body.bbox, emotion=track_event.final_emotion, face_confidence=body.body_confidence, emotion_confidence=body.body_emotion_confidence)`, чтобы штатный UI-плеер и `scripts/try.py` отрисовывали рамки и эмоции всех людей[cite: 12, 19, 23].
3. Зарегистрировать 3 новых пайплайна в `app/core/pipeline_registry.py` после `"yolo_vit"` (оставив `"yolo_vit"` первым элементом списка!)[cite: 21].

---

## 3. Чек-лист обязательной верификации

После реализации прогнать все 4 скрипта и убедиться, что каждый возвращает `0`:
1. `python scripts/emotion_cache_check.py`[cite: 20]
2. `QT_QPA_PLATFORM=offscreen python scripts/presets_check.py`[cite: 21]
3. `QT_QPA_PLATFORM=offscreen python scripts/streams_ui_check.py`[cite: 22]
4. `python scripts/e2e_check.py`[cite: 19]