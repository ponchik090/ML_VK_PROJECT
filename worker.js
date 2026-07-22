/**
 * worker.js
 *
 * Здесь происходит вся тяжёлая работа, чтобы не блокировать интерфейс:
 *  1) ML-модель (TensorFlow.js) анализирует изображение и подбирает силу
 *     коррекции: насколько растягивать контраст (auto-levels) и
 *     насколько усилить насыщенность цвета.
 *  2) Простой алгоритм на JS применяет эти параметры к каждому пикселю
 *     (включая лёгкую коррекцию баланса белого), с отчётом о прогрессе
 *     и возможностью прерывания.
 */

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js');

let currentTask = null; // { taskId, aborted }

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'start') {
    currentTask = { taskId: msg.taskId, aborted: false };
    try {
      await processImage(msg);
    } catch (err) {
      postMessage({ type: 'error', taskId: msg.taskId, message: err.message });
    }

  } else if (msg.type === 'abort') {
    if (currentTask && currentTask.taskId === msg.taskId) {
      currentTask.aborted = true;
    }
  }
};

async function processImage({ taskId, buffer, width, height }) {
  const pixels = new Uint8ClampedArray(buffer);
  console.time(`⏱ total ${taskId}`);

  postProgress(taskId, 0, 'processing');

  // ---------- Шаг 1: ML-модель подбирает параметры коррекции ----------
  console.time(`⏱ ML-анализ ${taskId}`);
  const params = analyzeAndPredict(pixels, width, height);
  console.timeEnd(`⏱ ML-анализ ${taskId}`);
  postProgress(taskId, 15, 'processing');

  // ---------- Шаг 2: применение параметров к изображению ----------
  const total = width * height;
  const progressStep = Math.max(1, Math.floor(total / 40)); // ~40 обновлений прогресса
  console.time(`⏱ применение коррекции ${taskId}`);

  for (let i = 0; i < total; i++) {
    const idx = i * 4;

    // 2.1 Растяжение контраста (auto-levels), ПОДМЕШАННОЕ по силе
    // (scale/offset уже "ослаблены" пропорционально params.stretchStrength
    // при расчёте в analyzeAndPredict — см. комментарий там).
    // Один и тот же scale/offset для всех каналов — не сдвигает цвет.
    let r = pixels[idx] * params.scale + params.offset;
    let g = pixels[idx + 1] * params.scale + params.offset;
    let b = pixels[idx + 2] * params.scale + params.offset;

    // 2.2 Лёгкая коррекция баланса белого (gray-world, частичная сила).
    r *= params.wbFactorR;
    g *= params.wbFactorG;
    b *= params.wbFactorB;

    // 2.3 Усиление насыщенности относительно яркости пикселя.
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    r = lum + (r - lum) * params.saturationBoost;
    g = lum + (g - lum) * params.saturationBoost;
    b = lum + (b - lum) * params.saturationBoost;

    pixels[idx] = clamp(r, 0, 255);
    pixels[idx + 1] = clamp(g, 0, 255);
    pixels[idx + 2] = clamp(b, 0, 255);
    // альфа-канал (idx+3) не трогаем

    if (i % progressStep === 0) {
      if (currentTask.aborted) {
        console.timeEnd(`⏱ применение коррекции ${taskId}`);
        console.timeEnd(`⏱ total ${taskId}`);
        postMessage({ type: 'aborted', taskId });
        return;
      }
      const progress = 15 + Math.floor((i / total) * 80);
      postProgress(taskId, progress, 'processing');
      await new Promise((res) => setTimeout(res, 0));
    }
  }

  console.timeEnd(`⏱ применение коррекции ${taskId}`);
  postProgress(taskId, 100, 'done');
  postMessage({ type: 'done', taskId, buffer: pixels.buffer }, [pixels.buffer]);
  console.timeEnd(`⏱ total ${taskId}`);
}

/**
 * Анализ изображения и предсказание параметров коррекции.
 * Возвращает {scale, offset, wbFactorR, wbFactorG, wbFactorB, saturationBoost}.
 */
function analyzeAndPredict(pixels, width, height) {
  let sortedLuminance;

  const stats = tf.tidy(() => {
    const rgba = tf.tensor3d(pixels, [height, width, 4], 'float32');
    const rgb = rgba.slice([0, 0, 0], [height, width, 3]);
    const small = tf.image.resizeBilinear(rgb, [128, 128]);

    const lum = small.mul(tf.tensor1d([0.299, 0.587, 0.114])).sum(-1);
    sortedLuminance = Float32Array.from(lum.dataSync()).sort();

    const { mean: meanLum } = tf.moments(lum);
    const { variance } = tf.moments(lum);
    const std = variance.sqrt();

    const maxC = small.max(-1);
    const minC = small.min(-1);
    const lightness = maxC.add(minC).div(2);
    const denom = tf.scalar(1).sub(lightness.div(127.5).sub(1).abs()).mul(255).add(1e-3);
    const satMap = maxC.sub(minC).div(denom);

    const meanRGB = small.mean([0, 1]).dataSync();

    return {
      brightness: meanLum.dataSync()[0] / 255,
      contrast: clamp(std.dataSync()[0] / 128, 0, 1),
      saturation: clamp(satMap.mean().dataSync()[0], 0, 1),
      meanR: meanRGB[0],
      meanG: meanRGB[1],
      meanB: meanRGB[2]
    };
  });

  // Модель решает: (1) насколько СИЛЬНО применять растяжение контраста
  // (0..1, не сам scale!) и (2) насколько усилить насыщенность.
  let { stretchStrength, saturationBoost } = predictParams(stats);

  // Для очень тёмных кадров (почти чёрное небо и т.п.) основной "контраст"
  // в тенях — это чаще шум матрицы, а не полезная деталь. Растягивая его
  // на полную силу, мы усиливаем именно шум. Поэтому при низкой средней
  // яркости плавно ограничиваем силу коррекции (не отключаем полностью —
  // немного улучшить такие фото всё равно можно).
  const DARK_THRESHOLD = 0.2; // ниже этой средней яркости (0..1) начинаем "придерживать" эффект
  const darkSafety = clamp(stats.brightness / DARK_THRESHOLD, 0, 1); // 0 = очень тёмное, 1 = не тёмное
  const dampFactor = 0.5 + 0.5 * darkSafety; // не менее 50% силы даже в самом тёмном случае
  stretchStrength *= dampFactor;
  saturationBoost = 1 + (saturationBoost - 1) * dampFactor;

  // Перцентиль для отсечения выбросов (шум/битые пиксели) — фиксированный,
  // небольшой, не связан с "силой" коррекции.
  const PERCENTILE_CLIP = 0.01; // 1%
  const n = sortedLuminance.length;
  const loIdx = clamp(Math.floor(n * PERCENTILE_CLIP), 0, n - 1);
  const hiIdx = clamp(Math.floor(n * (1 - PERCENTILE_CLIP)), 0, n - 1);
  const lo = sortedLuminance[loIdx];
  const hi = Math.max(sortedLuminance[hiIdx], lo + 1);

  // "Полное" auto-levels растяжение (как если бы применили на 100%)
  const rawScale = clamp(255 / (hi - lo), 1, 3); // не растягиваем сильнее чем в 3 раза
  const rawOffset = -lo * rawScale;

  // ГЛАВНОЕ ИСПРАВЛЕНИЕ: не применяем растяжение "жёстко" (иначе на уже
  // нормально освещённых фото света выбиваются в чистый белый и
  // теряются детали). Вместо этого ПЛАВНО подмешиваем эффект между
  // "как было" (scale=1, offset=0) и "полное растяжение" — сила
  // подмеса задаётся моделью (stretchStrength: чем ниже исходный
  // контраст, тем сильнее подмес, но не более ~80%).
  let scale = 1 + (rawScale - 1) * stretchStrength;
  let offset = rawOffset * stretchStrength;

  // ПРЕДОХРАНИТЕЛЬ: даже с подмесом выше, для изображений с ОЧЕНЬ узким
  // диапазоном яркости (например, почти однотонное светлое небо) rawScale
  // может быть большим (до 3х), и даже частичного подмеса достаточно,
  // чтобы утащить среднюю яркость сильно вниз — визуально светлое фото
  // становится тёмным. Поэтому явно ограничиваем, насколько вообще может
  // измениться СРЕДНЯЯ яркость кадра после коррекции — не больше чем на
  // MAX_BRIGHTNESS_SHIFT (в шкале 0..255) в любую сторону.
  const MAX_BRIGHTNESS_SHIFT = 40;
  const oldMean = stats.brightness * 255;
  const predictedNewMean = oldMean * scale + offset;
  const brightnessShift = predictedNewMean - oldMean;

  if (Math.abs(brightnessShift) > MAX_BRIGHTNESS_SHIFT) {
    const limitFactor = MAX_BRIGHTNESS_SHIFT / Math.abs(brightnessShift);
    scale = 1 + (scale - 1) * limitFactor;
    offset = offset * limitFactor;
  }

  // Баланс белого по методу "серого мира", тоже частичная сила.
  const WB_STRENGTH = 0.4;
  const MAX_WB_SHIFT = 0.18;
  const grayTarget = (stats.meanR + stats.meanG + stats.meanB) / 3;

  const wbFactorR = clamp(1 + (grayTarget / Math.max(stats.meanR, 1) - 1) * WB_STRENGTH, 1 - MAX_WB_SHIFT, 1 + MAX_WB_SHIFT);
  const wbFactorG = clamp(1 + (grayTarget / Math.max(stats.meanG, 1) - 1) * WB_STRENGTH, 1 - MAX_WB_SHIFT, 1 + MAX_WB_SHIFT);
  const wbFactorB = clamp(1 + (grayTarget / Math.max(stats.meanB, 1) - 1) * WB_STRENGTH, 1 - MAX_WB_SHIFT, 1 + MAX_WB_SHIFT);

  return { scale, offset, wbFactorR, wbFactorG, wbFactorB, saturationBoost };
}

/**
 * Небольшая полносвязная сеть (tf.layers.dense + sigmoid): по признакам
 * изображения [яркость, контраст, насыщенность] предсказывает два числа
 * 0..1 — "насколько сильно" применять растяжение контраста и усиление
 * насыщенности (не абсолютные величины, а именно СИЛУ/долю подмеса).
 *
 * Веса заданы аналитически (не через обучение на датасете) — сеть
 * реализует правило "чем слабее исходный параметр, тем сильнее
 * коррекция". Реальный объект tf.js делает predict(), но веса не
 * подбирались градиентным спуском — см. README про дообучение.
 */
function buildCorrectionModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 2, inputShape: [3], activation: 'sigmoid', useBias: true }));

  const K = 6;
  const B = 3;

  const kernel = tf.tensor2d([
    [0, 0],     // brightness почти не влияет
    [-K, 0],    // выше исходный контраст -> меньше сила растяжения
    [0, -K]     // выше исходная насыщенность -> меньше сила усиления
  ]);
  const bias = tf.tensor1d([B, B]);

  model.layers[0].setWeights([kernel, bias]);
  return model;
}

function predictParams({ brightness, contrast, saturation }) {
  return tf.tidy(() => {
    const input = tf.tensor2d([[brightness, contrast, saturation]]);
    const model = buildCorrectionModel();
    const [s0, s1] = model.predict(input).dataSync();

    return {
      // 0.15 (почти не трогаем) .. 0.80 (заметно растягиваем) —
      // никогда не 100%, чтобы не выбивать света в чистый белый
      stretchStrength: 0.15 + s0 * 0.65,
      // 1.0 .. 1.3 — только усиление насыщенности, умеренно
      saturationBoost: 1.0 + s1 * 0.3
    };
  });
}

function postProgress(taskId, progress, status) {
  postMessage({ type: 'progress', taskId, progress, status });
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}
