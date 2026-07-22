/**
 * ImageEnhancerAPI — модуль, реализующий API из технического задания:
 *  - submitTask(file)   -> возвращает taskId (Promise<string>)
 *  - getStatus(taskId)  -> {status, progress}
 *  - abortTask(taskId)  -> boolean
 *  - getResult(taskId)  -> url готового изображения или null
 *  - событие 'taskstatuschange' -> {taskId, status, progress}
 *
 * Вся тяжёлая обработка выполняется в отдельном потоке (Web Worker),
 * поэтому главный поток браузера не блокируется.
 */
class ImageEnhancerAPI extends EventTarget {

  constructor() {
    super();
    this.worker = new Worker('worker.js');
    this.worker.onmessage = (e) => this._handleWorkerMessage(e.data);
    this.tasks = new Map(); // taskId -> {status, progress, width, height, resultUrl, error}
  }

  /**
   * Метод постановки задачи.
   * Принимает File, возвращает идентификатор задачи.
   */
  async submitTask(file) {
    const taskId = crypto.randomUUID();
    this.tasks.set(taskId, { status: 'queued', progress: 0, width: 0, height: 0, resultUrl: null, error: null });
    this._emit(taskId);

    try {
      const { imageData, width, height } = await this._decodeFile(file);
      const task = this.tasks.get(taskId);
      task.width = width;
      task.height = height;
      task.status = 'processing';
      this._emit(taskId);

      // передаём буфер пикселей в worker как transferable (без копирования)
      this.worker.postMessage(
        { type: 'start', taskId, buffer: imageData.data.buffer, width, height },
        [imageData.data.buffer]
      );
    } catch (err) {
      const task = this.tasks.get(taskId);
      task.status = 'error';
      task.error = err.message || String(err);
      this._emit(taskId);
    }

    return taskId;
  }

  /**
   * Метод получения статуса задачи.
   */
  getStatus(taskId) {
    const t = this.tasks.get(taskId);
    if (!t) return null;
    return { status: t.status, progress: t.progress, error: t.error };
  }

  /**
   * Метод прерывания задачи.
   */
  abortTask(taskId) {
    const t = this.tasks.get(taskId);
    if (!t || t.status === 'done' || t.status === 'error' || t.status === 'aborted') {
      return false;
    }
    this.worker.postMessage({ type: 'abort', taskId });
    return true;
  }

  /**
   * Метод получения готового изображения.
   * Возвращает object URL, пригодный для <img src> или скачивания.
   */
  getResult(taskId) {
    const t = this.tasks.get(taskId);
    if (!t || t.status !== 'done') return null;
    return t.resultUrl;
  }

  // ------------------ внутренние методы ------------------

  async _decodeFile(file) {
    let blob = file;

    // HEIC/HEIF браузеры не декодируют нативно — конвертируем в JPEG через heic2any
    const isHeic = /heic|heif/i.test(file.type) || /\.(heic|heif)$/i.test(file.name);
    if (isHeic) {
      blob = await heic2any({ blob: file, toType: 'image/jpeg', quality: 0.92 });
    }

    const bitmap = await createImageBitmap(blob);

    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    return { imageData, width: canvas.width, height: canvas.height };
  }

  _handleWorkerMessage(msg) {
    const task = this.tasks.get(msg.taskId);
    if (!task) return;

    if (msg.type === 'progress') {
      task.status = msg.status;
      task.progress = msg.progress;
      this._emit(msg.taskId);

    } else if (msg.type === 'done') {
      const imageData = new ImageData(new Uint8ClampedArray(msg.buffer), task.width, task.height);
      const canvas = document.createElement('canvas');
      canvas.width = task.width;
      canvas.height = task.height;
      canvas.getContext('2d').putImageData(imageData, 0, 0);
      canvas.toBlob((resultBlob) => {
        if (task.resultUrl) URL.revokeObjectURL(task.resultUrl);
        task.resultUrl = URL.createObjectURL(resultBlob);
        task.status = 'done';
        task.progress = 100;
        this._emit(msg.taskId);
      }, 'image/png');

    } else if (msg.type === 'aborted') {
      task.status = 'aborted';
      this._emit(msg.taskId);

    } else if (msg.type === 'error') {
      task.status = 'error';
      task.error = msg.message;
      this._emit(msg.taskId);
    }
  }

  _emit(taskId) {
    const t = this.tasks.get(taskId);
    this.dispatchEvent(new CustomEvent('taskstatuschange', {
      detail: { taskId, status: t.status, progress: t.progress }
    }));
  }
}
