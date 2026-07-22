const api = new ImageEnhancerAPI();

const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const abortBtn = document.getElementById('abortBtn');
const progressBarInner = document.getElementById('progressBarInner');
const statusText = document.getElementById('statusText');
const originalImg = document.getElementById('originalImg');
const resultImg = document.getElementById('resultImg');
const downloadLink = document.getElementById('downloadLink');
const logBox = document.getElementById('logBox');

let selectedFile = null;
let currentTaskId = null;

function log(msg) {
  const time = new Date().toLocaleTimeString();
  logBox.textContent += `[${time}] ${msg}\n`;
  logBox.scrollTop = logBox.scrollHeight;
}

fileInput.addEventListener('change', () => {
  selectedFile = fileInput.files[0] || null;
  if (selectedFile) {
    originalImg.src = URL.createObjectURL(selectedFile);
    resultImg.removeAttribute('src');
    downloadLink.style.display = 'none';
    processBtn.disabled = false;
    statusText.textContent = `Файл выбран: ${selectedFile.name}`;
    log(`Выбран файл: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} МБ)`);
  } else {
    processBtn.disabled = true;
  }
});

processBtn.addEventListener('click', () => runProcessing(selectedFile));

// ---------- Демо-режим: готовые примеры в один клик ----------
document.querySelectorAll('.demo-btn').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const src = btn.dataset.src;
    const label = btn.dataset.label;
    log(`Демо: загрузка примера "${label}" (${src})`);

    document.querySelectorAll('.demo-btn').forEach((b) => (b.disabled = true));
    statusText.textContent = `Загрузка примера «${label}»…`;

    try {
      const response = await fetch(src);
      const blob = await response.blob();
      const demoFile = new File([blob], src.split('/').pop(), { type: blob.type });

      selectedFile = demoFile;
      fileInput.value = ''; // сбрасываем выбор из <input>, чтобы не путать источники
      originalImg.src = URL.createObjectURL(demoFile);
      resultImg.removeAttribute('src');
      downloadLink.style.display = 'none';

      await runProcessing(demoFile);
    } finally {
      document.querySelectorAll('.demo-btn').forEach((b) => (b.disabled = false));
    }
  });
});

async function runProcessing(file) {
  if (!file) return;
  processBtn.disabled = true;
  abortBtn.disabled = false;
  progressBarInner.style.width = '0%';
  statusText.textContent = 'Постановка задачи...';
  log('Вызов submitTask()');

  currentTaskId = await api.submitTask(file);
  log(`Получен taskId: ${currentTaskId}`);
}

abortBtn.addEventListener('click', () => {
  if (!currentTaskId) return;
  const ok = api.abortTask(currentTaskId);
  log(ok ? 'Вызван abortTask() — отмена отправлена' : 'abortTask() — отменять нечего');
});

api.addEventListener('taskstatuschange', (e) => {
  const { taskId, status, progress } = e.detail;
  if (taskId !== currentTaskId) return;

  progressBarInner.style.width = progress + '%';

  const statusLabels = {
    queued: 'В очереди…',
    processing: `Обработка… ${progress}%`,
    done: 'Готово!',
    error: 'Ошибка обработки',
    aborted: 'Обработка прервана пользователем'
  };
  statusText.textContent = statusLabels[status] || status;
  log(`Событие taskstatuschange: status=${status}, progress=${progress}`);

  if (status === 'done') {
    const resultUrl = api.getResult(taskId);
    resultImg.src = resultUrl;
    downloadLink.href = resultUrl;
    downloadLink.style.display = 'inline-block';
    processBtn.disabled = false;
    abortBtn.disabled = true;
  } else if (status === 'error' || status === 'aborted') {
    processBtn.disabled = false;
    abortBtn.disabled = true;
  }
});
