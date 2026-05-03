// ===== DOM References =====
const fileInput      = document.getElementById("fileInput");
const dropZone       = document.getElementById("dropZone");
const fileInfo       = document.getElementById("fileInfo");
const resultSection  = document.getElementById("resultSection");
const previewImage   = document.getElementById("previewImage");
const spinnerWrap    = document.getElementById("spinnerWrap");
const loadingMsg     = document.getElementById("loadingMsg");
const captionBox     = document.getElementById("captionBox");
const captionText    = document.getElementById("captionText");
const errorBox       = document.getElementById("errorBox");
const errorText      = document.getElementById("errorText");
const copyBtn        = document.getElementById("copyBtn");
const speakBtn       = document.getElementById("speakBtn");
const resetBtn       = document.getElementById("resetBtn");
const retryBtn       = document.getElementById("retryBtn");
const altList        = document.getElementById("altList");
const altSection     = document.getElementById("alternativesSection");
const webcamBtn      = document.getElementById("webcamBtn");
const webcamModal    = document.getElementById("webcamModal");
const webcamVideo    = document.getElementById("webcamVideo");
const webcamCanvas   = document.getElementById("webcamCanvas");
const captureBtn     = document.getElementById("captureBtn");
const closeWebcamBtn = document.getElementById("closeWebcamBtn");

// Witty loading messages
const LOADING_MSGS = [
  "Asking the AI nicely...",
  "Reading pixels...",
  "Crafting your caption...",
  "Processing context...",
  "Almost there...",
];

let lastFile = null;
let webcamStream = null;
let loadingInterval = null;

// ===== File Input & Drag-and-Drop =====
// FIX: Do NOT add a click listener on the entire dropZone.
// The <label for="fileInput"> already opens the file dialog.
// Adding dropZone.addEventListener("click", ...) causes the dialog to open twice.

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

// ===== File Validation & Preview =====

function handleFile(file) {
  const allowed = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Invalid file type. Please upload a PNG, JPG, GIF, or WEBP image.");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError("File too large. Maximum size is 20MB.");
    return;
  }

  fileInfo.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
  lastFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    showResultSection();
    uploadAndCaption(file);
  };
  reader.readAsDataURL(file);
}

// ===== UI State =====

function showResultSection() {
  resultSection.hidden = false;
  spinnerWrap.hidden = false;
  captionBox.hidden = true;
  errorBox.hidden = true;
  altSection.hidden = true;
  altList.innerHTML = "";
  startLoadingMessages();
}

function showCaptionUI(caption, alternatives) {
  stopLoadingMessages();
  spinnerWrap.hidden = true;
  captionBox.hidden = false;
  errorBox.hidden = true;
  captionText.textContent = caption;

  if (alternatives && alternatives.length > 0) {
    altList.innerHTML = "";
    alternatives.forEach((alt, i) => {
      const li = document.createElement("li");
      li.textContent = alt;
      li.setAttribute("data-num", i + 1);
      li.title = "Click to use this caption";
      li.addEventListener("click", () => {
        captionText.textContent = alt;
        li.style.borderColor = "var(--accent)";
        setTimeout(() => (li.style.borderColor = ""), 1000);
      });
      altList.appendChild(li);
    });
    altSection.hidden = false;
  }
}

function showError(message) {
  stopLoadingMessages();
  resultSection.hidden = false;
  spinnerWrap.hidden = true;
  captionBox.hidden = true;
  errorBox.hidden = false;
  errorText.textContent = message;
}

function startLoadingMessages() {
  let idx = 0;
  loadingMsg.textContent = LOADING_MSGS[0];
  loadingInterval = setInterval(() => {
    idx = (idx + 1) % LOADING_MSGS.length;
    loadingMsg.textContent = LOADING_MSGS[idx];
  }, 2000);
}

function stopLoadingMessages() {
  clearInterval(loadingInterval);
}

// ===== API Call to Flask /predict =====

async function uploadAndCaption(file) {
  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok || data.error) {
      showError(data.error || "An unexpected error occurred.");
      return;
    }
    showCaptionUI(data.caption, data.alternatives);
  } catch {
    showError("Network error. Please check your connection and try again.");
  }
}

// ===== Copy Caption =====

copyBtn.addEventListener("click", async () => {
  const text = captionText.textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }
  copyBtn.textContent = "Copied!";
  copyBtn.classList.add("copied");
  setTimeout(() => {
    copyBtn.textContent = "Copy";
    copyBtn.classList.remove("copied");
  }, 2000);
});

// ===== Voice Output =====

speakBtn.addEventListener("click", () => {
  const text = captionText.textContent;
  if (!text || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  speakBtn.textContent = "Speaking...";
  speakBtn.disabled = true;
  utterance.onend = utterance.onerror = () => {
    speakBtn.textContent = "Read Aloud";
    speakBtn.disabled = false;
  };
  speechSynthesis.speak(utterance);
});

// ===== Reset =====

resetBtn.addEventListener("click", resetUI);
retryBtn.addEventListener("click", () => {
  if (lastFile) {
    showResultSection();
    uploadAndCaption(lastFile);
  } else {
    resetUI();
  }
});

function resetUI() {
  resultSection.hidden = true;
  fileInfo.textContent = "";
  fileInput.value = "";
  lastFile = null;
  captionText.textContent = "";
  previewImage.src = "";
  altList.innerHTML = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ===== Webcam =====

webcamBtn.addEventListener("click", async () => {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamVideo.srcObject = webcamStream;
    webcamModal.hidden = false;
  } catch {
    showError("Could not access webcam. Please allow camera permission in your browser.");
  }
});

captureBtn.addEventListener("click", () => {
  webcamCanvas.width = webcamVideo.videoWidth;
  webcamCanvas.height = webcamVideo.videoHeight;
  webcamCanvas.getContext("2d").drawImage(webcamVideo, 0, 0);
  webcamCanvas.toBlob(
    (blob) => {
      const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
      stopWebcam();
      handleFile(file);
    },
    "image/jpeg",
    0.92
  );
});

closeWebcamBtn.addEventListener("click", stopWebcam);
webcamModal.addEventListener("click", (e) => {
  if (e.target === webcamModal) stopWebcam();
});

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  webcamModal.hidden = true;
}
