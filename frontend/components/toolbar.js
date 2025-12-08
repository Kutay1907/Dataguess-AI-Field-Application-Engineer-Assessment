export class Toolbar {
    constructor(callbacks) {
        this.callbacks = callbacks;
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.videoUpload = document.getElementById('videoUpload');
        this.engineStatus = document.getElementById('engineStatus');

        this.bindEvents();
    }

    bindEvents() {
        this.startBtn.addEventListener('click', () => this.callbacks.onStart());
        this.stopBtn.addEventListener('click', () => this.callbacks.onStop());
        this.videoUpload.addEventListener('change', (e) => this.callbacks.onUpload(e));
    }

    setRunning(running) {
        this.startBtn.disabled = running;
        this.stopBtn.disabled = !running;
        this.startBtn.classList.toggle('active', running);

        if (running) {
            this.engineStatus.innerHTML = '<span class="status-dot live"></span> Live';
            this.engineStatus.className = 'status-badge status-live';
        } else {
            this.engineStatus.innerHTML = '<span class="status-dot stopped"></span> Stopped';
            this.engineStatus.className = 'status-badge status-stopped';
        }
    }

    setUploading(uploading) {
        this.startBtn.disabled = uploading;
        if (uploading) {
            this.engineStatus.innerHTML = '<span class="status-dot processing"></span> Processing...';
            this.engineStatus.className = 'status-badge status-processing';
        }
    }
}
