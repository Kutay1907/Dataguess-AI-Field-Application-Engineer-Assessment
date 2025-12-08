export class VideoPanel {
    constructor() {
        this.videoStream = document.getElementById('videoStream');
        this.placeholder = document.getElementById('videoPlaceholder');
        this.sourceLabel = document.getElementById('currentSource');
        this.API_URL = 'http://127.0.0.1:8000';
        this.frameInterval = null;
    }

    startStream() {
        this.videoStream.style.display = 'block';
        this.placeholder.style.display = 'none';

        if (this.frameInterval) clearInterval(this.frameInterval);
        this.frameInterval = setInterval(() => {
            this.videoStream.src = `${this.API_URL}/frame?t=${new Date().getTime()}`;
        }, 100); // 10 FPS poll for UI smoothness (backend is faster)
    }

    stopStream() {
        if (this.frameInterval) clearInterval(this.frameInterval);
        this.frameInterval = null;
        // Optionally keep last frame or show placeholder
        // this.videoStream.style.display = 'none';
        // this.placeholder.style.display = 'flex';
    }

    setSource(name) {
        this.sourceLabel.textContent = name;
    }
}
