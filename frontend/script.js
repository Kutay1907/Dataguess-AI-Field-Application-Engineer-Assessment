import { Toolbar } from './components/toolbar.js';
import { VideoPanel } from './components/video_panel.js';
import { MetricsPanel } from './components/metrics.js';
import { LiveCounts } from './components/live_counts.js';
import { FPSGraph } from './components/fps_graph.js';

class App {
    constructor() {
        this.API_URL = 'http://127.0.0.1:8000';
        this.isRunning = false;
        this.pollInterval = null;

        // Initialize Components
        this.toolbar = new Toolbar({
            onStart: () => this.startVideo(),
            onStop: () => this.stopVideo(),
            onUpload: (e) => this.uploadVideo(e)
        });

        this.videoPanel = new VideoPanel();
        this.metrics = new MetricsPanel();
        this.counts = new LiveCounts();
        this.graph = new FPSGraph('fpsChart');

        // Initial State
        this.fetchMetrics(); // Fetch once to see if already running
    }

    async startVideo() {
        try {
            this.toolbar.setUploading(true); // temporary state
            await fetch(`${this.API_URL}/start`, { method: 'POST' });
            this.setRunningState(true);
        } catch (e) {
            console.error("Failed to start", e);
            alert("Failed to start video engine.");
        } finally {
            this.toolbar.setUploading(false);
        }
    }

    async stopVideo() {
        try {
            await fetch(`${this.API_URL}/stop`, { method: 'POST' });
            this.setRunningState(false);
        } catch (e) {
            console.error(e);
        }
    }

    async uploadVideo(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.toolbar.setUploading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${this.API_URL}/upload_video`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.status === 'success') {
                this.videoPanel.setSource(data.filename);
                this.setRunningState(true);
            } else {
                alert("Upload failed: " + data.detail);
            }
        } catch (e) {
            console.error(e);
            alert("Upload error");
        } finally {
            this.toolbar.setUploading(false);
            event.target.value = ''; // Reset
        }
    }

    setRunningState(running) {
        this.isRunning = running;
        this.toolbar.setRunning(running);

        if (running) {
            this.videoPanel.startStream();
            this.startPolling();
        } else {
            this.videoPanel.stopStream();
            this.stopPolling();
        }
    }

    startPolling() {
        if (this.pollInterval) clearInterval(this.pollInterval);
        this.pollInterval = setInterval(() => this.fetchMetrics(), 1000);
    }

    stopPolling() {
        if (this.pollInterval) clearInterval(this.pollInterval);
        this.pollInterval = null;
    }

    async fetchMetrics() {
        try {
            const res = await fetch(`${this.API_URL}/metrics`);
            const data = await res.json();

            if (data.video_engine) {
                const eng = data.video_engine;

                // Update UI Components
                this.metrics.update(eng);
                this.counts.update(eng.class_counts);
                this.graph.update(eng.fps);

                // Sync state if external change happened
                if (eng.active !== this.isRunning) {
                    this.setRunningState(eng.active);
                }
            }
        } catch (e) {
            // console.warn("Metrics fetch failed", e);
        }
    }
}

// Start App
window.addEventListener('DOMContentLoaded', () => {
    new App();
});
