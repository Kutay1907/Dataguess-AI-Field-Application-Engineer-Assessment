export class MetricsPanel {
    constructor() {
        this.fpsEl = document.getElementById('metric-fps');
        this.latencyEl = document.getElementById('metric-latency');
        this.countEl = document.getElementById('metric-count');
        this.backendEl = document.getElementById('metric-backend');
    }

    update(data) {
        if (!data) return;

        this.fpsEl.textContent = data.fps.toFixed(1);
        this.latencyEl.textContent = data.latency_ms.toFixed(1) + ' ms';
        this.countEl.textContent = data.detection_count;
        this.backendEl.textContent = data.backend.toUpperCase();
    }
}
