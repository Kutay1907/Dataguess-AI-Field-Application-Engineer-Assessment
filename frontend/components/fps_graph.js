export class FPSGraph {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.data = new Array(60).fill(0); // 60 frames history
        this.maxFPS = 60;

        // Resize observer for responsive canvas
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        this.canvas.width = this.canvas.parentElement.clientWidth;
        this.canvas.height = this.canvas.parentElement.clientHeight;
    }

    update(fps) {
        this.data.push(fps);
        this.data.shift();
        this.draw();
    }

    draw() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const ctx = this.ctx;

        ctx.clearRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = '#334155'; // Slate-700
        ctx.lineWidth = 1;
        ctx.beginPath();
        // Horizontal lines
        for (let i = 0; i <= 4; i++) {
            const y = (h / 4) * i;
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
        }
        ctx.stroke();

        // Graph Line
        ctx.strokeStyle = '#06b6d4'; // Cyan-500
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.beginPath();

        const step = w / (this.data.length - 1);

        this.data.forEach((val, i) => {
            const y = h - ((Math.min(val, this.maxFPS) / this.maxFPS) * h);
            const x = i * step;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Fill area
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.1)'; // Cyan transparent
        ctx.fill();

        // Max Label
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px Inter';
        ctx.fillText('60 FPS', 5, 12);
    }
}
