/**
 * Draws the image and bounding boxes on the canvas.
 * @param {HTMLCanvasElement} canvas 
 * @param {HTMLImageElement} image 
 * @param {Array} detections 
 */
function drawResults(canvas, image, detections) {
    const ctx = canvas.getContext('2d');

    // Set canvas dimensions to match image
    canvas.width = image.width;
    canvas.height = image.height;

    // Draw original image
    ctx.drawImage(image, 0, 0);

    // Style settings for bounding boxes
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#22c55e'; // Green
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textBaseline = 'top';

    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        const label = `${det.class_name} ${Math.round(det.score * 100)}%`;

        // Draw box
        ctx.beginPath();
        ctx.rect(x1, y1, width, height);
        ctx.stroke();

        // Draw label background
        const textWidth = ctx.measureText(label).width;
        const padding = 4;

        ctx.fillStyle = '#22c55e';
        ctx.fillRect(x1, y1 - 24, textWidth + padding * 2, 24);

        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + padding, y1 - 20);
    });
}

/**
 * Clears the canvas
 * @param {HTMLCanvasElement} canvas 
 */
function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
