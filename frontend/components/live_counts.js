export class LiveCounts {
    constructor() {
        this.listEl = document.getElementById('objectList');
        // Define expected classes for the "blue pill" view
        // Ideally this comes from config, but for FAE demo we hardcode common ones
        this.expectedClasses = ['person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle', 'dog'];
    }

    update(counts) {
        counts = counts || {};
        let html = '';

        // Merge detected counts with expected zero-counts
        const allKeys = new Set([...this.expectedClasses, ...Object.keys(counts)]);

        // Sort for stability
        const sortedKeys = Array.from(allKeys).sort();

        sortedKeys.forEach(key => {
            const count = counts[key] || 0;
            const activeClass = count > 0 ? 'active-count' : 'zero-count';

            html += `
                <li class="count-item ${activeClass}">
                    <span class="count-label">${this.capitalize(key)}</span>
                    <span class="count-pill">${count}</span>
                </li>
            `;
        });

        this.listEl.innerHTML = html;
    }

    capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}
