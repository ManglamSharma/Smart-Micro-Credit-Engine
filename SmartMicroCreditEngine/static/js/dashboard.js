/* ============================================================
   Dashboard charts - Chart.js
   Expects globals: CHART_DATA, FINAL_SCORE, SCORE_MIN, SCORE_MAX
   ============================================================ */
(function () {
    "use strict";

    const palette = {
        primary: "#4f7cff",
        accent: "#00d4a0",
        info: "#38bdf8",
        warning: "#ffb020",
        danger: "#ff5c7c",
        success: "#22c98e",
        grid: "rgba(122, 137, 184, 0.15)",
        text: "#9aa6cf",
    };

    Chart.defaults.color = palette.text;
    Chart.defaults.font.family = "'Inter', sans-serif";

    function scoreColor(score) {
        if (score >= 650) return palette.success;
        if (score >= 550) return palette.warning;
        return palette.danger;
    }

    // ---- 1. Gauge (doughnut) for the final score ----
    const gaugeEl = document.getElementById("gaugeChart");
    if (gaugeEl) {
        const pct = (FINAL_SCORE - SCORE_MIN) / (SCORE_MAX - SCORE_MIN);
        const filled = Math.max(0, Math.min(1, pct)) * 100;
        new Chart(gaugeEl, {
            type: "doughnut",
            data: {
                datasets: [{
                    data: [filled, 100 - filled],
                    backgroundColor: [scoreColor(FINAL_SCORE), "rgba(122,137,184,0.12)"],
                    borderWidth: 0,
                    circumference: 270,
                    rotation: 225,
                }],
            },
            options: {
                cutout: "78%",
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
            },
        });
    }

    // ---- 2. Credit score trend (line, multi-series) ----
    const trendEl = document.getElementById("scoreTrendChart");
    if (trendEl) {
        new Chart(trendEl, {
            type: "line",
            data: {
                labels: CHART_DATA.labels,
                datasets: [
                    {
                        label: "Final (Hybrid)",
                        data: CHART_DATA.final_scores,
                        borderColor: palette.success,
                        backgroundColor: "rgba(34,201,142,0.12)",
                        borderWidth: 3,
                        fill: true,
                        tension: 0.35,
                        pointRadius: 4,
                        pointBackgroundColor: palette.success,
                    },
                    {
                        label: "ML Score",
                        data: CHART_DATA.ml_scores,
                        borderColor: palette.primary,
                        borderWidth: 2,
                        tension: 0.35,
                        pointRadius: 3,
                        borderDash: [5, 4],
                    },
                    {
                        label: "Formula Score",
                        data: CHART_DATA.formula_scores,
                        borderColor: palette.accent,
                        borderWidth: 2,
                        tension: 0.35,
                        pointRadius: 3,
                        borderDash: [5, 4],
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                scales: {
                    y: { min: SCORE_MIN, max: SCORE_MAX, grid: { color: palette.grid } },
                    x: { grid: { color: palette.grid } },
                },
                plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 16 } } },
            },
        });
    }

    // ---- 3. Growth trend (bar) ----
    const growthEl = document.getElementById("growthChart");
    if (growthEl) {
        const growthVals = CHART_DATA.growth;
        new Chart(growthEl, {
            type: "bar",
            data: {
                labels: CHART_DATA.labels,
                datasets: [{
                    label: "Growth %",
                    data: growthVals,
                    backgroundColor: growthVals.map(v =>
                        v > 0 ? "rgba(34,201,142,0.75)" : (v < 0 ? "rgba(255,92,124,0.75)" : "rgba(122,137,184,0.5)")
                    ),
                    borderRadius: 6,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: palette.grid }, ticks: { callback: v => v + "%" } },
                    x: { grid: { display: false } },
                },
                plugins: { legend: { display: false } },
            },
        });
    }

    // ---- 4. Decision history (doughnut breakdown) ----
    const decEl = document.getElementById("decisionChart");
    if (decEl) {
        const counts = {};
        CHART_DATA.decisions.forEach(d => { counts[d] = (counts[d] || 0) + 1; });
        const decColors = {
            "APPROVED": palette.success,
            "REJECTED": palette.danger,
            "CONDITIONAL APPROVAL": palette.warning,
            "MANUAL REVIEW": palette.info,
        };
        const labels = Object.keys(counts);
        new Chart(decEl, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [{
                    data: labels.map(l => counts[l]),
                    backgroundColor: labels.map(l => decColors[l] || palette.primary),
                    borderColor: "#0b1020",
                    borderWidth: 3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "62%",
                plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 14, font: { size: 11 } } } },
            },
        });
    }
})();
