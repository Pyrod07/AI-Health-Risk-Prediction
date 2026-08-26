// For local development.
// For deployment, change this to your deployed FastAPI URL.
const API_URL = "http://127.0.0.1:8000";

const form = document.getElementById("predictionForm");
const predictBtn = document.getElementById("predictBtn");
const resultCard = document.getElementById("resultCard");
const emptyResult = document.getElementById("emptyResult");
const resultContent = document.getElementById("resultContent");
const historyContainer = document.getElementById("historyContainer");
const refreshBtn = document.getElementById("refreshBtn");

function numberValue(id) {
    return Number(document.getElementById(id).value);
}

function buildPayload() {
    return {
        age: numberValue("age"),
        sex: numberValue("sex"),
        cp: numberValue("cp"),
        trestbps: numberValue("trestbps"),
        chol: numberValue("chol"),
        fbs: numberValue("fbs"),
        restecg: numberValue("restecg"),
        thalach: numberValue("thalach"),
        exang: numberValue("exang"),
        oldpeak: numberValue("oldpeak"),
        slope: numberValue("slope"),
        ca: numberValue("ca"),
        thal: numberValue("thal")
    };
}

function setRiskVisual(percentage, level) {
    const gauge = document.getElementById("gauge");
    const chip = document.getElementById("riskChip");

    const safePercentage = Math.max(0, Math.min(100, Number(percentage) || 0));
    gauge.style.setProperty("--progress", `${safePercentage * 3.6}deg`);

    chip.textContent = level || "Risk";

    const normalized = String(level || "").toLowerCase();

    if (normalized.includes("high")) {
        chip.style.color = "#fb7185";
        chip.style.background = "rgba(251, 113, 133, .09)";
        gauge.style.background =
            `conic-gradient(#fb7185 ${safePercentage * 3.6}deg, rgba(148,163,184,.10) 0)`;
    } else if (normalized.includes("medium") || normalized.includes("moderate")) {
        chip.style.color = "#fbbf24";
        chip.style.background = "rgba(251, 191, 36, .09)";
        gauge.style.background =
            `conic-gradient(#fbbf24 ${safePercentage * 3.6}deg, rgba(148,163,184,.10) 0)`;
    } else {
        chip.style.color = "#34d399";
        chip.style.background = "rgba(52, 211, 153, .09)";
        gauge.style.background =
            `conic-gradient(#34d399 ${safePercentage * 3.6}deg, rgba(148,163,184,.10) 0)`;
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    predictBtn.disabled = true;
    predictBtn.querySelector(".btn-label").textContent = "Analyzing...";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(buildPayload())
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `HTTP ${response.status}`);
        }

        const result = await response.json();

        emptyResult.classList.add("hidden");
        resultContent.classList.remove("hidden");

        const percentage = Number(result.risk_percentage || 0);

        document.getElementById("riskPercentage").textContent =
            `${percentage.toFixed(1)}%`;

        document.getElementById("riskLevel").textContent =
            result.risk_level || "Unknown";

        document.getElementById("predictionClass").textContent =
            result.prediction === 1 ? "Higher risk" : "Lower risk";

        document.getElementById("predictionText").textContent =
            result.prediction === 1
                ? "The model detected a higher predicted cardiovascular risk from the supplied parameters."
                : "The model detected a lower predicted cardiovascular risk from the supplied parameters.";

        setRiskVisual(percentage, result.risk_level);

        await loadHistory();

    } catch (error) {
        console.error(error);
        alert(
            "Could not connect to the backend. Make sure FastAPI is running and CORS is enabled."
        );
    } finally {
        predictBtn.disabled = false;
        predictBtn.querySelector(".btn-label").textContent = "Analyze health risk";
    }
});

async function loadHistory() {
    historyContainer.innerHTML =
        `<div class="loading-state">Loading prediction history...</div>`;

    try {
        const response = await fetch(`${API_URL}/history`);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const history = await response.json();

        if (!history.length) {
            historyContainer.innerHTML =
                `<div class="empty-history">No predictions have been recorded yet.</div>`;
            return;
        }

        const rows = history.map(record => {
            const date = record.created_at
                ? new Date(record.created_at).toLocaleString()
                : "—";

            const risk = Number(record.risk_percentage || 0).toFixed(1);

            return `
                <tr>
                    <td class="history-id">#${record.id}</td>
                    <td>${record.age}</td>
                    <td>${record.prediction === 1 ? "Higher risk" : "Lower risk"}</td>
                    <td>${risk}%</td>
                    <td><span class="table-risk">${record.risk_level || "—"}</span></td>
                    <td>${date}</td>
                </tr>
            `;
        }).join("");

        historyContainer.innerHTML = `
            <div class="history-table-wrap">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Age</th>
                            <th>Prediction</th>
                            <th>Risk</th>
                            <th>Level</th>
                            <th>Created</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;

    } catch (error) {
        console.error(error);
        historyContainer.innerHTML =
            `<div class="error-state">Unable to load history. Check that the FastAPI backend is running.</div>`;
    }
}

refreshBtn.addEventListener("click", loadHistory);

loadHistory();
