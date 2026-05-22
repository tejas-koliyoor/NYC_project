const $ = (id) => document.getElementById(id);

function pretty(obj) {
    return JSON.stringify(obj, null, 2);
}

async function loadExample() {
    const out = $("out");
    out.textContent = "Loading example payload…";
    const res = await fetch("/example");
    if (!res.ok) {
        out.textContent = `Failed to load example: ${res.status} ${res.statusText}`;
        return;
    }
    const data = await res.json();
    $("json").value = pretty(data);
    out.textContent = "Example payload loaded ✅";
}

async function checkHealth() {
    const out = $("out");
    out.textContent = "Checking /health…";
    const res = await fetch("/health");
    const data = await res.json();
    out.textContent = pretty(data);
}

async function predict() {
    const out = $("out");
    out.textContent = "Calling /predict…";

    let payload;
    try {
        payload = JSON.parse($("json").value);
    } catch (e) {
        out.textContent = "Invalid JSON in textbox. Fix it and try again.";
        return;
    }

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    let data;
    try {
        data = await res.json();
    } catch {
        out.textContent = `Non-JSON response: ${res.status}`;
        return;
    }

    if (!res.ok) {
        out.textContent = pretty({ status: res.status, error: data });
        return;
    }

    out.textContent = pretty(data);
}

window.addEventListener("DOMContentLoaded", async () => {
    $("loadExample").addEventListener("click", loadExample);
    $("health").addEventListener("click", checkHealth);
    $("predict").addEventListener("click", predict);

    // Auto-load example on first open
    await loadExample();
});
