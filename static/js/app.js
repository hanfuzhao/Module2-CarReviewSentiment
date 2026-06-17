const COLORS = { negative: "#d9534f", neutral: "#f0ad4e", positive: "#5cb85c" };

const EXAMPLES = {
  mixed: "The engine is powerful and acceleration is quick, but the fuel economy is terrible and the price is way too high. The seats are comfortable though.",
  negation: "The brakes are not good at all and the engine is not powerful. I would not call this car reliable.",
  litotes: "The ride is not uncomfortable and the brakes are not bad at all. You won't be disappointed by the engine.",
  sarcasm: "Great, another trip to the dealer this week. I just love spending every weekend fixing this thing. Such a smooth ride, my coffee ends up on the ceiling.",
  positive: "Absolutely love this car. The engine is smooth, the interior is comfortable and quiet, and the fuel economy is excellent. Best value I've ever found.",
};

const $ = (id) => document.getElementById(id);

let accuracies = {};

async function loadAccuracies() {
  // Pull each model's accuracy by doing a tiny analyze call is wasteful; instead
  // the badge is filled from the analyze response. Set a placeholder for now.
  updateAccBadge();
}

function updateAccBadge(acc) {
  const badge = $("model-acc");
  if (acc === undefined) { badge.textContent = ""; return; }
  badge.textContent = `test accuracy: ${(acc * 100).toFixed(1)}%`;
}

async function analyze() {
  const text = $("review").value.trim();
  if (!text) { return; }
  const model = $("model").value;
  const btn = $("analyze");
  btn.disabled = true; btn.textContent = "Analyzing...";
  $("error").classList.add("hidden");

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });
    const data = await res.json();
    if (!res.ok) { throw new Error(data.error || "Request failed"); }
    render(data);
  } catch (e) {
    $("error").textContent = "Error: " + e.message;
    $("error").classList.remove("hidden");
    $("results").classList.add("hidden");
  } finally {
    btn.disabled = false; btn.textContent = "Analyze";
  }
}

function render(data) {
  $("results").classList.remove("hidden");
  updateAccBadge(data.model_accuracy);

  // feedback
  const fb = $("feedback");
  fb.textContent = data.feedback.message;
  fb.className = "feedback " + data.feedback.level;

  // overall
  const o = data.overall;
  const lab = $("overall-label");
  lab.textContent = o.label;
  lab.style.background = o.color;
  $("overall-bar").style.width = (o.confidence * 100) + "%";
  $("overall-bar").style.background = o.color;
  $("overall-conf").textContent = (o.confidence * 100).toFixed(1) + "%";
  $("overall-probs").innerHTML = Object.entries(o.probs)
    .map(([k, v]) => `<span>${k}: ${(v * 100).toFixed(0)}%</span>`).join("");

  // aspects
  const list = $("aspect-list");
  list.innerHTML = "";
  if (!data.aspects.length) {
    $("no-aspects").classList.remove("hidden");
  } else {
    $("no-aspects").classList.add("hidden");
    for (const a of data.aspects) {
      const card = document.createElement("div");
      card.className = "aspect-card";
      card.style.borderLeftColor = a.color;
      const clause = a.mentions[0] ? a.mentions[0].clause : "";
      card.innerHTML = `
        <div class="aspect-head">
          <span class="aspect-name">${a.aspect}</span>
          <span class="aspect-sent" style="color:${a.color}">${a.sentiment}
            &middot; ${(a.confidence * 100).toFixed(0)}%${a.n > 1 ? ` &middot; ${a.n} mentions` : ""}</span>
        </div>
        <div class="aspect-clause">"${escapeHtml(clause)}"</div>`;
      list.appendChild(card);
    }
  }
}

function escapeHtml(s) {
  const d = document.createElement("div"); d.textContent = s; return d.innerHTML;
}

document.addEventListener("DOMContentLoaded", () => {
  $("analyze").addEventListener("click", analyze);
  document.querySelectorAll(".ex").forEach((b) =>
    b.addEventListener("click", () => {
      $("review").value = EXAMPLES[b.dataset.kind];
      analyze();
    }));
  $("review").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
  });
  loadAccuracies();
});
