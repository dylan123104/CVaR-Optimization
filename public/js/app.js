const runBtn = document.getElementById("run-btn");
const statusEl = document.getElementById("status");

const baselineMetricsEl = document.getElementById("baseline-metrics");
const rebalanceMetricsEl = document.getElementById("rebalance-metrics");
const baselineTableEl = document.getElementById("baseline-table");
const rebalanceTableEl = document.getElementById("rebalance-table");

function getParams() {
  const rtRaw = document.getElementById("return_target").value.trim();
  let rtVal = "none";
  if (rtRaw.toLowerCase() === "equal_weight_mean") {
    rtVal = "equal_weight_mean";
  } else if (rtRaw !== "") {
    const num = Number(rtRaw);
    rtVal = isNaN(num) ? "none" : num;
  }
  return {
    tickers: document.getElementById("tickers").value || "AAPL,MSFT,NVDA,GOOGL,META,JPM,XOM",
    years: Number(document.getElementById("years").value || 3),
    alpha: Number(document.getElementById("alpha").value || 0.95),
    name_cap: Number(document.getElementById("name_cap").value || 0.2),
    sector_cap: Number(document.getElementById("sector_cap").value || 0.5),
    lambda_cvar: Number(document.getElementById("lambda_cvar").value || 0.8),
    buy_cost: Number(document.getElementById("buy_cost").value || 0.002),
    sell_cost: Number(document.getElementById("sell_cost").value || 0.002),
    turnover_cap: Number(document.getElementById("turnover_cap").value || 1.0),
    rebalance_day: document.getElementById("rebalance_day").value,
    return_target: rtVal,
    use_cached: document.getElementById("use_cached").checked,
  };
}

function card(label, value) {
  return `<div class="metric-card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

function renderMetrics(el, data, extra = {}) {
  const rows = [
    card("Mean return", data.mean_return.toFixed(6)),
    card("CVaR (loss)", data.CVaR_loss.toFixed(6)),
    card("VaR", data.VaR.toFixed(6)),
    card("Objective", data.objective.toFixed(6)),
  ];
  if (extra.turnover !== undefined) rows.push(card("Turnover", extra.turnover.toFixed(3)));
  el.innerHTML = rows.join("");
}

function renderTable(el, weights, buys, sells, extraInfo) {
  const tickers = Object.keys(weights);
  let html = "<table><tr><th>Ticker</th><th>Weight</th>";
  if (buys) html += "<th>Buy</th><th>Sell</th>";
  html += "</tr>";
  tickers.forEach(t => {
    html += `<tr><td>${t}</td><td>${weights[t].toFixed(4)}</td>`;
    if (buys) {
      html += `<td>${(buys[t] || 0).toFixed(4)}</td><td>${(sells[t] || 0).toFixed(4)}</td>`;
    }
    html += "</tr>";
  });
  html += "</table>";
  if (extraInfo) {
    html += `<div class="extra-info">${extraInfo}</div>`;
  }
  el.innerHTML = html;
}

function renderBar(divId, data, title) {
  const tickers = Object.keys(data);
  const vals = tickers.map(t => data[t]);
  const trace = { x: tickers, y: vals, type: "bar", marker: { color: "#6a8dff" } };
  const layout = { title, margin: { t: 30, l: 40, r: 10, b: 60 } };
  Plotly.newPlot(divId, [trace], layout, { displayModeBar: false });
}

function renderBuySell(divId, buy, sell) {
  const tickers = Object.keys({ ...buy, ...sell });
  const buyVals = tickers.map(t => buy[t] || 0);
  const sellVals = tickers.map(t => sell[t] || 0);
  const traces = [
    { x: tickers, y: buyVals, name: "Buy", type: "bar", marker: { color: "#2ecc71" } },
    { x: tickers, y: sellVals, name: "Sell", type: "bar", marker: { color: "#e74c3c" } },
  ];
  const layout = { title: "Buy / Sell", barmode: "group", margin: { t: 30, l: 40, r: 10, b: 60 } };
  Plotly.newPlot(divId, traces, layout, { displayModeBar: false });
}

function renderCvar(divId, losses, stats, title) {
  const trace = { x: losses, type: "histogram", marker: { color: "#9cb5ff" }, nbinsx: 80, histnorm: "probability" };
  const shapes = [
    { x0: stats.mean_loss_plot, x1: stats.mean_loss_plot, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: "green", dash: "dash" } },
    { x0: stats.VaR_plot, x1: stats.VaR_plot, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: "orange", dash: "dot" } },
    { x0: stats.CVaR_plot, x1: stats.CVaR_plot, y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: "red", dash: "dot" } },
  ];
  const layout = {
    title,
    margin: { t: 30, l: 40, r: 10, b: 40 },
    shapes,
    showlegend: false,
  };
  Plotly.newPlot(divId, [trace], layout, { displayModeBar: false });
}

async function runModels() {
  const params = getParams();
  statusEl.textContent = "Running…";
  runBtn.disabled = true;
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Request failed");
    }
    const data = await res.json();
    statusEl.textContent = "Done";
    // Baseline
    renderMetrics(baselineMetricsEl, data.baseline);
    const baseExtra = `
      Target return: ${data.baseline.target_return?.toFixed ? data.baseline.target_return.toFixed(6) : data.baseline.target_return}<br>
      Binding name caps: ${data.baseline.binding_names.join(", ") || "None"}<br>
      Binding sector caps: ${data.baseline.binding_sectors.join(", ") || "None"}
    `;
    renderTable(baselineTableEl, data.baseline.weights, null, null, baseExtra);
    renderBar("baseline-weights", data.baseline.weights, "Baseline weights");
    renderCvar("baseline-cvar", data.baseline.losses, {
      mean_loss_plot: data.baseline.mean_loss_plot,
      VaR_plot: data.baseline.VaR_plot,
      CVaR_plot: data.baseline.CVaR_plot,
    }, "Baseline CVaR");
    // Rebalance
    renderMetrics(rebalanceMetricsEl, data.rebalance, { turnover: data.rebalance.turnover });
    const rebExtra = `
      Target return: ${data.rebalance.target_return?.toFixed ? data.rebalance.target_return.toFixed(6) : data.rebalance.target_return}<br>
      Binding name caps (w0): ${data.rebalance.binding_names0.join(", ") || "None"}<br>
      Binding name caps (w1): ${data.rebalance.binding_names1.join(", ") || "None"}<br>
      Binding sector caps: ${data.rebalance.binding_sectors.join(", ") || "None"}
    `;
    renderTable(rebalanceTableEl, data.rebalance.w1, data.rebalance.buy, data.rebalance.sell, rebExtra);
    renderBar("rebalance-weights", data.rebalance.w1, "Rebalance weights (post)");
    renderBuySell("rebalance-buysells", data.rebalance.buy, data.rebalance.sell);
    renderCvar("rebalance-cvar", data.rebalance.losses, {
      mean_loss_plot: data.rebalance.mean_loss_plot,
      VaR_plot: data.rebalance.VaR_plot,
      CVaR_plot: data.rebalance.CVaR_plot,
    }, "Rebalance CVaR");
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runModels);

// Set default tickers and return target
document.getElementById("tickers").value = "AAPL,MSFT,NVDA,GOOGL,META,AMZN,JPM,XOM,CVX,UNH";
document.getElementById("return_target").value = "equal_weight_mean";
