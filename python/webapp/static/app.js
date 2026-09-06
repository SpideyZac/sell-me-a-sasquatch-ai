// Vanilla JS, no build step. One page, three modes toggled by tab buttons.

let MODELS = ["random"];
let CARD_CLASSES = []; // [{value, label}]

async function loadStaticData() {
  const [modelsRes, classesRes] = await Promise.all([fetch("/api/models"), fetch("/api/card_classes")]);
  const modelsJson = await modelsRes.json();
  const classesJson = await classesRes.json();
  MODELS = ["random", ...modelsJson.models];
  CARD_CLASSES = classesJson.classes;
}

function modelOptionsHtml() {
  return MODELS.map((m) => `<option value="${m}">${m}</option>`).join("");
}

function cardClassOptionsHtml(includeEmpty) {
  const empty = includeEmpty ? `<option value="">-- empty --</option>` : "";
  return empty + CARD_CLASSES.map((c) => `<option value="${c.value}">${c.label}</option>`).join("");
}

function cardLabel(card) {
  return card.name ? `${card.name}` : `card#${card.id}`;
}

// tabs

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

// shared: seat-model selects

function renderSeatModelSelects(container, numPlayers, excludeSeat) {
  container.innerHTML = "";
  for (let i = 0; i < numPlayers; i++) {
    if (i === excludeSeat) continue;
    const label = document.createElement("label");
    label.textContent = `player_${i} model `;
    const select = document.createElement("select");
    select.dataset.seat = i;
    select.innerHTML = modelOptionsHtml();
    label.appendChild(select);
    container.appendChild(label);
  }
}

function collectSeatModels(container, numPlayers, humanSeat) {
  const seatModels = new Array(numPlayers).fill("random");
  container.querySelectorAll("select").forEach((sel) => {
    seatModels[parseInt(sel.dataset.seat, 10)] = sel.value;
  });
  if (humanSeat !== undefined) seatModels[humanSeat] = "random"; // unused for the human's own seat
  return seatModels;
}

// ============================== WATCH ========================================

let watchGameId = null;
let watchAutoplayTimer = null;

function initWatch() {
  const numSel = document.getElementById("watch-num-players");
  const seatModelsDiv = document.getElementById("watch-seat-models");
  const refreshSeats = () => renderSeatModelSelects(seatModelsDiv, parseInt(numSel.value, 10));
  numSel.addEventListener("change", refreshSeats);
  refreshSeats();

  document.getElementById("watch-start").addEventListener("click", async () => {
    const numPlayers = parseInt(numSel.value, 10);
    const seatModels = collectSeatModels(seatModelsDiv, numPlayers);
    const seed = document.getElementById("watch-seed").value;
    const res = await fetch("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "watch", num_players: numPlayers, seat_models: seatModels, seed: seed || null }),
    });
    const data = await res.json();
    if (data.error) return alert(data.error);
    watchGameId = data.game_id;
    document.getElementById("watch-setup").hidden = true;
    document.getElementById("watch-view").hidden = false;
    renderWatchState(data.state);
  });

  document.getElementById("watch-step").addEventListener("click", async () => {
    if (!watchGameId) return;
    const res = await fetch(`/api/games/${watchGameId}/advance`, { method: "POST" });
    const data = await res.json();
    renderWatchState(data.state);
  });

  document.getElementById("watch-autoplay").addEventListener("change", (e) => {
    if (e.target.checked) {
      watchAutoplayTimer = setInterval(async () => {
        if (!watchGameId) return;
        const res = await fetch(`/api/games/${watchGameId}/advance`, { method: "POST" });
        const data = await res.json();
        renderWatchState(data.state);
        if (data.state.is_game_over) {
          clearInterval(watchAutoplayTimer);
          e.target.checked = false;
        }
      }, 500);
    } else {
      clearInterval(watchAutoplayTimer);
    }
  });

  document.getElementById("watch-new").addEventListener("click", () => {
    clearInterval(watchAutoplayTimer);
    document.getElementById("watch-autoplay").checked = false;
    watchGameId = null;
    document.getElementById("watch-setup").hidden = false;
    document.getElementById("watch-view").hidden = true;
  });
}

function renderWatchState(state) {
  document.getElementById("watch-status").innerHTML = `
    <b>Phase:</b> ${state.phase} &nbsp; <b>Turn leader:</b> player_${state.turn_leader} &nbsp;
    <b>Active:</b> ${state.active_players.map((p) => "player_" + p).join(", ")}
    ${state.is_game_over ? `<br><b class="winner">GAME OVER - winner: player_${state.winner}</b>` : ""}
  `;

  const playersDiv = document.getElementById("watch-players");
  playersDiv.innerHTML = state.hands
    .map((hand, p) => {
      const collection = state.collections[p];
      return `<div class="player-card">
        <h3>player_${p} ${p === state.turn_leader ? "(buyer/leader)" : ""} - ${state.point_tokens[p]} tokens
          <small>(${state.seat_specs[p]})</small></h3>
        <div><b>Hand:</b> ${hand.map(cardLabel).join(", ") || "(empty)"}</div>
        <div><b>Collection:</b> ${collection.map(cardLabel).join(", ") || "(empty)"}</div>
      </div>`;
    })
    .join("");

  const dealsDiv = document.getElementById("watch-deals");
  dealsDiv.innerHTML =
    "<h3>Active deals</h3>" +
    (state.deals.length
      ? state.deals
          .map(
            (d) =>
              `<div class="deal">seller=player_${d.seller}: revealed [${d.revealed.map(cardLabel).join(", ") || "none"}], ${d.num_hidden} hidden</div>`
          )
          .join("")
      : "<div>none</div>");

  document.getElementById("watch-log").innerHTML = "<h3>Log</h3>" + state.log.map((l) => `<div>${l}</div>`).join("");
}

// ============================== PLAY =========================================

let playGameId = null;

function initPlay() {
  const numSel = document.getElementById("play-num-players");
  const seatSel = document.getElementById("play-human-seat");
  const seatModelsDiv = document.getElementById("play-seat-models");

  const refresh = () => {
    const n = parseInt(numSel.value, 10);
    seatSel.innerHTML = Array.from({ length: n }, (_, i) => `<option value="${i}">player_${i}</option>`).join("");
    renderSeatModelSelects(seatModelsDiv, n, parseInt(seatSel.value || "0", 10));
  };
  numSel.addEventListener("change", refresh);
  seatSel.addEventListener("change", refresh);
  refresh();

  document.getElementById("play-start").addEventListener("click", async () => {
    const numPlayers = parseInt(numSel.value, 10);
    const humanSeat = parseInt(seatSel.value, 10);
    const seatModels = collectSeatModels(seatModelsDiv, numPlayers, humanSeat);
    const seed = document.getElementById("play-seed").value;
    const res = await fetch("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "play", num_players: numPlayers, human_seat: humanSeat, seat_models: seatModels, seed: seed || null }),
    });
    const data = await res.json();
    if (data.error) return alert(data.error);
    playGameId = data.game_id;
    document.getElementById("play-setup").hidden = true;
    document.getElementById("play-view").hidden = false;
    renderPlayState(data.state);
  });

  document.getElementById("play-new").addEventListener("click", () => {
    playGameId = null;
    document.getElementById("play-setup").hidden = false;
    document.getElementById("play-view").hidden = true;
  });
}

async function playAct(actionIndex) {
  const res = await fetch(`/api/games/${playGameId}/act`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action_index: actionIndex }),
  });
  const data = await res.json();
  if (data.error) return alert(data.error);
  renderPlayState(data.state);
}

function renderPlayState(state) {
  document.getElementById("play-status").innerHTML = `
    <b>You are:</b> player_${state.your_seat} &nbsp; <b>Phase:</b> ${state.phase} &nbsp;
    <b>Turn leader:</b> player_${state.turn_leader}
    ${state.is_game_over ? `<br><b class="winner">GAME OVER - winner: player_${state.winner}${state.winner === state.your_seat ? " (you win!)" : ""}</b>` : ""}
  `;

  document.getElementById("play-hand").innerHTML =
    `<h3>Your hand (${state.point_tokens[state.your_seat]} tokens)</h3>` + state.hand.map(cardLabel).join(", ");

  document.getElementById("play-players").innerHTML =
    "<h3>Everyone's public info</h3>" +
    state.collections
      .map((c, p) => `<div class="player-card"><b>player_${p}</b> - ${state.point_tokens[p]} tokens - collection: ${c.map(cardLabel).join(", ") || "(empty)"}</div>`)
      .join("");

  document.getElementById("play-deals").innerHTML =
    "<h3>Active deals</h3>" +
    (state.deals.length
      ? state.deals.map((d) => `<div class="deal">seller=player_${d.seller}: revealed [${d.revealed.map(cardLabel).join(", ") || "none"}], ${d.num_hidden} hidden</div>`).join("")
      : "<div>none</div>");

  const actionsDiv = document.getElementById("play-actions");
  if (state.your_turn && !state.is_game_over) {
    actionsDiv.innerHTML = "<h3>Your move</h3>" + state.legal_actions.map((a) => `<button class="action-btn" data-idx="${a.index}">${a.label}</button>`).join("");
    actionsDiv.querySelectorAll(".action-btn").forEach((btn) => btn.addEventListener("click", () => playAct(parseInt(btn.dataset.idx, 10))));
  } else if (!state.is_game_over) {
    actionsDiv.innerHTML = "<h3>Waiting for other players...</h3>";
  } else {
    actionsDiv.innerHTML = "";
  }

  document.getElementById("play-log").innerHTML = "<h3>Log</h3>" + state.log.map((l) => `<div>${l}</div>`).join("");
}

// ============================== ADVISOR ======================================

function initAdvisorDealOffer() {
  const modelSel = document.getElementById("adv-deal-model");
  modelSel.innerHTML = modelOptionsHtml();

  const handDiv = document.getElementById("adv-deal-hand");
  handDiv.innerHTML = Array.from({ length: 5 }, (_, i) => `<label>Card ${i + 1} <select data-slot="${i}">${cardClassOptionsHtml(i >= 3)}</select></label>`).join("");

  document.getElementById("adv-deal-go").addEventListener("click", async () => {
    const numPlayers = parseInt(document.getElementById("adv-deal-num-players").value, 10);
    const hand = Array.from(handDiv.querySelectorAll("select"))
      .map((s) => s.value)
      .filter((v) => v !== "");
    if (hand.length < 3) return alert("Enter at least 3 cards.");
    const res = await fetch("/api/advisor/deal_offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hand, num_players: numPlayers, model: modelSel.value }),
    });
    const data = await res.json();
    const resultDiv = document.getElementById("adv-deal-result");
    if (data.error) {
      resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
      return;
    }
    resultDiv.innerHTML = `
      <p class="disclaimer">${data.disclaimer}</p>
      <h3>Ranked deals to offer</h3>
      <ol>${data.recommendations.map((r) => `<li>${r.cards.join(", ")} <small>(score ${r.score})</small></li>`).join("")}</ol>
      ${data.reveal_recommendation ? `<p><b>For the top combo, reveal:</b> ${data.reveal_recommendation}</p>` : ""}
    `;
  });
}

function initAdvisorChooseDeal() {
  const modelSel = document.getElementById("adv-choose-model");
  modelSel.innerHTML = modelOptionsHtml();
  const numSel = document.getElementById("adv-choose-num-players");
  const sellersDiv = document.getElementById("adv-choose-sellers");

  const refreshSellers = () => {
    const n = parseInt(numSel.value, 10) - 1;
    sellersDiv.innerHTML = Array.from(
      { length: n },
      (_, i) => `
      <div class="seller-slot">
        <b>Seller ${i + 1}</b>
        <label>Revealed card <select data-seller="${i}" class="seller-revealed">${cardClassOptionsHtml(true)}</select></label>
        <label>Hidden cards <input type="number" data-seller="${i}" class="seller-hidden" min="0" max="3" value="2"></label>
      </div>`
    ).join("");
  };
  numSel.addEventListener("change", refreshSellers);
  refreshSellers();

  document.getElementById("adv-choose-go").addEventListener("click", async () => {
    const numPlayers = parseInt(numSel.value, 10);
    const revealedSelects = Array.from(sellersDiv.querySelectorAll(".seller-revealed"));
    const hiddenInputs = Array.from(sellersDiv.querySelectorAll(".seller-hidden"));
    const sellers = revealedSelects.map((sel, i) => ({
      revealed: sel.value ? [sel.value] : [],
      num_hidden: parseInt(hiddenInputs[i].value, 10) || 0,
    }));
    const res = await fetch("/api/advisor/choose_deal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sellers, num_players: numPlayers, model: modelSel.value }),
    });
    const data = await res.json();
    const resultDiv = document.getElementById("adv-choose-result");
    if (data.error) {
      resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
      return;
    }
    resultDiv.innerHTML = `
      <p class="disclaimer">${data.disclaimer}</p>
      <h3>Ranked sellers to choose</h3>
      <ol>${data.recommendations.map((r) => `<li>Seller ${r.seller_slot + 1} <small>(score ${r.score})</small></li>`).join("")}</ol>
    `;
  });
}

// boot

(async function main() {
  await loadStaticData();
  setupTabs();
  initWatch();
  initPlay();
  initAdvisorDealOffer();
  initAdvisorChooseDeal();
})();
