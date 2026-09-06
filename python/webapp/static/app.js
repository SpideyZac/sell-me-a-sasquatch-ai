// Vanilla JS, no build step. One page, three modes toggled by tab buttons.

let MODELS = ["random"];

async function loadStaticData() {
  const modelsRes = await fetch("/api/models");
  const modelsJson = await modelsRes.json();
  MODELS = ["random", ...modelsJson.models];
}

function modelOptionsHtml() {
  return MODELS.map((m) => `<option value="${m}">${m}</option>`).join("");
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

// shared: "who goes first / buys first" select (blank = random)

function renderFirstPlayerSelect(select, numPlayers) {
  const previous = select.value;
  select.innerHTML =
    `<option value="">Random</option>` + Array.from({ length: numPlayers }, (_, i) => `<option value="${i}">player_${i}</option>`).join("");
  if (previous && parseInt(previous, 10) < numPlayers) select.value = previous;
}

function readFirstPlayer(select) {
  return select.value === "" ? null : parseInt(select.value, 10);
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
  const firstPlayerSel = document.getElementById("watch-first-player");
  const refreshSeats = () => {
    renderSeatModelSelects(seatModelsDiv, parseInt(numSel.value, 10));
    renderFirstPlayerSelect(firstPlayerSel, parseInt(numSel.value, 10));
  };
  numSel.addEventListener("change", refreshSeats);
  refreshSeats();

  document.getElementById("watch-start").addEventListener("click", async () => {
    const numPlayers = parseInt(numSel.value, 10);
    const seatModels = collectSeatModels(seatModelsDiv, numPlayers);
    const seed = document.getElementById("watch-seed").value;
    const res = await fetch("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "watch",
        num_players: numPlayers,
        seat_models: seatModels,
        first_player: readFirstPlayer(firstPlayerSel),
        seed: seed || null,
      }),
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
  const firstPlayerSel = document.getElementById("play-first-player");

  const refresh = () => {
    const n = parseInt(numSel.value, 10);
    seatSel.innerHTML = Array.from({ length: n }, (_, i) => `<option value="${i}">player_${i}</option>`).join("");
    renderSeatModelSelects(seatModelsDiv, n, parseInt(seatSel.value || "0", 10));
    renderFirstPlayerSelect(firstPlayerSel, n);
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
      body: JSON.stringify({
        mode: "play",
        num_players: numPlayers,
        human_seat: humanSeat,
        seat_models: seatModels,
        first_player: readFirstPlayer(firstPlayerSel),
        seed: seed || null,
      }),
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
    <b>Turn leader:</b> player_${state.turn_leader} &nbsp;
    <b>Draw pile:</b> ${state.draw_pile_len} &nbsp; <b>Discard pile:</b> ${state.discard_pile_len}
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

// ============================== ADVISOR (Live Game tracker) =================
// For an actual live physical game: no simulated deck, no AI seats. It's
// the real rules engine underneath (via /api/live), so collections, point
// totals, discard/draw piles, and set trade-ins are all handled for you -
// you just answer whatever the current prompt is asking for (your starting
// hand, a card as it's revealed, or your own move) and the state below
// updates to match.

let liveGameId = null;

function initAdvisor() {
  const numSel = document.getElementById("advisor-num-players");
  const seatSel = document.getElementById("advisor-human-seat");
  const firstPlayerSel = document.getElementById("advisor-first-player");
  const modelSel = document.getElementById("advisor-model");
  modelSel.innerHTML = modelOptionsHtml();

  const refresh = () => {
    const n = parseInt(numSel.value, 10);
    seatSel.innerHTML = Array.from({ length: n }, (_, i) => `<option value="${i}">player_${i}</option>`).join("");
    renderFirstPlayerSelect(firstPlayerSel, n);
  };
  numSel.addEventListener("change", refresh);
  refresh();

  document.getElementById("advisor-start").addEventListener("click", async () => {
    const numPlayers = parseInt(numSel.value, 10);
    const humanSeat = parseInt(seatSel.value, 10);
    const seed = document.getElementById("advisor-seed").value;
    const res = await fetch("/api/live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        num_players: numPlayers,
        human_seat: humanSeat,
        first_player: readFirstPlayer(firstPlayerSel),
        seed: seed || null,
        advisor_model: modelSel.value,
      }),
    });
    const data = await res.json();
    if (data.error) return alert(data.error);
    liveGameId = data.live_id;
    document.getElementById("advisor-setup").hidden = true;
    document.getElementById("advisor-view").hidden = false;
    renderLiveState(data.state);
  });

  document.getElementById("advisor-new").addEventListener("click", () => {
    liveGameId = null;
    document.getElementById("advisor-setup").hidden = false;
    document.getElementById("advisor-view").hidden = true;
  });
}

async function liveRespond(payload) {
  const res = await fetch(`/api/live/${liveGameId}/respond`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.error) alert(data.error);
  if (data.state) renderLiveState(data.state);
}

function kindSelectHtml(kindOptions) {
  const opts = kindOptions
    .map((k) => `<option value="${k.value}" ${k.remaining <= 0 ? "disabled" : ""}>${k.label} (${k.remaining} left)</option>`)
    .join("");
  return `<select class="kind-select">${opts}</select>`;
}

function liveActionButtonHtml(a) {
  const badge = "score" in a ? ` <small>(score ${a.score})</small>` : "";
  const star = a.recommended ? "★ " : "";
  return `<button class="action-btn${a.recommended ? " recommended" : ""}" data-idx="${a.index}">${star}${a.label}${badge}</button>`;
}

function renderLivePrompt(state) {
  const promptDiv = document.getElementById("advisor-prompt");
  const prompt = state.prompt;

  if (prompt.type === "pin_hand") {
    const heading = prompt.context === "starting_hand" ? "Enter your starting hand" : "What did you draw?";
    promptDiv.innerHTML = `
      <h3>${heading}</h3>
      <div class="pin-slots">${Array.from({ length: prompt.count }, () => kindSelectHtml(state.kind_options)).join("")}</div>
      <button id="prompt-submit">Confirm</button>
    `;
    document.getElementById("prompt-submit").addEventListener("click", () => {
      liveRespond({ kinds: Array.from(promptDiv.querySelectorAll(".kind-select")).map((s) => s.value) });
    });
    return;
  }

  if (prompt.type === "pin_resolution") {
    promptDiv.innerHTML = `
      <h3>The deal is resolving - enter the still-hidden card(s)</h3>
      <div class="pin-slots">${prompt.cards.map((c) => `<label>player_${c.seller}'s hidden card #${c.position} ${kindSelectHtml(state.kind_options)}</label>`).join("")}</div>
      <button id="prompt-submit">Confirm</button>
    `;
    document.getElementById("prompt-submit").addEventListener("click", () => {
      liveRespond({ kinds: Array.from(promptDiv.querySelectorAll(".kind-select")).map((s) => s.value) });
    });
    return;
  }

  if (prompt.type === "reveal_kind") {
    promptDiv.innerHTML = `<h3>${prompt.label}</h3>${kindSelectHtml(state.kind_options)}<button id="prompt-submit">Confirm</button>`;
    document.getElementById("prompt-submit").addEventListener("click", () => {
      liveRespond({ kind: promptDiv.querySelector(".kind-select").value });
    });
    return;
  }

  if (prompt.type === "choose_action") {
    const heading = prompt.your_turn ? "Your move" : `player_${prompt.seat}'s move <small>(enter what actually happened)</small>`;
    promptDiv.innerHTML = `<h3>${heading}</h3>` + prompt.options.map(liveActionButtonHtml).join("");
    promptDiv.querySelectorAll(".action-btn").forEach((btn) => btn.addEventListener("click", () => liveRespond({ index: parseInt(btn.dataset.idx, 10) })));
    return;
  }

  promptDiv.innerHTML = ""; // game_over - status line already shows the winner
}

function renderLiveState(state) {
  document.getElementById("advisor-status").innerHTML = `
    <b>You are:</b> player_${state.your_seat} &nbsp; <b>Phase:</b> ${state.phase} &nbsp;
    <b>Turn leader:</b> player_${state.turn_leader} &nbsp;
    <b>Draw pile:</b> ${state.draw_pile_len} &nbsp; <b>Discard pile:</b> ${state.discard_pile_len}
    ${state.is_game_over ? `<br><b class="winner">GAME OVER - winner: player_${state.winner}${state.winner === state.your_seat ? " (you win!)" : ""}</b>` : ""}
  `;

  document.getElementById("advisor-hand").innerHTML =
    `<h3>Your hand (${state.point_tokens[state.your_seat]} tokens)</h3>` + (state.hand.map(cardLabel).join(", ") || "(empty)");

  document.getElementById("advisor-players").innerHTML =
    "<h3>Everyone's public info</h3>" +
    state.collections
      .map((c, p) => `<div class="player-card"><b>player_${p}</b> - ${state.point_tokens[p]} tokens - collection: ${c.map(cardLabel).join(", ") || "(empty)"}</div>`)
      .join("");

  document.getElementById("advisor-deals").innerHTML =
    "<h3>Active deals</h3>" +
    (state.deals.length
      ? state.deals.map((d) => `<div class="deal">seller=player_${d.seller}: revealed [${d.revealed.map(cardLabel).join(", ") || "none"}], ${d.num_hidden} hidden</div>`).join("")
      : "<div>none</div>");

  renderLivePrompt(state);

  document.getElementById("advisor-log").innerHTML = "<h3>Log</h3>" + state.log.map((l) => `<div>${l}</div>`).join("");
}

// boot

(async function main() {
  await loadStaticData();
  setupTabs();
  initWatch();
  initPlay();
  initAdvisor();
})();
