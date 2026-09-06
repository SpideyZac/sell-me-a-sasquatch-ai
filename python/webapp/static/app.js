// Vanilla JS, no build step. One page, three modes toggled by tab buttons.

let MODELS = ["random"];

async function loadStaticData() {
  const modelsRes = await fetch("/api/models");
  const modelsJson = await modelsRes.json();
  MODELS = ["random", ...modelsJson.models];
}

function modelOptionsHtml(includeManual) {
  const manual = includeManual ? `<option value="manual">Manual (I'll enter their moves)</option>` : "";
  return manual + MODELS.map((m) => `<option value="${m}">${m}</option>`).join("");
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

function renderSeatModelSelects(container, numPlayers, excludeSeat, includeManual) {
  container.innerHTML = "";
  for (let i = 0; i < numPlayers; i++) {
    if (i === excludeSeat) continue;
    const label = document.createElement("label");
    label.textContent = `player_${i} model `;
    const select = document.createElement("select");
    select.dataset.seat = i;
    select.innerHTML = modelOptionsHtml(includeManual);
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

// Both modes put a human in one seat with AI (or random) opponents in the
// rest, backed by a real game session - so hands, collections, deals, and
// the draw/discard piles are all tracked exactly by the engine. Advisor
// additionally asks the server to rank the human's own legal actions with a
// model (see `score`/`recommended` on each action below); Play does not.

const seatedGames = {}; // prefix -> game_id

function initSeatedGame(prefix, mode) {
  const numSel = document.getElementById(`${prefix}-num-players`);
  const seatSel = document.getElementById(`${prefix}-human-seat`);
  const seatModelsDiv = document.getElementById(`${prefix}-seat-models`);
  const advisorModelSel = mode === "advisor" ? document.getElementById(`${prefix}-model`) : null;
  if (advisorModelSel) advisorModelSel.innerHTML = modelOptionsHtml();

  const refresh = () => {
    const n = parseInt(numSel.value, 10);
    seatSel.innerHTML = Array.from({ length: n }, (_, i) => `<option value="${i}">player_${i}</option>`).join("");
    renderSeatModelSelects(seatModelsDiv, n, parseInt(seatSel.value || "0", 10), mode === "advisor");
  };
  numSel.addEventListener("change", refresh);
  seatSel.addEventListener("change", refresh);
  refresh();

  document.getElementById(`${prefix}-start`).addEventListener("click", async () => {
    const numPlayers = parseInt(numSel.value, 10);
    const humanSeat = parseInt(seatSel.value, 10);
    const seatModels = collectSeatModels(seatModelsDiv, numPlayers, humanSeat);
    const seed = document.getElementById(`${prefix}-seed`).value;
    const body = { mode, num_players: numPlayers, human_seat: humanSeat, seat_models: seatModels, seed: seed || null };
    if (advisorModelSel) body.advisor_model = advisorModelSel.value;
    const res = await fetch("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) return alert(data.error);
    seatedGames[prefix] = data.game_id;
    document.getElementById(`${prefix}-setup`).hidden = true;
    document.getElementById(`${prefix}-view`).hidden = false;
    renderSeatedGameState(prefix, data.state);
  });

  document.getElementById(`${prefix}-new`).addEventListener("click", () => {
    delete seatedGames[prefix];
    document.getElementById(`${prefix}-setup`).hidden = false;
    document.getElementById(`${prefix}-view`).hidden = true;
  });
}

async function seatedGameAct(prefix, seat, actionIndex) {
  const res = await fetch(`/api/games/${seatedGames[prefix]}/act`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seat, action_index: actionIndex }),
  });
  const data = await res.json();
  if (data.error) return alert(data.error);
  renderSeatedGameState(prefix, data.state);
}

function actionButtonHtml(a) {
  const badge = "score" in a ? ` <small>(score ${a.score})</small>` : "";
  const star = a.recommended ? "★ " : "";
  return `<button class="action-btn${a.recommended ? " recommended" : ""}" data-idx="${a.index}">${star}${a.label}${badge}</button>`;
}

function renderSeatedGameState(prefix, state) {
  document.getElementById(`${prefix}-status`).innerHTML = `
    <b>You are:</b> player_${state.your_seat} &nbsp; <b>Phase:</b> ${state.phase} &nbsp;
    <b>Turn leader:</b> player_${state.turn_leader} &nbsp;
    <b>Draw pile:</b> ${state.draw_pile_len} &nbsp; <b>Discard pile:</b> ${state.discard_pile_len}
    ${state.is_game_over ? `<br><b class="winner">GAME OVER - winner: player_${state.winner}${state.winner === state.your_seat ? " (you win!)" : ""}</b>` : ""}
  `;

  document.getElementById(`${prefix}-hand`).innerHTML =
    `<h3>Your hand (${state.point_tokens[state.your_seat]} tokens)</h3>` + state.hand.map(cardLabel).join(", ");

  document.getElementById(`${prefix}-players`).innerHTML =
    "<h3>Everyone's public info</h3>" +
    state.collections
      .map((c, p) => `<div class="player-card"><b>player_${p}</b> - ${state.point_tokens[p]} tokens - collection: ${c.map(cardLabel).join(", ") || "(empty)"}</div>`)
      .join("");

  document.getElementById(`${prefix}-deals`).innerHTML =
    "<h3>Active deals</h3>" +
    (state.deals.length
      ? state.deals.map((d) => `<div class="deal">seller=player_${d.seller}: revealed [${d.revealed.map(cardLabel).join(", ") || "none"}], ${d.num_hidden} hidden</div>`).join("")
      : "<div>none</div>");

  const actionsDiv = document.getElementById(`${prefix}-actions`);
  if (state.acting_seat !== null && state.acting_seat !== undefined && !state.is_game_over) {
    const heading = state.your_turn
      ? "<h3>Your move</h3>"
      : `<h3>player_${state.acting_seat}'s move <small>(manual - enter what they actually did)</small></h3>
         <div class="manual-hand"><b>Their hand:</b> ${state.acting_hand.map(cardLabel).join(", ") || "(empty)"}</div>`;
    actionsDiv.innerHTML = heading + state.legal_actions.map(actionButtonHtml).join("");
    actionsDiv
      .querySelectorAll(".action-btn")
      .forEach((btn) => btn.addEventListener("click", () => seatedGameAct(prefix, state.acting_seat, parseInt(btn.dataset.idx, 10))));
  } else if (!state.is_game_over) {
    actionsDiv.innerHTML = "<h3>Waiting for other players...</h3>";
  } else {
    actionsDiv.innerHTML = "";
  }

  document.getElementById(`${prefix}-log`).innerHTML = "<h3>Log</h3>" + state.log.map((l) => `<div>${l}</div>`).join("");
}

// boot

(async function main() {
  await loadStaticData();
  setupTabs();
  initWatch();
  initSeatedGame("play", "play");
  initSeatedGame("advisor", "advisor");
})();
