import numpy as np
import json
import random
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import re

data = np.load("artefacts/filtered_embeddings.npz")
W = data["W"]

with open("artefacts/filtered_word_idx.json", "r") as f:
    word_idx = json.load(f)

with open("artefacts/filtered_idx_word.json", "r") as f:
    idx_word = json.load(f)

idx_word = {int(k): v for k, v in idx_word.items()}


game_state = {
    "target": None, "target_id": None, "target_vec": None,
    "W_norm": None, "word_idx": None, "idx_word": None,
    "step": 0, "max_steps": 10, "guesses": [], "hints": [], "status": "playing"
}

def get_hint_word(sims, idx_word, rank_target, freq_min=75, freq_max=10000, exclude=None):
    """Return the word at `rank_target` similarity, skipping any words in `exclude`."""
    exclude = set(exclude) if exclude else set()
    sorted_ids = np.argsort(-sims)
    filtered = sorted_ids[np.isin(sorted_ids, np.arange(freq_min, freq_max))]
    
    # Walk through candidates, skipping already-guessed words
    count = 0
    for idx in filtered:
        word = idx_word[int(idx)]
        if word in exclude:
            continue
        if count == rank_target:
            return word
        count += 1
    return None  # Fallback: entire filtered vocab exhausted (extremely unlikely)


def init_game(W, word_idx, idx_word, max_steps=10, target=None):
    vocab_size = W.shape[0]
    W_norm = W / np.linalg.norm(W, axis=1, keepdims=True)
    if target is None:
        target = random.choice(list(word_idx.keys())[75:10000])
    target_id = word_idx[target]
    target_vec = W_norm[target_id]
    sims = np.dot(W_norm, target_vec)
    opening_hint = get_hint_word(sims, idx_word, rank_target=50)
    game_state.update({
        "target": target, "target_id": target_id,
        "target_vec": target_vec, "W_norm": W_norm,
        "word_idx": word_idx, "idx_word": idx_word,
        "step": 0, "max_steps": max_steps,
        "guesses": [], "hints": [], "status": "playing",
        "opening_hint": opening_hint
    })


def process_guess(guess):
    gs = game_state
    if gs["status"] != "playing":
        return {"error": "Game over."}
    guess = guess.strip().lower()
    if guess not in gs["word_idx"]:
        return {"error": "Word not in vocabulary."}
    
    guessed_words = {g["word"] for g in gs["guesses"]}
    if guess in guessed_words:
        return {"error": f"Already guessed '{guess}'."}
    if guess == gs.get("opening_hint"):
        return {"error": f"{guess} is a hint word."}
    
    gs["step"] += 1
    guess_id = gs["word_idx"][guess]
    if guess_id == gs["target_id"]:
        gs["status"] = "won"
        gs["guesses"].append({"word": guess, "rank": 1, "step": gs["step"]})
        return {"rank": 1, "won": True}
    sims = np.dot(gs["W_norm"], gs["target_vec"])
    rank = int(np.sum(sims > sims[guess_id]) + 1)
    gs["guesses"].append({"word": guess, "rank": rank, "step": gs["step"]})
    hint = None   

    guessed_words.add(guess) 


    if gs["step"] == 3:
        hw = get_hint_word(sims, gs["idx_word"], rank_target=10, exclude=guessed_words)
        hint = f'Hint (step 3): a related word is "{hw}"'

    if gs["step"] == 6:
        hw = get_hint_word(sims, gs["idx_word"], rank_target=5, exclude=guessed_words)
        hint = f'Hint (step 6): a closer word is "{hw}"'

    if gs["step"] == 9:
        hw = get_hint_word(sims, gs["idx_word"], rank_target=2, exclude=guessed_words)
        hint = f'Hint (step 9): a very close word is "{hw}"'
        
        gs["hints"].append(hint)                
    if gs["step"] >= gs["max_steps"]:
        gs["status"] = "lost"



    return {"rank": rank, "won": False, "hint": hint,
            "lost": gs["status"] == "lost",
            "target": gs["target"] if gs["status"] == "lost" else None}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Word Ascent</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&display=swap" rel="stylesheet"/>
<style>
  :root{
    --bg:#060a0f;--panel:#0d1117;--border:#1e2d3d;
    --accent:#00ffe7;--dim:#4a6b7c;--warn:#ff6b35;--good:#39ff14;
    --text:#c9d8e4;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Share Tech Mono',monospace;
       min-height:100vh;display:flex;flex-direction:column;align-items:center;
       padding:2rem 1rem;overflow-x:hidden;}
  body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:999;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.08) 2px,rgba(0,0,0,.08) 4px);}

  /* ── SCREENS ── */
  .screen{display:none;flex-direction:column;align-items:center;width:100%;max-width:540px;}
  .screen.active{display:flex;}

  /* ── SHARED TITLE ── */
  h1{font-family:'Bebas Neue',sans-serif;font-size:clamp(1.8rem,6vw,3.2rem);
     letter-spacing:.2em;color:var(--accent);text-shadow:0 0 30px var(--accent),0 0 60px rgba(0,255,231,.3);
     margin-bottom:.25rem;}
  .subtitle{color:var(--dim);font-size:.75rem;letter-spacing:.4em;margin-bottom:2rem;}

  /* ── MENU ── */
  .rules-box{width:100%;border:1px solid var(--border);background:var(--panel);
    padding:1.5rem;margin-bottom:2rem;}
  .rules-box h2{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:.2em;
    color:var(--accent);margin-bottom:1rem;border-bottom:1px solid var(--border);padding-bottom:.5rem;}
  .rules-box ul{list-style:none;display:flex;flex-direction:column;gap:.65rem;}
  .rules-box li{font-size:.82rem;line-height:1.6;color:var(--text);padding-left:1.2rem;position:relative;}
  .rules-box li::before{content:'›';position:absolute;left:0;color:var(--accent);}
  .rules-box .note{margin-top:1rem;padding:.6rem .8rem;border-left:3px solid var(--warn);
    color:var(--warn);font-size:.78rem;line-height:1.6;}

  /* ── PLAY BUTTON ── */
  .play-btn{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:.3em;
    padding:.7rem 3rem;background:transparent;border:2px solid var(--accent);color:var(--accent);
    cursor:pointer;transition:all .25s;position:relative;overflow:hidden;}
  .play-btn::after{content:'';position:absolute;inset:0;background:var(--accent);
    transform:translateX(-105%);transition:transform .25s ease;}
  .play-btn:hover::after{transform:translateX(0);}
  .play-btn span{position:relative;z-index:1;transition:color .25s;}
  .play-btn:hover span{color:var(--bg);}

  /* ── GAME ── */
  .steps-bar{width:100%;max-width:540px;display:flex;gap:4px;margin-bottom:2rem;}
  .step-pip{flex:1;height:6px;background:var(--border);border-radius:2px;transition:background .3s;}
  .step-pip.done{background:var(--accent);}
  .step-pip.danger{background:var(--warn);}
  .input-row{display:flex;gap:.5rem;width:100%;max-width:540px;margin-bottom:1.5rem;}
  #guessInput{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--accent);
    font-family:'Share Tech Mono',monospace;font-size:1.1rem;padding:.6rem 1rem;
    outline:none;caret-color:var(--accent);transition:border-color .2s;}
  #guessInput:focus{border-color:var(--accent);box-shadow:0 0 10px rgba(0,255,231,.2);}
  #guessInput::placeholder{color:var(--dim);}
  button{background:transparent;border:1px solid var(--accent);color:var(--accent);
    font-family:'Share Tech Mono',monospace;font-size:.9rem;padding:.6rem 1.2rem;
    cursor:pointer;letter-spacing:.1em;transition:all .2s;}
  button:hover{background:var(--accent);color:var(--bg);}
  #feedback{min-height:1.4rem;font-size:.85rem;color:var(--warn);
    margin-bottom:1rem;letter-spacing:.05em;text-align:center;}
  .log{width:100%;max-width:540px;display:flex;flex-direction:column;gap:.4rem;}
  .log-entry{display:flex;align-items:center;gap:1rem;padding:.5rem .8rem;
    background:var(--panel);border-left:3px solid var(--border);animation:slideIn .25s ease;}
  .log-entry.rank-top{border-left-color:var(--good);}
  .log-entry.rank-mid{border-left-color:var(--accent);}
  .log-entry.rank-low{border-left-color:var(--dim);}
  .log-entry.rank-bad{border-left-color:var(--warn);}
  .log-word{flex:1;font-size:1rem;color:#fff;}
  .log-rank{font-size:.85rem;color:var(--dim);}
  .log-rank span{color:var(--accent);}
  .log-entry.won-entry{border-left-color:var(--good);background:#0d1f0d;}
  .hint-box{width:100%;max-width:540px;margin-top:.5rem;padding:.6rem 1rem;
    background:#0d1117;border:1px dashed var(--warn);color:var(--warn);
    font-size:.8rem;letter-spacing:.05em;animation:slideIn .3s ease;}

  /* ── OVERLAY ── */
  .overlay{display:none;position:fixed;inset:0;background:rgba(6,10,15,.92);
    align-items:center;justify-content:center;z-index:100;flex-direction:column;gap:.75rem;
    padding:1.5rem;}
  .overlay.show{display:flex;}
  .overlay h2{font-family:'Bebas Neue',sans-serif;font-size:3rem;letter-spacing:.2em;}
  .overlay.won h2{color:var(--good);text-shadow:0 0 20px var(--good);}
  .overlay.lost h2{color:var(--warn);text-shadow:0 0 20px var(--warn);}
  .overlay p{color:var(--dim);font-size:.85rem;text-align:center;}
  .overlay .word-reveal{font-size:1.5rem;color:#fff;letter-spacing:.1em;}
  #overlayGuesses{width:100%;max-width:380px;max-height:260px;overflow-y:auto;
    flex-direction:column;gap:.3rem;display:none;}
  #overlayGuesses .log-entry{animation:none;}

  @keyframes slideIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:none}}
</style>
</head>
<body>

<!-- ══════════ MENU SCREEN ══════════ -->
<div class="screen active" id="menuScreen">
  <h1>Word Ascent</h1>
  <p class="subtitle">SEMANTIC WORD GUESSER</p>

  <div class="rules-box">
    <h2>How to Play</h2>
    <ul>
      <li>A secret target word has been chosen. Your goal is to guess it within <strong>10 attempts</strong>.</li>
      <li>Every word in the vocabulary is ranked by how semantically similar it is to the target. The closer a word is in meaning, the lower its rank — rank <strong>#1</strong> means you found the target.</li>
      <li>After each guess you receive a <strong>rank</strong> telling you how close you are. Use that signal to zero in on the target.</li>
      <li>At steps 3, 6, and 9 you receive a <strong>hint</strong> — a word that is semantically very close to the target.</li>
      <li>At the start of each game you also receive an <strong>opening hint</strong> to help you get your bearings.</li>
    </ul>
    <div class="note">
      ⚠ All words are <strong>base forms only</strong> — no plurals, no "-ing", no "-ed". If your guess isn't in the vocabulary, you'll be told. When in doubt, go with the simplest root form of the word.
    </div>
  </div>

  <button class="play-btn" onclick="startGame()"><span>PLAY</span></button>
</div>

<!-- ══════════ GAME SCREEN ══════════ -->
<div class="screen" id="gameScreen">
  <h1>Word Ascent</h1>
  <p class="subtitle">SEMANTIC WORD GUESSER</p>
  <div class="steps-bar" id="stepsBar"></div>
  <div class="input-row">
    <input id="guessInput" type="text" placeholder="enter a word..." autocomplete="off" autocorrect="off" spellcheck="false"/>
    <button id="submitBtn" onclick="submitGuess()">GUESS</button>
  </div>
  <div id="feedback"></div>
  <div class="log" id="log"></div>
</div>

<!-- ══════════ OVERLAY ══════════ -->
<div class="overlay" id="overlay">
  <h2 id="overlayTitle"></h2>
  <p id="overlayMsg"></p>
  <p class="word-reveal" id="overlayWord"></p>
  <button onclick="startGame()" style="margin-top:.5rem">PLAY AGAIN</button>
  <button id="viewGuessesBtn" onclick="toggleGuesses()">VIEW GUESSES</button>
  <div id="overlayGuesses"></div>
</div>

<script>
let MAX_STEPS = 10;
let step = 0;

function showScreen(id){
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function buildPips(){
  const bar=document.getElementById('stepsBar');
  bar.innerHTML='';
  for(let i=0;i<MAX_STEPS;i++){
    const d=document.createElement('div');
    d.className='step-pip';
    d.id='pip'+i;
    bar.appendChild(d);
  }
}

function updatePips(){
  for(let i=0;i<MAX_STEPS;i++){
    const p=document.getElementById('pip'+i);
    if(i<step) p.classList.add(i>=MAX_STEPS-4?'danger':'done');
  }
}

function rankClass(r){
  if(r<=50) return 'rank-top';
  if(r<=500) return 'rank-mid';
  if(r<=2000) return 'rank-low';
  return 'rank-bad';
}

async function submitGuess(){
  const inp=document.getElementById('guessInput');
  const word=inp.value.trim();
  if(!word) return;
  inp.value='';
  document.getElementById('feedback').textContent='';
  document.getElementById('submitBtn').disabled=true;
  const res=await fetch('/guess?word='+encodeURIComponent(word));
  const data=await res.json();
  document.getElementById('submitBtn').disabled=false;
  inp.focus();
  if(data.error){
    document.getElementById('feedback').textContent='⚠ '+data.error;
    return;
  }
  step++;
  updatePips();
  const log=document.getElementById('log');
  const entry=document.createElement('div');
  entry.className='log-entry '+(data.won?'won-entry':rankClass(data.rank));
  entry.innerHTML=`<span class="log-word">${word}</span><span class="log-rank">rank <span>#${data.rank}</span></span>`;
  log.prepend(entry);
  if(data.hint){
    const h=document.createElement('div');
    h.className='hint-box';
    h.textContent='💡 '+data.hint;
    log.prepend(h);
  }
  if(data.won) showOverlay(true,null,step);
  else if(data.lost) showOverlay(false,data.target,step);
}

function showOverlay(won,target,steps){
  const ov=document.getElementById('overlay');
  ov.classList.add('show',won?'won':'lost');
  document.getElementById('overlayTitle').textContent=won?'YOU GOT IT':'GAME OVER';
  document.getElementById('overlayMsg').textContent=won
    ?`Solved in ${steps} step${steps>1?'s':''}!`
    :`The word was:`;
  document.getElementById('overlayWord').textContent=won?'':target;
  // populate guesses from the live log
  const box=document.getElementById('overlayGuesses');
  box.innerHTML='';
  const entries=[...document.querySelectorAll('#log .log-entry')];
  [...entries].reverse().forEach(e=>{
    const clone=e.cloneNode(true);
    box.appendChild(clone);
  });
}

function toggleGuesses(){
  const box=document.getElementById('overlayGuesses');
  const btn=document.getElementById('viewGuessesBtn');
  const visible=box.style.display==='flex';
  box.style.display=visible?'none':'flex';
  btn.textContent=visible?'VIEW GUESSES':'HIDE GUESSES';
}

document.getElementById('guessInput').addEventListener('keydown',e=>{
  if(e.key==='Enter') submitGuess();
});

async function startGame(){
  document.getElementById('overlay').classList.remove('show','won','lost');
  document.getElementById('overlayGuesses').style.display='none';
  document.getElementById('viewGuessesBtn').textContent='VIEW GUESSES';
  const res=await fetch('/new');
  const data=await res.json();
  MAX_STEPS=data.max_steps;
  step=0;
  document.getElementById('log').innerHTML='';
  document.getElementById('feedback').textContent='';
  buildPips();
  updatePips();
  if(data.hint){
    const log=document.getElementById('log');
    const h=document.createElement('div');
    h.className='hint-box';
    h.textContent='💡 Starting hint: the word is semantically close to "'+data.hint+'"';
    log.appendChild(h);
  }
  showScreen('gameScreen');
  document.getElementById('guessInput').focus();
}
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path.startswith('/new'):
            init_game(W, word_idx, idx_word, game_state["max_steps"])
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "hint": game_state["opening_hint"], "max_steps": game_state["max_steps"]}).encode())
        elif self.path.startswith('/guess'):
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            word = params.get('word',[''])[0]
            result = process_guess(word)
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())            
        else:
            self.send_response(404); self.end_headers()
          

def play_contexto_ui(W, word_idx, idx_word, max_steps=15, target=None, port=8765):
    init_game(W, word_idx, idx_word, max_steps, target)
    server = HTTPServer(('localhost', port), Handler)
    url = f'http://localhost:{port}'
    print(f"[Word Ascent] Opening game at {url}")
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    server.serve_forever()

play_contexto_ui(W, word_idx, idx_word, max_steps=10)

