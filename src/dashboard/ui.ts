/**
 * Mind Palace Dashboard — UI Renderer (v2.0 — Step 8)
 *
 * Pure CSS + Vanilla JS single-page dashboard.
 * No build step, no Tailwind, no framework — served as a template literal.
 *
 * ═══════════════════════════════════════════════════════════════════
 * DESIGN:
 *   - Dark glassmorphism theme with purple/blue gradients
 *   - Animated neural network background
 *   - Auto-discovers projects on load
 *   - Real-time data from storage API
 *   - Responsive grid layout
 * ═══════════════════════════════════════════════════════════════════
 */

export function renderDashboardHTML(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Prism MCP — Mind Palace</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg-primary: #0a0e1a;
      --bg-secondary: #111827;
      --bg-glass: rgba(17, 24, 39, 0.6);
      --border-glass: rgba(139, 92, 246, 0.15);
      --border-glow: rgba(139, 92, 246, 0.3);
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent-purple: #8b5cf6;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --gradient-hero: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
      --radius: 16px;
      --radius-sm: 10px;
      --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }

    body {
      background: var(--bg-primary);
      color: var(--text-primary);
      font-family: var(--font-sans);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* ─── Animated Background ─── */
    .bg-grid {
      position: fixed; inset: 0; z-index: 0;
      background-image:
        radial-gradient(circle at 20% 30%, rgba(139,92,246,0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(59,130,246,0.06) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(6,182,212,0.04) 0%, transparent 60%);
      animation: bgPulse 8s ease-in-out infinite alternate;
    }
    @keyframes bgPulse {
      0% { opacity: 0.6; }
      100% { opacity: 1; }
    }

    .container { position: relative; z-index: 1; max-width: 1280px; margin: 0 auto; padding: 2rem; }

    /* ─── Header ─── */
    header {
      display: flex; justify-content: space-between; align-items: center;
      padding-bottom: 1.5rem; margin-bottom: 2rem;
      border-bottom: 1px solid var(--border-glass);
    }
    .logo {
      font-size: 1.75rem; font-weight: 700;
      background: var(--gradient-hero); -webkit-background-clip: text;
      background-clip: text; -webkit-text-fill-color: transparent;
      display: flex; align-items: center; gap: 0.5rem;
    }
    .logo-icon { -webkit-text-fill-color: initial; font-size: 1.5rem; }
    .version-badge {
      font-size: 0.7rem; font-weight: 500; padding: 0.2rem 0.6rem;
      border-radius: 999px; background: rgba(139,92,246,0.15);
      color: var(--accent-purple); border: 1px solid rgba(139,92,246,0.3);
      -webkit-text-fill-color: initial;
    }

    /* ─── Project Selector ─── */
    .selector {
      display: flex; gap: 0.75rem; align-items: center;
    }
    .selector select, .selector button {
      font-family: var(--font-sans); font-size: 0.875rem;
      border-radius: var(--radius-sm); outline: none;
      transition: all 0.2s ease;
    }
    .selector select {
      background: var(--bg-secondary); color: var(--text-primary);
      border: 1px solid var(--border-glass); padding: 0.6rem 1rem;
      min-width: 220px; cursor: pointer;
    }
    .selector select:hover { border-color: var(--border-glow); }
    .selector button {
      background: var(--gradient-hero); color: white; border: none;
      padding: 0.6rem 1.25rem; font-weight: 600; cursor: pointer;
    }
    .selector button:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(139,92,246,0.3); }
    .selector button:active { transform: translateY(0); }

    /* ─── Glass Cards ─── */
    .card {
      background: var(--bg-glass); backdrop-filter: blur(16px);
      border: 1px solid var(--border-glass); border-radius: var(--radius);
      padding: 1.5rem; transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover { border-color: var(--border-glow); box-shadow: 0 0 20px rgba(139,92,246,0.05); }
    .card-title {
      font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.1em; margin-bottom: 1rem; display: flex;
      align-items: center; gap: 0.5rem;
    }
    .card-title .dot {
      width: 8px; height: 8px; border-radius: 50%; display: inline-block;
    }

    /* ─── Grid Layout ─── */
    .grid { display: grid; gap: 1.5rem; }
    .grid-main { grid-template-columns: 1fr 2fr; }
    @media (max-width: 900px) { .grid-main { grid-template-columns: 1fr; } }

    /* ─── State Panel ─── */
    .summary-text { color: var(--text-secondary); font-size: 0.9rem; line-height: 1.7; margin-bottom: 1rem; }
    .todo-list { list-style: none; padding: 0; }
    .todo-list li {
      padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 0.85rem; color: var(--text-secondary);
      display: flex; align-items: flex-start; gap: 0.5rem;
    }
    .todo-list li::before { content: '→'; color: var(--accent-cyan); font-weight: 600; flex-shrink: 0; }
    .todo-list li:last-child { border-bottom: none; }

    /* ─── Git Metadata ─── */
    .git-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 0.85rem;
    }
    .git-row:last-child { border-bottom: none; }
    .git-label { color: var(--text-muted); }
    .git-value { font-family: var(--font-mono); color: var(--text-primary); font-size: 0.8rem; }

    /* ─── Timeline Items ─── */
    .timeline { display: flex; flex-direction: column; gap: 0.75rem; max-height: 400px; overflow-y: auto; }
    .timeline::-webkit-scrollbar { width: 4px; }
    .timeline::-webkit-scrollbar-track { background: transparent; }
    .timeline::-webkit-scrollbar-thumb { background: var(--border-glass); border-radius: 2px; }

    .timeline-item {
      padding: 0.875rem 1rem; background: rgba(15,23,42,0.6);
      border-radius: var(--radius-sm); border-left: 3px solid var(--accent-amber);
      font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;
      transition: background 0.2s ease;
    }
    .timeline-item:hover { background: rgba(15,23,42,0.9); }
    .timeline-item.history { border-left-color: var(--accent-purple); }
    .timeline-item .meta {
      font-size: 0.7rem; font-family: var(--font-mono);
      color: var(--text-muted); margin-bottom: 0.25rem;
      display: flex; justify-content: space-between;
    }
    .timeline-item .badge {
      display: inline-block; padding: 0.1rem 0.4rem; border-radius: 4px;
      font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    }
    .badge-purple { background: rgba(139,92,246,0.2); color: var(--accent-purple); }
    .badge-amber { background: rgba(245,158,11,0.2); color: var(--accent-amber); }
    .badge-green { background: rgba(16,185,129,0.2); color: var(--accent-green); }

    .briefing-text {
      font-size: 0.9rem; color: var(--text-secondary); line-height: 1.8;
      white-space: pre-wrap;
    }

    /* ─── Visual Memory ─── */
    .visual-list { list-style: none; padding: 0; }
    .visual-list li {
      padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 0.85rem; color: var(--text-secondary);
      display: flex; align-items: flex-start; gap: 0.5rem;
    }
    .visual-list li:last-child { border-bottom: none; }
    .visual-id {
      font-family: var(--font-mono); font-size: 0.75rem;
      color: var(--accent-rose); font-weight: 500;
    }
    .visual-date {
      font-size: 0.7rem; color: var(--text-muted); margin-left: auto;
      font-family: var(--font-mono); white-space: nowrap;
    }

    /* ─── Empty / Loading States ─── */
    .empty {
      text-align: center; padding: 3rem 1rem; color: var(--text-muted);
      font-size: 0.9rem;
    }
    .empty .emoji { font-size: 2.5rem; margin-bottom: 0.75rem; }
    .loading { display: none; text-align: center; padding: 2rem; color: var(--accent-purple); }
    .spinner {
      display: inline-block; width: 24px; height: 24px;
      border: 3px solid rgba(139,92,246,0.2); border-top-color: var(--accent-purple);
      border-radius: 50%; animation: spin 0.8s linear infinite;
      margin-right: 0.5rem; vertical-align: middle;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    #content { display: none; }

    /* ─── Fade in animation ─── */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .fade-in { animation: fadeIn 0.4s ease-out forwards; }
  </style>
</head>
<body>
  <div class="bg-grid"></div>
  <div class="container">
    <header>
      <div class="logo">
        <span class="logo-icon">🧠</span>
        Prism Mind Palace
        <span class="version-badge">v2.0</span>
      </div>
      <div class="selector">
        <select id="projectSelect">
          <option value="">Loading projects...</option>
        </select>
        <button onclick="loadProject()">Inspect</button>
      </div>
    </header>

    <div id="welcome" class="empty">
      <div class="emoji">🔮</div>
      <p>Select a project to inspect its neural state.</p>
    </div>

    <div id="loading" class="loading">
      <span class="spinner"></span> Neural link establishing...
    </div>

    <div id="content" class="grid grid-main fade-in">
      <!-- Left Column -->
      <div class="grid" style="align-content: start;">
        <!-- Current State -->
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--accent-blue)"></span> Current State <span id="versionBadge" class="badge badge-purple" style="margin-left:auto"></span></div>
          <div class="summary-text" id="summary"></div>
          <div class="card-title" style="margin-top:0.5rem"><span class="dot" style="background:var(--accent-cyan)"></span> Pending TODOs</div>
          <ul class="todo-list" id="todos"></ul>
        </div>

        <!-- Git Metadata -->
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--accent-green)"></span> Git Metadata</div>
          <div class="git-row"><span class="git-label">Branch</span><span class="git-value" id="gitBranch">—</span></div>
          <div class="git-row"><span class="git-label">Commit</span><span class="git-value" id="gitSha">—</span></div>
          <div class="git-row"><span class="git-label">Key Context</span><span class="git-value" id="keyContext" style="font-family:var(--font-sans);max-width:200px;text-align:right">—</span></div>
        </div>

        <!-- Morning Briefing -->
        <div class="card" id="briefingCard" style="display:none">
          <div class="card-title"><span class="dot" style="background:var(--accent-amber)"></span> Morning Briefing 🌅</div>
          <div class="briefing-text" id="briefingText"></div>
        </div>

        <!-- Visual Memory -->
        <div class="card" id="visualCard" style="display:none">
          <div class="card-title"><span class="dot" style="background:var(--accent-rose)"></span> Visual Memory 🖼️</div>
          <ul class="visual-list" id="visualList"></ul>
        </div>
      </div>

      <!-- Right Column -->
      <div class="grid" style="align-content: start;">
        <!-- Time Travel -->
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--accent-purple)"></span> Time Travel History 🕰️</div>
          <div class="timeline" id="historyTimeline"></div>
        </div>

        <!-- Ledger -->
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--accent-amber)"></span> Session Ledger</div>
          <div class="timeline" id="ledgerTimeline"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Auto-load project list on page load
    (async function() {
      try {
        const res = await fetch('/api/projects');
        const data = await res.json();
        const select = document.getElementById('projectSelect');
        if (data.projects && data.projects.length > 0) {
          select.innerHTML = '<option value="">— Select a project —</option>' +
            data.projects.map(function(p) { return '<option value="' + p + '">' + p + '</option>'; }).join('');
        } else {
          select.innerHTML = '<option value="">No projects found</option>';
        }
      } catch(e) {
        document.getElementById('projectSelect').innerHTML = '<option value="">Error loading projects</option>';
      }
    })();

    async function loadProject() {
      var project = document.getElementById('projectSelect').value;
      if (!project) return;

      document.getElementById('welcome').style.display = 'none';
      document.getElementById('content').style.display = 'none';
      document.getElementById('loading').style.display = 'block';

      try {
        var res = await fetch('/api/project?name=' + encodeURIComponent(project));
        var data = await res.json();

        // ─── Populate Context ───
        var ctx = data.context || {};
        document.getElementById('versionBadge').textContent = 'v' + (ctx.version || '?');
        document.getElementById('summary').textContent = ctx.last_summary || ctx.summary || 'No summary available.';

        var todos = ctx.pending_todo || ctx.active_context || [];
        var todoList = document.getElementById('todos');
        if (Array.isArray(todos) && todos.length > 0) {
          todoList.innerHTML = todos.map(function(t) { return '<li>' + escapeHtml(t) + '</li>'; }).join('');
        } else {
          todoList.innerHTML = '<li style="color:var(--text-muted)">No pending TODOs</li>';
        }

        // ─── Git ───
        var meta = ctx.metadata || {};
        document.getElementById('gitBranch').textContent = meta.git_branch || ctx.active_branch || '—';
        document.getElementById('gitSha').textContent = meta.last_commit_sha ? meta.last_commit_sha.substring(0, 12) : '—';
        document.getElementById('keyContext').textContent = ctx.key_context || '—';

        // ─── Morning Briefing ───
        var briefingCard = document.getElementById('briefingCard');
        if (meta.morning_briefing) {
          document.getElementById('briefingText').textContent = meta.morning_briefing;
          briefingCard.style.display = 'block';
        } else {
          briefingCard.style.display = 'none';
        }

        // ─── Visual Memory ───
        var visualCard = document.getElementById('visualCard');
        var visuals = meta.visual_memory || [];
        if (visuals.length > 0) {
          document.getElementById('visualList').innerHTML = visuals.map(function(v) {
            var dateStr = v.timestamp ? v.timestamp.split('T')[0] : '';
            return '<li><span class="visual-id">[' + escapeHtml(v.id) + ']</span> ' +
              escapeHtml(v.description) +
              '<span class="visual-date">' + dateStr + '</span></li>';
          }).join('');
          visualCard.style.display = 'block';
        } else {
          visualCard.style.display = 'none';
        }

        // ─── History Timeline ───
        var historyEl = document.getElementById('historyTimeline');
        if (data.history && data.history.length > 0) {
          historyEl.innerHTML = data.history.map(function(h) {
            var snap = h.snapshot || {};
            var summary = snap.last_summary || snap.summary || 'Snapshot';
            return '<div class="timeline-item history">' +
              '<div class="meta"><span class="badge badge-purple">v' + h.version + '</span>' +
              '<span>' + formatDate(h.created_at) + '</span></div>' +
              escapeHtml(summary) + '</div>';
          }).join('');
        } else {
          historyEl.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem;text-align:center">No time travel history yet.</div>';
        }

        // ─── Ledger Timeline ───
        var ledgerEl = document.getElementById('ledgerTimeline');
        if (data.ledger && data.ledger.length > 0) {
          ledgerEl.innerHTML = data.ledger.map(function(l) {
            var summary = l.summary || l.content || 'Entry';
            var decisions = l.decisions;
            var extra = '';
            if (decisions && decisions.length > 0) {
              try {
                var parsed = typeof decisions === 'string' ? JSON.parse(decisions) : decisions;
                if (Array.isArray(parsed) && parsed.length > 0) {
                  extra = '<div style="margin-top:0.3rem;font-size:0.75rem;color:var(--accent-cyan)">Decisions: ' + parsed.join(', ') + '</div>';
                }
              } catch(e) {}
            }
            return '<div class="timeline-item">' +
              '<div class="meta"><span class="badge badge-amber">session</span>' +
              '<span>' + formatDate(l.created_at) + '</span></div>' +
              escapeHtml(summary) + extra + '</div>';
          }).join('');
        } else {
          ledgerEl.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem;text-align:center">No ledger entries yet.</div>';
        }

        document.getElementById('content').className = 'grid grid-main fade-in';
        document.getElementById('content').style.display = 'grid';
      } catch(e) {
        alert('Failed to load project data: ' + e.message);
      } finally {
        document.getElementById('loading').style.display = 'none';
      }
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function formatDate(isoStr) {
      if (!isoStr) return '';
      try {
        var d = new Date(isoStr);
        return d.toLocaleDateString(undefined, { month:'short', day:'numeric' }) + ' ' +
               d.toLocaleTimeString(undefined, { hour:'2-digit', minute:'2-digit' });
      } catch(e) { return isoStr; }
    }

    // Allow Enter key in select to trigger load
    document.getElementById('projectSelect').addEventListener('change', loadProject);
  </script>
</body>
</html>`;
}
