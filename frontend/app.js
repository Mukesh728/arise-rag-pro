/**
 * MRSPTU StudyRAG - Interactive Frontend Application
 * ==================================================
 * Handles:
 * - Dynamic tab switching (registered FIRST in init)
 * - Study Assistant queries with exact citation grounding & guardrail alerts
 * - Retrieval and LLM latency reporting
 * - PDF syllabus upload & token chunking ingestion
 * - Viva Evaluation Benchmark runner (Hit Rate@K Vector vs Hybrid RRF)
 * - Metadata syncing & Corpus Manager
 */

function initMRSPTUApp() {
  // ---------------------------------------------------------
  // Global State
  // ---------------------------------------------------------
  const state = {
    selectedTab: "assistant",
    serverHealth: null,
    filterOptions: { subjects: [], units_by_subject: {}, total_documents: 0, total_chunks: 0 },
    lastQueryResponse: null,
    currentUser: null,
  };

  // ---------------------------------------------------------
  // Auth Verification & Profile UI
  // ---------------------------------------------------------
  async function checkAuth() {
    try {
      const res = await fetch("/api/auth/me");
      if (res.status === 401) {
        window.location.href = "/login";
        return null;
      }
      if (!res.ok) return null;
      const user = await res.json();
      state.currentUser = user;
      updateUserUI(user);
      return user;
    } catch (err) {
      console.warn("Auth check warning:", err);
      return null;
    }
  }

  function updateUserUI(user) {
    if (!user) return;
    const profileEmail = document.getElementById("profile-email");
    const profileRole = document.getElementById("profile-role");
    const profileAvatar = document.getElementById("profile-avatar");
    const railAvatar = document.getElementById("rail-avatar");

    if (profileEmail) {
      profileEmail.textContent = user.email || "";
      profileEmail.title = user.email || "";
    }
    if (profileRole) profileRole.textContent = user.role === "admin" ? "Administrator" : "Student";

    const emailPrefix = (user.email || "").split("@")[0];
    let initials = "CS";
    if (emailPrefix) {
      const parts = emailPrefix.split(/[._-]/).filter(Boolean);
      if (parts.length >= 2) {
        initials = (parts[0][0] + parts[1][0]).toUpperCase();
      } else if (emailPrefix.length >= 2) {
        initials = emailPrefix.substring(0, 2).toUpperCase();
      } else {
        initials = emailPrefix[0].toUpperCase();
      }
    }

    if (profileAvatar) profileAvatar.textContent = initials;
    if (railAvatar) railAvatar.textContent = initials;

    // PDF Ingest tab and action buttons are accessible to all logged-in accounts
    const tabBtnIngest = document.getElementById("tab-btn-ingest");
    const railIngest = document.querySelector('[data-go="ingest"]');
    const fabIngest = document.querySelector('.fab[data-go="ingest"]');
    if (tabBtnIngest) tabBtnIngest.style.display = "";
    if (railIngest) railIngest.style.display = "";
    if (fabIngest) fabIngest.style.display = "";

    // Fetch account-scoped uploaded documents
    fetchUserDocuments();
  }

  // ---------------------------------------------------------
  // 1. Tab Switching Logic - REGISTERED FIRST BEFORE ANY FETCH
  // ---------------------------------------------------------
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");

  if (tabButtons && tabButtons.length > 0) {
    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        if (!targetTab) return;
        state.selectedTab = targetTab;

        tabButtons.forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        if (tabContents) {
          tabContents.forEach((c) => c.classList.remove("active"));
        }

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        const targetContent = document.getElementById(`tab-${targetTab}`);
        if (targetContent) {
          targetContent.classList.add("active");
        }

        if (targetTab === "documents" || targetTab === "ingest") {
          refreshCorpusData();
          fetchUserDocuments();
        }
      });
    });
  }

  // ---------------------------------------------------------
  // DOM Elements Selection (with defensive lookup)
  // ---------------------------------------------------------
  // Header Elements
  const headerChunkCount = document.getElementById("header-chunk-count");
  const statusLabel = document.getElementById("status-label");
  const statusDot = document.getElementById("status-dot");

  // Query Form Elements
  const ragQueryForm = document.getElementById("rag-query-form");
  const queryInput = document.getElementById("query-input");
  const filterSubject = document.getElementById("filter-subject");
  const filterUnit = document.getElementById("filter-unit");
  const paramTopK = document.getElementById("param-top-k");
  const valTopK = document.getElementById("val-top-k");
  const paramThreshold = document.getElementById("param-threshold");
  const valThreshold = document.getElementById("val-threshold");
  const btnSubmitQuery = document.getElementById("btn-submit-query");

  // Results Panel Elements
  const resultsEmptyState = document.getElementById("results-empty-state");
  const resultsActiveContainer = document.getElementById("results-active-container");
  const resModeBadge = document.getElementById("res-mode-badge");
  const resTimeBadge = document.getElementById("res-time-badge");
  const resCitationCountBadge = document.getElementById("res-citation-count-badge");
  const resParamsBadge = document.getElementById("res-params-badge");
  const fallbackAlertBox = document.getElementById("fallback-alert-box");
  const fallbackDescText = document.getElementById("fallback-desc-text");
  const fallbackQuoteMsg = document.getElementById("fallback-quote-msg");
  const answerTextContent = document.getElementById("answer-text-content");
  const citationsList = document.getElementById("citations-list");
  const btnToggleChunks = document.getElementById("btn-toggle-chunks");
  const rawChunksDrawer = document.getElementById("raw-chunks-drawer");
  const btnCloseDrawer = document.getElementById("btn-close-drawer");
  const rawChunksTableContainer = document.getElementById("raw-chunks-table-container");
  const btnCopyAnswer = document.getElementById("btn-copy-answer");

  // Ingestion Elements
  const ingestPdfForm = document.getElementById("ingest-pdf-form");
  const pdfDropZone = document.getElementById("pdf-drop-zone");
  const pdfFileInput = document.getElementById("pdf-file-input");
  const selectedFilePreview = document.getElementById("selected-file-preview");
  const fileNameLabel = document.getElementById("file-name-label");
  const btnRemoveFile = document.getElementById("btn-remove-file");
  const inputSubject = document.getElementById("input-subject");
  const inputUnit = document.getElementById("input-unit");
  const adminTokenInput = document.getElementById("admin-token-input");
  const btnStartIngest = document.getElementById("btn-start-ingest");
  const ingestResultCard = document.getElementById("ingest-result-card");
  const ingestSuccessMsg = document.getElementById("ingest-success-msg");
  const ingestPagesStat = document.getElementById("ingest-pages-stat");
  const ingestChunksStat = document.getElementById("ingest-chunks-stat");
  const ingestDocidStat = document.getElementById("ingest-docid-stat");
  const btnGenerateSamplePdf = document.getElementById("btn-generate-sample-pdf");

  // Benchmark Elements
  const evalSubjectFilter = document.getElementById("eval-subject-filter");
  const evalTopK = document.getElementById("eval-top-k");
  const btnRunBenchmark = document.getElementById("btn-run-benchmark");
  const valVectorHitRate = document.getElementById("val-vector-hit-rate");
  const valVectorHits = document.getElementById("val-vector-hits");
  const barVectorFill = document.getElementById("bar-vector-fill");
  const valHybridHitRate = document.getElementById("val-hybrid-hit-rate");
  const valHybridHits = document.getElementById("val-hybrid-hits");
  const barHybridFill = document.getElementById("bar-hybrid-fill");
  const valDeltaImprovement = document.getElementById("val-delta-improvement");
  const benchmarkTableBody = document.getElementById("benchmark-table-body");

  // Corpus Elements
  const cstatDocs = document.getElementById("cstat-docs");
  const cstatChunks = document.getElementById("cstat-chunks");
  const cstatSubjects = document.getElementById("cstat-subjects");
  const subjectUnitsDirectory = document.getElementById("subject-units-directory");
  const btnRefreshCorpus = document.getElementById("btn-refresh-corpus");

  // Toast
  const toastNotification = document.getElementById("toast-notification");

  // Admin Token Storage (sessionStorage only)
  const ADMIN_TOKEN_KEY = "arise-admin-token";
  function getAdminToken() {
    try {
      return sessionStorage.getItem(ADMIN_TOKEN_KEY) || "";
    } catch (e) {
      return "";
    }
  }
  function setAdminToken(val) {
    try {
      if (val) {
        sessionStorage.setItem(ADMIN_TOKEN_KEY, val);
      } else {
        sessionStorage.removeItem(ADMIN_TOKEN_KEY);
      }
    } catch (e) {}
  }

  if (adminTokenInput) {
    adminTokenInput.value = getAdminToken();
    adminTokenInput.addEventListener("input", (e) => {
      setAdminToken(e.target.value.trim());
    });
  }

  // Logout Button
  const btnLogout = document.getElementById("btn-logout");
  if (btnLogout) {
    btnLogout.addEventListener("click", async () => {
      try {
        await fetch("/api/auth/logout", { method: "POST" });
      } catch (e) {}
      window.location.href = "/login";
    });
  }

  // History Elements
  const historyList = document.getElementById("history-list");
  const historyClear = document.getElementById("history-clear");
  const HISTORY_STORAGE_KEY = "arise-history";
  const MAX_HISTORY_ITEMS = 10;

  function getStoredHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      console.warn("Failed to read arise-history from localStorage:", e);
      return [];
    }
  }

  function saveHistoryItem(item) {
    if (!item || !item.question || !item.question.trim()) return;
    try {
      const current = getStoredHistory();
      const trimmedQ = item.question.trim();
      const filtered = current.filter(
        (h) => h && h.question && h.question.trim().toLowerCase() !== trimmedQ.toLowerCase()
      );
      filtered.unshift({
        question: trimmedQ,
        mode: item.mode || "hybrid",
        subject: item.subject || "all",
        timestamp: item.timestamp || Date.now(),
      });
      const trimmedList = filtered.slice(0, MAX_HISTORY_ITEMS);
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(trimmedList));
      renderHistoryList(trimmedList);
    } catch (e) {
      console.warn("Failed to save arise-history to localStorage:", e);
    }
  }

  function clearHistory() {
    try {
      localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch (e) {
      console.warn("Failed to clear arise-history from localStorage:", e);
    }
    renderHistoryList([]);
  }

  function renderHistoryList(items) {
    if (!historyList) return;
    historyList.innerHTML = "";
    if (!items || items.length === 0) {
      const emptyLi = document.createElement("li");
      emptyLi.className = "history-empty";
      emptyLi.textContent = "Questions you ask will show up here.";
      historyList.appendChild(emptyLi);
      return;
    }

    items.forEach((item) => {
      const li = document.createElement("li");
      li.className = "history-item";
      li.setAttribute("role", "button");
      li.setAttribute("tabindex", "0");
      li.title = item.question;

      const dot = document.createElement("span");
      dot.className = "history-dot";
      li.appendChild(dot);

      const textWrap = document.createElement("div");
      textWrap.className = "history-text";

      const qSpan = document.createElement("span");
      qSpan.className = "history-q";
      qSpan.textContent = item.question;
      textWrap.appendChild(qSpan);

      if (item.subject && item.subject !== "all") {
        const metaSpan = document.createElement("span");
        metaSpan.className = "history-meta";
        metaSpan.textContent = item.subject;
        textWrap.appendChild(metaSpan);
      }

      li.appendChild(textWrap);

      const onSelect = () => {
        if (queryInput) {
          queryInput.value = item.question;
          const assistantBtn =
            document.getElementById("tab-btn-assistant") ||
            document.querySelector('[data-tab="assistant"]');
          if (assistantBtn && !assistantBtn.classList.contains("active")) {
            assistantBtn.click();
          }
          queryInput.focus();
        }
      };

      li.addEventListener("click", onSelect);
      li.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      });

      historyList.appendChild(li);
    });
  }

  if (historyClear) {
    historyClear.addEventListener("click", () => {
      clearHistory();
    });
  }

  // Initial render of saved questions on page load
  renderHistoryList(getStoredHistory());

  // ---------------------------------------------------------
  // Helper: Toast Notifications
  // ---------------------------------------------------------
  function showToast(message, type = "info") {
    if (!toastNotification) {
      console.log(`[Toast ${type}]: ${message}`);
      return;
    }
    toastNotification.textContent = message;
    toastNotification.className = `toast-notification ${type} show`;
    setTimeout(() => {
      toastNotification.classList.remove("show");
    }, 3800);
  }

  // Slider inputs value syncing (debounced via requestAnimationFrame to avoid layout thrashing)
  if (paramTopK && valTopK) {
    let topKRaf = null;
    paramTopK.addEventListener("input", (e) => {
      if (topKRaf) cancelAnimationFrame(topKRaf);
      topKRaf = requestAnimationFrame(() => {
        valTopK.textContent = e.target.value;
      });
    });
  }

  if (paramThreshold && valThreshold) {
    let threshRaf = null;
    paramThreshold.addEventListener("input", (e) => {
      if (threshRaf) cancelAnimationFrame(threshRaf);
      threshRaf = requestAnimationFrame(() => {
        valThreshold.textContent = parseFloat(e.target.value).toFixed(2);
      });
    });
  }

  // Quick prompt chips
  document.querySelectorAll(".quick-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (queryInput) {
        queryInput.value = chip.getAttribute("data-prompt") || "";
        queryInput.focus();
      }
    });
  });

  // ---------------------------------------------------------
  // Data Fetching: Health & Metadata Filters
  // ---------------------------------------------------------
  async function fetchHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error(`Health check failed (${res.status})`);
      const data = await res.json();
      state.serverHealth = data;

      if (headerChunkCount) headerChunkCount.textContent = `${data.total_chunks || 0} Chunks`;
      if (statusLabel) statusLabel.textContent = "ChromaDB Online";
      if (statusDot) statusDot.className = "status-dot online";
    } catch (err) {
      console.warn("Server health poll warning:", err);
      if (statusLabel) statusLabel.textContent = "Connecting...";
      if (statusDot) statusDot.className = "status-dot";
    }
  }

  async function fetchFilters() {
    try {
      const res = await fetch("/api/filters");
      if (!res.ok) throw new Error(`Filters endpoint error (${res.status})`);
      const data = await res.json();
      state.filterOptions = data || { subjects: [], units_by_subject: {}, total_documents: 0, total_chunks: 0 };

      populateDropdowns(state.filterOptions);
      updateCorpusStats(state.filterOptions);
    } catch (err) {
      console.warn("Filters fetch warning:", err);
    }
  }

  function populateDropdowns(data) {
    if (!data) return;
    const subjects = data.subjects || [];

    // Query filter subject dropdown
    if (filterSubject) {
      const currentSubj = filterSubject.value;
      filterSubject.innerHTML = `<option value="all">All Subjects</option>`;
      subjects.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        if (s === currentSubj) opt.selected = true;
        filterSubject.appendChild(opt);
      });
    }

    // Evaluation benchmark subject dropdown
    if (evalSubjectFilter) {
      evalSubjectFilter.innerHTML = `<option value="all">All Subjects</option>`;
      subjects.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        evalSubjectFilter.appendChild(opt);
      });
    }

    updateUnitDropdown();
  }

  function updateUnitDropdown() {
    if (!filterSubject || !filterUnit) return;
    const selectedSubj = filterSubject.value;
    filterUnit.innerHTML = `<option value="all">All Units</option>`;

    if (selectedSubj && selectedSubj !== "all" && state.filterOptions && state.filterOptions.units_by_subject) {
      const units = state.filterOptions.units_by_subject[selectedSubj] || [];
      units.forEach((u) => {
        const opt = document.createElement("option");
        opt.value = u;
        const label = String(u).startsWith("Unit ") ? String(u) : `Unit ${u}`;
        opt.textContent = label;
        filterUnit.appendChild(opt);
      });
    }
  }

  if (filterSubject) {
    filterSubject.addEventListener("change", updateUnitDropdown);
  }

  function updateCorpusStats(data) {
    if (!data) return;
    if (cstatDocs) cstatDocs.textContent = data.total_documents || 0;
    if (cstatChunks) cstatChunks.textContent = data.total_chunks || 0;
    if (cstatSubjects) cstatSubjects.textContent = (data.subjects || []).length;

    renderSubjectDirectory(data);
  }

  function renderSubjectDirectory(data) {
    if (!subjectUnitsDirectory) return;
    const subjects = (data && data.subjects) || [];

    if (subjects.length === 0) {
      subjectUnitsDirectory.innerHTML = `
        <div class="empty-tree-msg">
          No syllabus materials indexed yet. Use the "PDF Syllabus Ingest" tab to upload your first document.
        </div>
      `;
      return;
    }

    subjectUnitsDirectory.innerHTML = "";
    subjects.forEach((subj) => {
      const units = (data.units_by_subject && data.units_by_subject[subj]) || [];
      const box = document.createElement("div");
      box.className = "subject-item-box";
      box.innerHTML = `
        <div class="subject-title-row">
          <div class="subject-heading">📚 ${escapeHtml(subj)}</div>
          <span class="chunk-badge">${units.length} Unit(s)</span>
        </div>
        <div class="units-badges-flex">
          ${units.map((u) => {
            const label = String(u).startsWith("Unit ") ? String(u) : `Unit ${u}`;
            return `<span class="unit-badge-pill">${escapeHtml(label)}</span>`;
          }).join("")}
        </div>
      `;
      subjectUnitsDirectory.appendChild(box);
    });
  }

  if (btnRefreshCorpus) {
    btnRefreshCorpus.addEventListener("click", () => {
      fetchHealth();
      fetchFilters();
      showToast("Corpus statistics refreshed.", "info");
    });
  }

  // ---------------------------------------------------------
  // RAG Query & Question Answering
  // ---------------------------------------------------------
  if (ragQueryForm) {
    ragQueryForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const query = queryInput ? queryInput.value.trim() : "";
      if (!query) return;

      const modeInput = document.querySelector('input[name="search_mode"]:checked');
      const mode = modeInput ? modeInput.value : "hybrid";
      const subject = filterSubject ? filterSubject.value : "all";
      const unit = filterUnit ? filterUnit.value : "all";
      const top_k = paramTopK ? parseInt(paramTopK.value, 10) : 5;
      const threshold = paramThreshold ? parseFloat(paramThreshold.value) : 0.32;

      // Save to recent questions history immediately at submit time
      saveHistoryItem({
        question: query,
        mode: mode,
        subject: subject,
        timestamp: Date.now(),
      });

      // UI Loading state with smooth skeleton loader
      if (btnSubmitQuery) {
        btnSubmitQuery.classList.add("loading");
        btnSubmitQuery.disabled = true;
      }
      if (resultsEmptyState) resultsEmptyState.style.display = "none";
      if (resultsActiveContainer) resultsActiveContainer.style.display = "block";
      if (fallbackAlertBox) fallbackAlertBox.style.display = "none";
      if (answerTextContent) {
        answerTextContent.innerHTML = `
          <div class="skeleton-loader" aria-label="Searching syllabus and synthesizing answer...">
            <div class="skeleton-line skeleton-title"></div>
            <div class="skeleton-line" style="width: 96%"></div>
            <div class="skeleton-line" style="width: 88%"></div>
            <div class="skeleton-line" style="width: 92%"></div>
            <div class="skeleton-line" style="width: 65%"></div>
          </div>
        `;
      }
      if (citationsList) {
        citationsList.innerHTML = `
          <div class="skeleton-card"></div>
          <div class="skeleton-card"></div>
        `;
      }

      const requestStartTime = performance.now();
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
      }, 90000); // 90 second timeout

      try {
        const payload = {
          query: query,
          question: query,
          subject: subject === "all" ? null : subject,
          subject_filter: subject === "all" ? null : subject,
          unit: unit === "all" ? null : unit,
          unit_filter: unit === "all" ? null : unit,
          mode: mode,
          strategy: mode === "hybrid" ? "hybrid_rrf" : "vector",
          top_k: parseInt(top_k, 10) || 5,
          threshold: parseFloat(threshold) || 0.32,
        };

        const res = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        const responseArrivalTime = performance.now();
        const roundTripLatencyMs = Math.round(responseArrivalTime - requestStartTime);

        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        state.lastQueryResponse = data;
        renderQueryResponse(data, roundTripLatencyMs);
      } catch (err) {
        if (err.name === "AbortError") {
          console.error("Query timed out after 90 seconds");
          showToast("Request timed out after 90 seconds. The server took too long to answer. Please try again.", "error");
          if (answerTextContent) {
            answerTextContent.innerHTML = `
              <div class="empty-state-card" style="padding: 20px; border-radius: var(--radius-md); background: var(--canvas); border: 1px solid var(--line);">
                <h4 style="margin: 0 0 6px; color: var(--red); font-size: 14px; font-weight: 600;">Request Timed Out</h4>
                <p style="margin: 0; color: var(--ink-2); font-size: 13px;">The retrieval request exceeded 90 seconds. Please check server load or try a more specific question.</p>
              </div>
            `;
          }
        } else {
          console.error("Query error:", err);
          showToast(`Error: ${err.message}`, "error");
          if (answerTextContent) {
            answerTextContent.innerHTML = `
              <div class="empty-state-card" style="padding: 20px; border-radius: var(--radius-md); background: var(--canvas); border: 1px solid var(--line);">
                <h4 style="margin: 0 0 6px; color: var(--red); font-size: 14px; font-weight: 600;">Unable to Generate Response</h4>
                <p style="margin: 0; color: var(--ink-2); font-size: 13px; line-height: 1.5;">${escapeHtml(err.message || "An unexpected error occurred while processing your request.")}</p>
                <p style="margin: 8px 0 0; color: var(--muted); font-size: 11.5px;">Note: If Gemini API quota is exhausted, retrieval endpoints and benchmarks continue working with grounded snippets.</p>
              </div>
            `;
          }
        }
        if (citationsList) citationsList.innerHTML = "";
      } finally {
        clearTimeout(timeoutId);
        if (btnSubmitQuery) {
          btnSubmitQuery.classList.remove("loading");
          btnSubmitQuery.disabled = false;
        }
      }
    });
  }

  function renderQueryResponse(data, clientLatencyMs) {
    if (!data) return;
    if (resultsEmptyState) resultsEmptyState.style.display = "none";
    if (resultsActiveContainer) resultsActiveContainer.style.display = "block";

    // Mode Badge
    if (resModeBadge) {
      resModeBadge.textContent = data.mode === "hybrid" ? "⚡ Hybrid RRF (k=60)" : "🔍 Dense Vector";
    }

    // Latency Badge: Show Retrieval, LLM, and Total
    const totalLatency = clientLatencyMs !== undefined ? clientLatencyMs : Math.round(data.execution_time_ms || 0);
    const retMs = data.retrieval_time_ms !== undefined ? Math.round(data.retrieval_time_ms) : null;
    const llmMs = data.llm_time_ms !== undefined ? Math.round(data.llm_time_ms) : null;
    if (resTimeBadge) {
      if (retMs !== null && llmMs !== null) {
        resTimeBadge.textContent = `⚡ Ret: ${retMs}ms | LLM: ${llmMs}ms | Total: ${totalLatency}ms`;
      } else {
        resTimeBadge.textContent = `⚡ ${totalLatency}ms`;
      }
    }

    if (resCitationCountBadge) {
      resCitationCountBadge.textContent = `${(data.citations || []).length} Citations`;
    }

    // Used params badge: show top_k, threshold, and strategy used for this query
    if (resParamsBadge) {
      const usedTopK = paramTopK ? paramTopK.value : 5;
      const usedThreshold = paramThreshold ? parseFloat(paramThreshold.value).toFixed(2) : "0.32";
      const usedStrategy = data.mode === "hybrid" ? "Hybrid RRF" : "Dense Vector";
      resParamsBadge.textContent = `🎛 Top ${usedTopK} chunks | Threshold: ${usedThreshold} | ${usedStrategy}`;
    }

    // Guardrail Fallback Warning
    if (fallbackAlertBox) {
      if (data.is_fallback) {
        fallbackAlertBox.style.display = "flex";
        if (fallbackQuoteMsg) fallbackQuoteMsg.textContent = `"${data.answer}"`;

        if (fallbackDescText) {
          if (data.fallback_message) {
            fallbackDescText.textContent = data.fallback_message;
          } else {
            const activeThreshold = (data.threshold !== undefined && data.threshold !== null)
              ? parseFloat(data.threshold).toFixed(2)
              : (paramThreshold ? parseFloat(paramThreshold.value).toFixed(2) : "0.32");
            fallbackDescText.textContent = `The similarity score of the uploaded material did not meet the confidence threshold (${activeThreshold}). The system prevented an ungrounded hallucination and returned the exact required fallback message:`;
          }
        }
      } else {
        fallbackAlertBox.style.display = "none";
      }
    }

    // Format Answer text with page citation pills
    const formattedAnswer = formatAnswerMarkdown(data.answer);
    if (answerTextContent) {
      answerTextContent.innerHTML = formattedAnswer;
    }

    // Render Citations
    renderCitations(data.citations || []);

    // Render Raw Chunks Table
    renderRawChunksTable(data.retrieved_chunks || []);
  }

  function formatAnswerMarkdown(text) {
    if (!text) return "";
    let clean = escapeHtml(text);

    // Convert [Page X] or [Page X, Y] into styled citation badges
    clean = clean.replace(/\[Page\s+(\d+(?:,\s*\d+)*)\]/gi, (match, pages) => {
      return `<span class="page-citation-pill">📄 Page ${pages}</span>`;
    });

    // Paragraph line breaks
    const paragraphs = clean.split("\n\n");
    return paragraphs
      .map((p) => {
        let pClean = p.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        pClean = pClean.replace(/`([^`]+)`/g, "<code>$1</code>");
        return `<p>${pClean.replace(/\n/g, "<br>")}</p>`;
      })
      .join("");
  }

  function renderCitations(citations) {
    if (!citationsList) return;
    citationsList.innerHTML = "";
    if (!citations || citations.length === 0) {
      citationsList.innerHTML = `<div class="empty-tree-msg">No citations met the threshold criteria.</div>`;
      return;
    }

    citations.forEach((c) => {
      const card = document.createElement("div");
      card.className = "citation-card";

      const scoreLabel =
        c.similarity_score !== null && c.similarity_score !== undefined
          ? `Score: ${(c.similarity_score * 100).toFixed(1)}%`
          : "";

      const rawUnit = c.unit ? String(c.unit).replace(/^Unit\s*/i, "").trim() : "";
      const unitBadge = rawUnit ? ` • Unit ${escapeHtml(rawUnit)}` : "";

      card.innerHTML = `
        <div class="citation-top-row">
          <div class="citation-badges-left">
            <span class="badge-page">📄 Page ${c.page_number}</span>
            <span class="badge-subject-unit">${escapeHtml(c.subject || "")}${unitBadge}</span>
            <span class="badge-subject-unit">${escapeHtml(c.filename || "")}</span>
          </div>
          <span class="citation-score-badge">${scoreLabel}</span>
        </div>
        <div class="citation-snippet-text">${escapeHtml(c.source_snippet)}</div>
      `;
      citationsList.appendChild(card);
    });
  }

  function renderRawChunksTable(chunks) {
    if (!rawChunksTableContainer) return;
    if (!chunks || chunks.length === 0) {
      rawChunksTableContainer.innerHTML = `<div class="empty-tree-msg">No raw chunks returned.</div>`;
      return;
    }

    const rows = chunks
      .map((c) => {
        const vScore = (c.vector_score !== null && c.vector_score !== undefined) ? Number(c.vector_score).toFixed(4) : "N/A";
        const bmScore = (c.bm25_score !== null && c.bm25_score !== undefined) ? Number(c.bm25_score).toFixed(4) : "N/A";
        const rrfScore = (c.rrf_score !== null && c.rrf_score !== undefined) ? Number(c.rrf_score).toFixed(6) : "N/A";
        const page = c.metadata ? c.metadata.page_number : "-";
        const snippet = escapeHtml((c.text || "").substring(0, 85)) + "...";

        return `
          <tr>
            <td><strong>#${c.rank || 1}</strong></td>
            <td><code>${c.chunk_id}</code></td>
            <td>Page ${page}</td>
            <td><span style="color: var(--accent-blue);">${vScore}</span></td>
            <td><span style="color: var(--accent-purple);">${bmScore}</span></td>
            <td><strong style="color: var(--accent-emerald);">${rrfScore}</strong></td>
            <td style="font-size: 0.78rem; color: var(--text-secondary);">${snippet}</td>
          </tr>
        `;
      })
      .join("");

    rawChunksTableContainer.innerHTML = `
      <table class="benchmark-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Chunk ID</th>
            <th>Page</th>
            <th>Vector Score</th>
            <th>BM25 Score</th>
            <th>RRF Fusion Score</th>
            <th>Snippet Preview</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  if (btnToggleChunks && rawChunksDrawer) {
    btnToggleChunks.addEventListener("click", () => {
      const isVisible = rawChunksDrawer.style.display === "block";
      rawChunksDrawer.style.display = isVisible ? "none" : "block";
    });
  }

  if (btnCloseDrawer && rawChunksDrawer) {
    btnCloseDrawer.addEventListener("click", () => {
      rawChunksDrawer.style.display = "none";
    });
  }

  if (btnCopyAnswer) {
    btnCopyAnswer.addEventListener("click", () => {
      if (state.lastQueryResponse && state.lastQueryResponse.answer) {
        navigator.clipboard.writeText(state.lastQueryResponse.answer);
        showToast("Answer copied to clipboard!", "success");
        const span = btnCopyAnswer.querySelector("span");
        btnCopyAnswer.classList.add("btn-copied");
        if (span) span.textContent = "Copied!";
        setTimeout(() => {
          btnCopyAnswer.classList.remove("btn-copied");
          if (span) span.textContent = "Copy";
        }, 1800);
      }
    });
  }

  // ---------------------------------------------------------
  // PDF Ingestion & File Handling
  // ---------------------------------------------------------
  if (pdfFileInput) {
    pdfFileInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      handleFileSelected(file);
    });
  }

  if (pdfDropZone) {
    pdfDropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      pdfDropZone.classList.add("drag-over");
    });

    pdfDropZone.addEventListener("dragleave", () => {
      pdfDropZone.classList.remove("drag-over");
    });

    pdfDropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      pdfDropZone.classList.remove("drag-over");
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        if (pdfFileInput) pdfFileInput.files = e.dataTransfer.files;
        handleFileSelected(file);
      }
    });
  }

  function handleFileSelected(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      showToast("Only PDF documents are supported.", "error");
      return;
    }
    if (fileNameLabel) fileNameLabel.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    if (selectedFilePreview) selectedFilePreview.style.display = "inline-flex";

    // Auto populate subject from filename if input is empty
    if (inputSubject && !inputSubject.value) {
      const cleanName = file.name.replace(/\.pdf$/i, "").replace(/[_-]/g, " ");
      inputSubject.value = cleanName;
    }
  }

  if (btnRemoveFile) {
    btnRemoveFile.addEventListener("click", (e) => {
      e.stopPropagation();
      if (pdfFileInput) pdfFileInput.value = "";
      if (selectedFilePreview) selectedFilePreview.style.display = "none";
    });
  }

  if (ingestPdfForm) {
    ingestPdfForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = pdfFileInput && pdfFileInput.files ? pdfFileInput.files[0] : null;
      if (!file) {
        showToast("Please choose a PDF file to upload.", "error");
        return;
      }

      const subject = inputSubject ? inputSubject.value.trim() : "";
      const unit = inputUnit ? inputUnit.value.trim() : "";

      if (!subject || !unit) {
        showToast("Please specify both Subject and Unit.", "error");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);
      formData.append("subject", subject);
      formData.append("unit", unit);

      if (btnStartIngest) {
        btnStartIngest.classList.add("loading");
        btnStartIngest.disabled = true;
      }
      if (ingestResultCard) ingestResultCard.style.display = "none";

      const headers = {};
      const adminToken = getAdminToken();
      if (adminToken) {
        headers["X-Admin-Token"] = adminToken;
      }

      try {
        const res = await fetch("/api/ingest", {
          method: "POST",
          headers: headers,
          body: formData,
        });

        if (res.status === 403) {
          throw new Error("Admin authorization required (403). Enter the valid Admin token above.");
        }

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || "PDF ingestion failed");
        }

        const data = await res.json();
        showToast(`Successfully indexed ${file.name}!`, "success");

        // Show Ingestion Result Card
        if (ingestSuccessMsg) ingestSuccessMsg.textContent = data.message;
        if (ingestPagesStat) ingestPagesStat.textContent = `Pages: ${data.total_pages}`;
        if (ingestChunksStat) ingestChunksStat.textContent = `Chunks: ${data.total_chunks}`;
        if (ingestDocidStat) ingestDocidStat.textContent = `Doc ID: ${data.doc_id}`;
        if (ingestResultCard) ingestResultCard.style.display = "flex";

        // Refresh filters, health count and account documents
        fetchHealth();
        fetchFilters();
        fetchUserDocuments();

        // Reset file input
        if (pdfFileInput) pdfFileInput.value = "";
        if (selectedFilePreview) selectedFilePreview.style.display = "none";
      } catch (err) {
        showToast(`Ingestion Error: ${err.message}`, "error");
      } finally {
        if (btnStartIngest) {
          btnStartIngest.classList.remove("loading");
          btnStartIngest.disabled = false;
        }
      }
    });
  }

  // ---------------------------------------------------------
  // Sample Syllabus Generator (For Instant Testing)
  // ---------------------------------------------------------
  if (btnGenerateSamplePdf) {
    btnGenerateSamplePdf.addEventListener("click", async () => {
      btnGenerateSamplePdf.disabled = true;
      btnGenerateSamplePdf.textContent = "Generating Sample Syllabus...";

      try {
        const headers = {};
        const adminToken = getAdminToken();
        if (adminToken) {
          headers["X-Admin-Token"] = adminToken;
        }
        const res = await fetch("/api/generate-sample-syllabus", {
          method: "POST",
          headers: headers,
        });
        if (res.status === 403) {
          throw new Error("Admin authorization required (403). Enter the valid Admin token in the Ingest tab.");
        }
        if (!res.ok) {
          throw new Error("Sample syllabus creation not supported directly or failed");
        }
        const data = await res.json();
        showToast(data.message || "Sample syllabus indexed!", "success");
        fetchHealth();
        fetchFilters();
      } catch (err) {
        showToast("Uploading a real course PDF via the dropzone above is recommended for testing.", "info");
      } finally {
        btnGenerateSamplePdf.disabled = false;
        btnGenerateSamplePdf.textContent = "⚡ Generate Sample OS Syllabus PDF";
      }
    });
  }

  // Exposed helper to delete document using admin token if set
  window.deleteDocument = async function (docId) {
    const headers = {};
    const adminToken = getAdminToken();
    if (adminToken) {
      headers["X-Admin-Token"] = adminToken;
    }
    const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
      method: "DELETE",
      headers: headers,
    });
    if (res.status === 403) {
      throw new Error("Admin authorization required (403). Enter the valid Admin token in the Ingest tab.");
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Document deletion failed");
    }
    return await res.json();
  };

  // ---------------------------------------------------------
  // User Uploaded Documents Tracking & Rendering
  // ---------------------------------------------------------
  async function fetchUserDocuments() {
    const ingestContainer = document.getElementById("user-docs-container");
    const corpusContainer = document.getElementById("corpus-user-docs-container");
    if (!ingestContainer && !corpusContainer) return;

    try {
      const res = await fetch("/api/user/documents");
      if (res.status === 401) {
        const msg = `<div style="padding: 18px; text-align: center; color: var(--muted); font-size: 13px;">Please log in to see documents uploaded to your account.</div>`;
        if (ingestContainer) ingestContainer.innerHTML = msg;
        if (corpusContainer) corpusContainer.innerHTML = msg;
        return;
      }
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }
      const data = await res.json();
      const docs = data.documents || [];
      if (ingestContainer) renderUserDocsList(docs, ingestContainer);
      if (corpusContainer) renderUserDocsList(docs, corpusContainer);
    } catch (err) {
      console.warn("Failed to fetch user documents:", err);
      const errMsg = `<div style="padding: 18px; text-align: center; color: var(--muted); font-size: 13px;">Unable to load account documents.</div>`;
      if (ingestContainer) ingestContainer.innerHTML = errMsg;
      if (corpusContainer) corpusContainer.innerHTML = errMsg;
    }
  }

  function renderUserDocsList(docs, targetEl) {
    if (!targetEl) return;
    if (!docs || docs.length === 0) {
      targetEl.innerHTML = `
        <div style="padding: 24px; text-align: center; background: var(--canvas-subtle, rgba(0,0,0,0.02)); border: 1px dashed var(--line); border-radius: var(--radius-md); color: var(--ink-2); font-size: 13px;">
          <p style="margin: 0 0 6px; font-weight: 600; color: var(--ink);">No documents uploaded yet under this account.</p>
          <span style="font-size: 12px; color: var(--muted);">Upload your course syllabus or lecture notes above to start asking questions!</span>
        </div>
      `;
      return;
    }

    const html = docs.map(doc => {
      const uploadDate = doc.uploaded_at ? new Date(doc.uploaded_at * 1000).toLocaleDateString(undefined, {
        month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit"
      }) : "Recently";

      return `
        <div class="user-doc-item" style="display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; margin-bottom: 10px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-md); box-shadow: var(--shadow-sm, none);">
          <div style="display: flex; align-items: flex-start; gap: 12px; min-width: 0;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: rgba(220, 38, 38, 0.1); color: #dc2626; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
              <svg viewBox="0 0 24 24" style="width: 20px; height: 20px; stroke: currentColor; fill: none; stroke-width: 2;">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <path d="M14 2v6h6" />
              </svg>
            </div>
            <div style="min-width: 0;">
              <div style="font-weight: 600; font-size: 13.5px; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(doc.filename)}">
                ${escapeHtml(doc.filename)}
              </div>
              <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; font-size: 11.5px;">
                <span style="background: rgba(59, 130, 246, 0.1); color: #2563eb; padding: 2px 8px; border-radius: 4px; font-weight: 500;">
                  📚 ${escapeHtml(doc.subject)}
                </span>
                <span style="background: rgba(16, 185, 129, 0.1); color: #059669; padding: 2px 8px; border-radius: 4px; font-weight: 500;">
                  📖 Unit ${escapeHtml(doc.unit)}
                </span>
                <span style="color: var(--muted); padding: 2px 4px;">
                  📄 ${doc.total_pages || 0} pages · 🧩 ${doc.total_chunks || 0} chunks
                </span>
                <span style="color: var(--muted); padding: 2px 4px;">
                  🕒 ${uploadDate}
                </span>
              </div>
            </div>
          </div>
          <div style="flex-shrink: 0; margin-left: 12px;">
            <button type="button" class="btn-soft delete-doc-btn" onclick="deleteUserDocumentHandler('${escapeHtml(doc.id)}', '${escapeHtml(doc.filename)}')" style="color: var(--red); border-color: rgba(220, 38, 38, 0.2); font-size: 12px; padding: 6px 10px; cursor: pointer;">
              🗑 Remove
            </button>
          </div>
        </div>
      `;
    }).join("");

    targetEl.innerHTML = html;
  }

  window.deleteUserDocumentHandler = async function(docId, filename) {
    if (!confirm(`Are you sure you want to remove "${filename}" from your account and the knowledge base?`)) {
      return;
    }
    try {
      const headers = {};
      const adminToken = getAdminToken();
      if (adminToken) {
        headers["X-Admin-Token"] = adminToken;
      }
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
        headers: headers
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Deletion failed");
      }
      showToast(`Removed "${filename}" successfully`, "success");
      fetchUserDocuments();
      fetchHealth();
      fetchFilters();
    } catch (err) {
      showToast(`Error deleting document: ${err.message}`, "error");
    }
  };

  const btnRefreshUserDocs = document.getElementById("btn-refresh-user-docs");
  if (btnRefreshUserDocs) {
    btnRefreshUserDocs.addEventListener("click", () => {
      fetchUserDocuments();
    });
  }

  const btnRefreshCorpusUserDocs = document.getElementById("btn-refresh-corpus-user-docs");
  if (btnRefreshCorpusUserDocs) {
    btnRefreshCorpusUserDocs.addEventListener("click", () => {
      fetchUserDocuments();
    });
  }

  // ---------------------------------------------------------
  // Viva Evaluation Benchmark Suite
  // ---------------------------------------------------------
  const defaultBenchmarkQueries = [
    {
      id: 1,
      question: "What are the seven graphical design concepts including Balance and visual weight in screen design?",
      expected_page: 3,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 2,
      question: "What is visible language and what are its primary components such as typography and layout?",
      expected_page: 4,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 3,
      question: "What is grid-based design and what role do columns, rows, gutters and margins play?",
      expected_page: 6,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 4,
      question: "Explain usability design principles and concepts derived from Don Norman such as affordance and feedback.",
      expected_page: 7,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 5,
      question: "What are Ben Shneiderman's Eight Golden Rules for interface design?",
      expected_page: 8,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 6,
      question: "Explain Jakob Nielsen's 10 Usability Heuristics for user interface evaluation.",
      expected_page: 10,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 7,
      question: "Compare Shneiderman's Golden Rules with Nielsen's Heuristics in terms of design guidelines vs evaluation checklist.",
      expected_page: 12,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 8,
      question: "What is Heuristic Evaluation and why are 3 to 5 expert evaluators recommended?",
      expected_page: 13,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 9,
      question: "What is an HCI design pattern and when should designers use reusable patterns?",
      expected_page: 14,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 10,
      question: "How should a student structure answers in an HCI viva including definition, purpose and example?",
      expected_page: 15,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
    {
      id: 11,
      question: "What are the core subjects and course codes listed in MRSPTU 7th Semester B.Tech CSE study scheme?",
      expected_page: 2,
      subject: "CSE Sem 7 Syllabus",
      unit: null,
    },
    {
      id: 12,
      question: "What are the course objectives, outcomes, and contents of Soft Computing BCSED1-712?",
      expected_page: 3,
      subject: "CSE Sem 7 Syllabus",
      unit: null,
    },
    {
      id: 13,
      question: "What are the syllabus topics and recommended textbooks by Dix and Finlay for Human Computer Interaction BCSED1-713?",
      expected_page: 4,
      subject: "CSE Sem 7 Syllabus",
      unit: "Unit 4",
    },
    {
      id: 14,
      question: "What are the course outcomes and syllabus topics for Cryptography & Network Security BCSED1-723?",
      expected_page: 5,
      subject: "CSE Sem 7 Syllabus",
      unit: "Unit 4",
    },
    {
      id: 15,
      question: "What topics are covered in Artificial Intelligence BCSED1-724 including heuristic search and knowledge representation?",
      expected_page: 6,
      subject: "CSE Sem 7 Syllabus",
      unit: "Unit 4",
    },
    {
      id: 16,
      question: "What natural resources and forest or water conservation topics are covered in Environmental Sciences Unit-I?",
      expected_page: 7,
      subject: "CSE Sem 7 Syllabus",
      unit: "Unit 1",
    },
    {
      id: 17,
      question: "What environmental pollution types and human health topics are covered in Environmental Sciences?",
      expected_page: 8,
      subject: "CSE Sem 7 Syllabus",
      unit: "Unit 4",
    },
    {
      id: 18,
      question: "Explain the concepts of ecosystems, producers, consumers, and food chains in the syllabus.",
      expected_page: 9,
      subject: "CSE Sem 7 Syllabus",
      unit: "Unit 4",
    },
    {
      id: 19,
      question: "What topics are covered in Constitution of India BMNCC0-001 including Directive Principles and Article 19 and 21?",
      expected_page: 10,
      subject: "CSE Sem 7 Syllabus",
      unit: "Unit 4",
    },
    {
      id: 20,
      question: "What is the difference between Recognition vs Recall and Affordance vs Mapping in UI design?",
      expected_page: 12,
      subject: "Human Computer Interaction",
      unit: "Unit 3",
    },
  ];

  if (btnRunBenchmark) {
    btnRunBenchmark.addEventListener("click", async () => {
      btnRunBenchmark.classList.add("loading");
      btnRunBenchmark.disabled = true;

      const selectedSubj = (evalSubjectFilter && evalSubjectFilter.value !== "all") ? evalSubjectFilter.value : null;
      const topK = evalTopK ? parseInt(evalTopK.value, 10) : 3;

      const payload = {
        queries: defaultBenchmarkQueries,
        subject: selectedSubj,
        top_k: topK,
      };

      try {
        const res = await fetch("/api/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || "Benchmark run failed");
        }

        const report = await res.json();
        renderBenchmarkReport(report);
        showToast("Evaluation Benchmark complete!", "success");
      } catch (err) {
        showToast(`Benchmark Error: ${err.message}`, "error");
      } finally {
        btnRunBenchmark.classList.remove("loading");
        btnRunBenchmark.disabled = false;
      }
    });
  }

  function renderBenchmarkReport(report) {
    if (!report) return;
    const vRate = ((report.vector_hit_rate || 0) * 100).toFixed(1);
    const hRate = ((report.hybrid_hit_rate || 0) * 100).toFixed(1);
    const delta = ((report.delta_improvement || 0) * 100).toFixed(1);

    const isReduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function countUp(el, targetNum, prefix = "", suffix = "%") {
      if (!el) return;
      if (isReduced) {
        el.textContent = `${prefix}${targetNum.toFixed(1)}${suffix}`;
        return;
      }
      const startTime = performance.now();
      const duration = 450;
      function step(now) {
        const p = Math.min((now - startTime) / duration, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        const cur = (targetNum * ease).toFixed(1);
        el.textContent = `${prefix}${cur}${suffix}`;
        if (p < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    const vNum = parseFloat(vRate) || 0;
    const hNum = parseFloat(hRate) || 0;
    const dNum = parseFloat(delta) || 0;

    countUp(valVectorHitRate, vNum, "", "%");
    countUp(valHybridHitRate, hNum, "", "%");
    countUp(valDeltaImprovement, Math.abs(dNum), dNum >= 0 ? "+" : "-", "%");

    if (valVectorHits) valVectorHits.textContent = `Hits: ${report.vector_hits || 0} / ${report.total_queries || 0} queries`;
    if (barVectorFill) barVectorFill.style.width = `${vRate}%`;

    if (valHybridHits) valHybridHits.textContent = `Hits: ${report.hybrid_hits || 0} / ${report.total_queries || 0} queries`;
    if (barHybridFill) barHybridFill.style.width = `${hRate}%`;

    // Render detailed comparison table
    if (!benchmarkTableBody) return;
    const rows = (report.detailed_results || []).map((r) => {
      const vBadge = r.vector_hit
        ? `<span class="status-badge hit">✓ Hit</span>`
        : `<span class="status-badge miss">✗ Miss</span>`;
      const hBadge = r.hybrid_hit
        ? `<span class="status-badge hit">✓ Hit</span>`
        : `<span class="status-badge miss">✗ Miss</span>`;

      let winner = `<span class="winner-badge winner-tie">Tie</span>`;
      if (r.hybrid_hit && !r.vector_hit) {
        winner = `<span class="winner-badge winner-hybrid">⚡ Hybrid Win</span>`;
      } else if (r.vector_hit && !r.hybrid_hit) {
        winner = `<span class="winner-badge">Vector Win</span>`;
      }

      const vPages = (r.vector_top_pages && r.vector_top_pages.length > 0) ? r.vector_top_pages.map((p) => `P${p}`).join(", ") : "-";
      const hPages = (r.hybrid_top_pages && r.hybrid_top_pages.length > 0) ? r.hybrid_top_pages.map((p) => `P${p}`).join(", ") : "-";

      return `
        <tr>
          <td><strong>#${r.id}</strong></td>
          <td>${escapeHtml(r.question)}</td>
          <td><span class="badge-page">Page ${r.expected_page}</span></td>
          <td><code>[${vPages}]</code></td>
          <td>${vBadge}</td>
          <td><code>[${hPages}]</code></td>
          <td>${hBadge}</td>
          <td>${winner}</td>
        </tr>
      `;
    });

    benchmarkTableBody.innerHTML = rows.join("");
  }

  // ---------------------------------------------------------
  // Corpus Data Refresh
  // ---------------------------------------------------------
  function refreshCorpusData() {
    fetchHealth();
    fetchFilters();
  }

  // Utility HTML Escape
  function escapeHtml(text) {
    if (!text) return "";
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ---------------------------------------------------------
  // Initial Data Fetching (Auth Check, Health & Metadata Filters)
  // ---------------------------------------------------------
  checkAuth();

  try {
    fetchHealth();
  } catch (err) {
    console.warn("Initial fetchHealth error:", err);
  }
  try {
    fetchFilters();
  } catch (err) {
    console.warn("Initial fetchFilters error:", err);
  }
}

// ---------------------------------------------------------
// Bootstrap: execute immediately if already loaded, else on DOMContentLoaded
// ---------------------------------------------------------
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initMRSPTUApp);
} else {
  initMRSPTUApp();
}
