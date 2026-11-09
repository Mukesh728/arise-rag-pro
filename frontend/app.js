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
  };

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

        if (targetTab === "documents") {
          refreshCorpusData();
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

  // Slider inputs value syncing
  if (paramTopK && valTopK) {
    paramTopK.addEventListener("input", (e) => {
      valTopK.textContent = e.target.value;
    });
  }

  if (paramThreshold && valThreshold) {
    paramThreshold.addEventListener("input", (e) => {
      valThreshold.textContent = parseFloat(e.target.value).toFixed(2);
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
      const top_k = paramTopK ? parseInt(paramTopK.value, 10) : 3;
      const threshold = paramThreshold ? parseFloat(paramThreshold.value) : 0.32;

      // UI Loading state
      if (btnSubmitQuery) {
        btnSubmitQuery.classList.add("loading");
        btnSubmitQuery.disabled = true;
      }
      const requestStartTime = performance.now();

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
          top_k: parseInt(top_k, 10) || 3,
          threshold: parseFloat(threshold) || 0.32,
        };

        const res = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
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
        console.error("Query error:", err);
        showToast(`Error: ${err.message}`, "error");
      } finally {
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
      const usedTopK = paramTopK ? paramTopK.value : 3;
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

      try {
        const res = await fetch("/api/ingest", {
          method: "POST",
          body: formData,
        });

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

        // Refresh filters and health count
        fetchHealth();
        fetchFilters();

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
        const res = await fetch("/api/generate-sample-syllabus", { method: "POST" });
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

    if (valVectorHitRate) valVectorHitRate.textContent = `${vRate}%`;
    if (valVectorHits) valVectorHits.textContent = `Hits: ${report.vector_hits || 0} / ${report.total_queries || 0} queries`;
    if (barVectorFill) barVectorFill.style.width = `${vRate}%`;

    if (valHybridHitRate) valHybridHitRate.textContent = `${hRate}%`;
    if (valHybridHits) valHybridHits.textContent = `Hits: ${report.hybrid_hits || 0} / ${report.total_queries || 0} queries`;
    if (barHybridFill) barHybridFill.style.width = `${hRate}%`;

    if (valDeltaImprovement) valDeltaImprovement.textContent = `${delta >= 0 ? "+" : ""}${delta}%`;

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
  // Initial Data Fetching (Health & Metadata Filters)
  // ---------------------------------------------------------
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
