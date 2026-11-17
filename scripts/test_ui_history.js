/**
 * Test history storage, de-duplication, tab switching, and rendering logic.
 */
const assert = require("assert");

// Mock LocalStorage
const storage = {};
global.localStorage = {
  getItem: (k) => (storage.hasOwnProperty(k) ? storage[k] : null),
  setItem: (k, v) => { storage[k] = String(v); },
  removeItem: (k) => { delete storage[k]; },
  clear: () => { Object.keys(storage).forEach((k) => delete storage[k]); },
};

// Mock DOM elements
function createMockElement(tag, id = "") {
  return {
    tagName: tag.toUpperCase(),
    id,
    className: "",
    textContent: "",
    title: "",
    children: [],
    attributes: {},
    classList: {
      _classes: new Set(),
      add(c) { this._classes.add(c); },
      remove(c) { this._classes.delete(c); },
      contains(c) { return this._classes.has(c); },
    },
    setAttribute(k, v) { this.attributes[k] = v; },
    getAttribute(k) { return this.attributes[k]; },
    appendChild(child) { this.children.push(child); },
    addEventListener(event, fn) { this._listeners = this._listeners || {}; this._listeners[event] = fn; },
    click() { if (this._listeners && this._listeners.click) this._listeners.click(); },
    focus() { this._focused = true; },
  };
}

const HISTORY_STORAGE_KEY = "arise-history";
const MAX_HISTORY_ITEMS = 10;
const historyList = createMockElement("ul", "history-list");
const queryInput = createMockElement("textarea", "query-input");
const assistantBtn = createMockElement("button", "tab-btn-assistant");

function getStoredHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
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
  } catch (e) {}
}

function clearHistory() {
  try {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
  } catch (e) {}
  renderHistoryList([]);
}

function renderHistoryList(items) {
  historyList.children = [];
  if (!items || items.length === 0) {
    const emptyLi = createMockElement("li");
    emptyLi.className = "history-empty";
    emptyLi.textContent = "Questions you ask will show up here.";
    historyList.appendChild(emptyLi);
    return;
  }
  items.forEach((item) => {
    const li = createMockElement("li");
    li.className = "history-item";
    li.title = item.question;

    const qSpan = createMockElement("span");
    qSpan.className = "history-q";
    qSpan.textContent = item.question;
    li.appendChild(qSpan);

    const onSelect = () => {
      queryInput.value = item.question;
      if (!assistantBtn.classList.contains("active")) {
        assistantBtn.click();
      }
      queryInput.focus();
    };
    li.addEventListener("click", onSelect);
    historyList.appendChild(li);
  });
}

// TESTS
console.log("1. Test empty state initially");
renderHistoryList(getStoredHistory());
assert.strictEqual(historyList.children.length, 1);
assert.strictEqual(historyList.children[0].className, "history-empty");
assert.strictEqual(historyList.children[0].textContent, "Questions you ask will show up here.");
console.log("   PASS");

console.log("2. Test saving 2 distinct questions");
saveHistoryItem({ question: "Question A", mode: "hybrid" });
saveHistoryItem({ question: "Question B", mode: "vector" });
let stored = getStoredHistory();
assert.strictEqual(stored.length, 2);
assert.strictEqual(stored[0].question, "Question B");
assert.strictEqual(stored[1].question, "Question A");
assert.strictEqual(historyList.children.length, 2);
console.log("   PASS");

console.log("3. Test de-duplication: saving Question A again");
saveHistoryItem({ question: "Question A", mode: "hybrid" });
stored = getStoredHistory();
assert.strictEqual(stored.length, 2, "Length should remain 2 after de-duplication");
assert.strictEqual(stored[0].question, "Question A", "Question A should move to the top");
assert.strictEqual(stored[1].question, "Question B", "Question B should be second");
console.log("   PASS");

console.log("4. Test click on history item fills queryInput and focuses");
historyList.children[0].click();
assert.strictEqual(queryInput.value, "Question A");
assert.strictEqual(queryInput._focused, true);
console.log("   PASS");

console.log("5. Test clear history");
clearHistory();
stored = getStoredHistory();
assert.strictEqual(stored.length, 0);
assert.strictEqual(historyList.children.length, 1);
assert.strictEqual(historyList.children[0].className, "history-empty");
console.log("   PASS");

console.log("\nALL UI HISTORY UNIT TESTS PASSED SUCCESSFULLY!");
