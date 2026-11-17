const http = require("http");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const PORT = 9222;
const HTML_PATH = "file:///" + path.resolve(__dirname, "../frontend/index.html").replace(/\\/g, "/");
const TEMP_USER_DATA = path.resolve(__dirname, "../.edge_perf_temp");

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function fetchJson(url, method = "GET") {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: parsedUrl.port || PORT,
        path: parsedUrl.pathname + parsedUrl.search,
        method: method,
        headers: {
          Host: `localhost:${parsedUrl.port || PORT}`,
        },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(new Error(`Failed to parse JSON (${data}): ${e.message}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.end();
  });
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.id = 1;
    this.callbacks = new Map();
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => resolve();
      this.ws.onerror = (err) => reject(err);
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.id && this.callbacks.has(msg.id)) {
          const cb = this.callbacks.get(msg.id);
          this.callbacks.delete(msg.id);
          if (msg.error) cb.reject(new Error(msg.error.message));
          else cb.resolve(msg.result);
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.id++;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const res = await this.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    return res && res.result ? res.result.value : null;
  }

  async captureScreenshot(outputPath, width, height) {
    if (width && height) {
      await this.send("Emulation.setDeviceMetricsOverride", {
        width,
        height,
        deviceScaleFactor: 1,
        mobile: width < 600,
      });
    }
    const res = await this.send("Page.captureScreenshot", { format: "png" });
    const buffer = Buffer.from(res.data, "base64");
    fs.writeFileSync(outputPath, buffer);
  }

  close() {
    if (this.ws) this.ws.close();
  }
}

async function runMeasurement(tag = "baseline") {
  console.log(`\n========================================`);
  console.log(`Starting Measurement: ${tag}`);
  console.log(`========================================\n`);

  const runUserData = path.resolve(__dirname, `../.edge_perf_temp_${tag}_${Date.now()}`);
  if (!fs.existsSync(runUserData)) {
    fs.mkdirSync(runUserData, { recursive: true });
  }

  const edgeProcess = spawn(EDGE_PATH, [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${runUserData}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-gpu",
    "about:blank",
  ]);

  edgeProcess.stderr.on("data", (d) => console.log("EDGE STDERR:", d.toString()));
  edgeProcess.stdout.on("data", (d) => console.log("EDGE STDOUT:", d.toString()));
  edgeProcess.on("exit", (code) => console.log("EDGE EXITED with code:", code));

  let versionInfo = null;
  for (let i = 0; i < 30; i++) {
    try {
      versionInfo = await fetchJson(`http://127.0.0.1:${PORT}/json/version`);
      if (versionInfo) break;
    } catch (e) {
      if (i % 5 === 0) console.log(`DevTools poll (${i}/30): ${e.message}`);
      await delay(300);
    }
  }

  if (!versionInfo) {
    edgeProcess.kill();
    throw new Error("Could not connect to Edge DevTools");
  }

  // Create target page via PUT
  const newTab = await fetchJson(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(HTML_PATH)}`, "PUT");
  const cdp = new CDPClient(newTab.webSocketDebuggerUrl);
  await cdp.connect();

  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 1366,
    height: 768,
    deviceScaleFactor: 1,
    mobile: false,
  });

  // Wait for initial load
  await delay(1500);

  // 1. Measure Core Web Vitals (LCP, CLS, DOMContentLoaded, Load)
  const perfMetrics = await cdp.evaluate(`
    new Promise((resolve) => {
      let lcp = 0;
      let cls = 0;
      let longTasks = [];

      try {
        const poLcp = new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          if (entries.length) lcp = entries[entries.length - 1].startTime;
        });
        poLcp.observe({ type: 'largest-contentful-paint', buffered: true });
      } catch (e) {}

      try {
        const poCls = new PerformanceObserver((entryList) => {
          for (const entry of entryList.getEntries()) {
            if (!entry.hadRecentInput) cls += entry.value;
          }
        });
        poCls.observe({ type: 'layout-shift', buffered: true });
      } catch (e) {}

      try {
        const poLt = new PerformanceObserver((entryList) => {
          for (const entry of entryList.getEntries()) {
            longTasks.push({ duration: entry.duration, startTime: entry.startTime });
          }
        });
        poLt.observe({ type: 'longtask', buffered: true });
      } catch (e) {}

      setTimeout(() => {
        const nav = performance.getEntriesByType('navigation')[0] || {};
        resolve({
          domContentLoaded: nav.domContentLoadedEventEnd || 0,
          loadEvent: nav.loadEventEnd || 0,
          lcp: Math.round(lcp || (nav.domInteractive || 0)),
          cls: Number(cls.toFixed(4)),
          longTaskCount: longTasks.length,
          totalLongTaskDuration: Math.round(longTasks.reduce((s, t) => s + t.duration, 0)),
        });
      }, 500);
    })
  `);

  // 2. Measure Tab Switching Latency (Home -> Benchmark -> Home)
  const tabSwitchMetrics = await cdp.evaluate(`
    new Promise(async (resolve) => {
      const benchBtn = document.getElementById('tab-btn-benchmark') || document.querySelector('[data-tab="benchmark"]');
      const assistBtn = document.getElementById('tab-btn-assistant') || document.querySelector('[data-tab="assistant"]');

      if (!benchBtn || !assistBtn) {
        resolve({ switchTimeToBenchmarkMs: 0, switchTimeToAssistantMs: 0 });
        return;
      }

      // Switch to Benchmark
      const t0 = performance.now();
      benchBtn.click();
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      const tBench = performance.now() - t0;

      // Switch back to Assistant
      const t1 = performance.now();
      assistBtn.click();
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      const tAssist = performance.now() - t1;

      resolve({
        switchTimeToBenchmarkMs: Number(tBench.toFixed(2)),
        switchTimeToAssistantMs: Number(tAssist.toFixed(2)),
      });
    })
  `);

  // 3. Measure 5s Scripted Scroll Frame Rate and Dropped Frames
  const scrollMetrics = await cdp.evaluate(`
    new Promise((resolve) => {
      const container = document.querySelector('.main-container') || window;
      const scrollTarget = container === window ? document.documentElement : container;
      const startTime = performance.now();
      const frameDeltas = [];
      let lastFrame = performance.now();

      function scrollLoop(now) {
        const delta = now - lastFrame;
        frameDeltas.push(delta);
        lastFrame = now;

        const progress = (now - startTime) / 5000;
        if (progress < 1) {
          const maxScroll = Math.max(100, scrollTarget.scrollHeight - scrollTarget.clientHeight);
          const scrollPos = (Math.sin(progress * Math.PI * 4) * 0.5 + 0.5) * maxScroll;
          if (container === window) window.scrollTo(0, scrollPos);
          else container.scrollTop = scrollPos;
          requestAnimationFrame(scrollLoop);
        } else {
          const totalFrames = frameDeltas.length;
          const droppedFrames = frameDeltas.filter(d => d > 16.7).length; // > 60fps frame budget
          const maxFrameDelta = Math.max(...frameDeltas);
          const avgFrameRate = totalFrames / 5;
          resolve({
            totalFrames,
            droppedFrames,
            dropRatePct: Number(((droppedFrames / totalFrames) * 100).toFixed(1)),
            avgFrameRate: Number(avgFrameRate.toFixed(1)),
            maxFrameDeltaMs: Number(maxFrameDelta.toFixed(1)),
          });
        }
      }
      requestAnimationFrame(scrollLoop);
    })
  `);

  // Screenshots output directory
  const shotsDir = path.resolve(__dirname, `../screenshots_${tag}`);
  if (!fs.existsSync(shotsDir)) fs.mkdirSync(shotsDir, { recursive: true });

  // Capture Screenshots:
  // Desktop Light
  await cdp.evaluate(`document.documentElement.setAttribute('data-theme', 'light')`);
  await delay(100);
  await cdp.captureScreenshot(path.join(shotsDir, "desktop_light_1366x768.png"), 1366, 768);

  // Desktop Dark
  await cdp.evaluate(`document.documentElement.setAttribute('data-theme', 'dark')`);
  await delay(100);
  await cdp.captureScreenshot(path.join(shotsDir, "desktop_dark_1366x768.png"), 1366, 768);

  // Mobile Light
  await cdp.evaluate(`document.documentElement.setAttribute('data-theme', 'light')`);
  await delay(100);
  await cdp.captureScreenshot(path.join(shotsDir, "mobile_light_390x844.png"), 390, 844);

  // Mobile Dark
  await cdp.evaluate(`document.documentElement.setAttribute('data-theme', 'dark')`);
  await delay(100);
  await cdp.captureScreenshot(path.join(shotsDir, "mobile_dark_390x844.png"), 390, 844);

  try {
    await cdp.send("Browser.close");
  } catch (e) {}
  cdp.close();
  try {
    edgeProcess.kill();
  } catch (e) {}

  const report = {
    tag,
    timestamp: new Date().toISOString(),
    metrics: {
      lcpMs: perfMetrics.lcp,
      cls: perfMetrics.cls,
      domContentLoadedMs: Math.round(perfMetrics.domContentLoaded),
      longTaskCount: perfMetrics.longTaskCount,
      totalLongTaskDurationMs: perfMetrics.totalLongTaskDuration,
      tabSwitchToBenchmarkMs: tabSwitchMetrics.switchTimeToBenchmarkMs,
      tabSwitchToAssistantMs: tabSwitchMetrics.switchTimeToAssistantMs,
      scrollTest: scrollMetrics,
    },
    screenshots: [
      path.join(shotsDir, "desktop_light_1366x768.png"),
      path.join(shotsDir, "desktop_dark_1366x768.png"),
      path.join(shotsDir, "mobile_light_390x844.png"),
      path.join(shotsDir, "mobile_dark_390x844.png"),
    ],
  };

  const reportFile = path.resolve(__dirname, `../perf_report_${tag}.json`);
  fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));

  console.log(`\n--- Results for [${tag}] ---`);
  console.log(`LCP:                      ${report.metrics.lcpMs} ms`);
  console.log(`CLS:                      ${report.metrics.cls}`);
  console.log(`DOM Content Loaded:       ${report.metrics.domContentLoadedMs} ms`);
  console.log(`Long Tasks:               ${report.metrics.longTaskCount} (total: ${report.metrics.totalLongTaskDurationMs} ms)`);
  console.log(`Tab Switch -> Benchmark:  ${report.metrics.tabSwitchToBenchmarkMs} ms`);
  console.log(`Tab Switch -> Assistant:  ${report.metrics.tabSwitchToAssistantMs} ms`);
  console.log(`5s Scroll Avg FPS:        ${report.metrics.scrollTest.avgFrameRate} fps`);
  console.log(`5s Scroll Dropped Frames: ${report.metrics.scrollTest.droppedFrames} / ${report.metrics.scrollTest.totalFrames} (${report.metrics.scrollTest.dropRatePct}%)`);
  console.log(`5s Scroll Max Frame:      ${report.metrics.scrollTest.maxFrameDeltaMs} ms`);
  console.log(`Screenshots saved to:     ${shotsDir}`);
  console.log(`Report JSON saved to:     ${reportFile}\n`);

  return report;
}

if (require.main === module) {
  const tag = process.argv[2] || "baseline";
  runMeasurement(tag).catch((err) => {
    console.error("Measurement error:", err);
    process.exit(1);
  });
}

module.exports = { runMeasurement };
