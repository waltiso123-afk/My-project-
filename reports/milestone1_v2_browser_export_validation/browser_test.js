/* global ort */
/* Real-browser onnxruntime-web test for SegFormer-B0 @768 binary export.
   Loads the actual .onnx, builds a 768x768x3 input, runs ONE forward pass,
   reports output shape/dtype + execution provider, and converts logits->binary mask. */
(function () {
  const logEl = document.getElementById("log");
  const statusEl = document.getElementById("status");
  const lines = [];
  function log(msg, cls) {
    lines.push(cls ? `[${cls.toUpperCase()}] ${msg}` : msg);
    logEl.textContent = lines.join("\n");
    console.log(msg);
  }
  function setStatus(msg, cls) {
    statusEl.textContent = msg;
    statusEl.className = cls || "";
  }

  const RES = 768;
  const MODEL_URL = "./segformer_b0_768_binary.onnx";

  async function tryProvider(ep) {
    // wasm files served from the same CDN as ort.min.js (version-matched)
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    ort.env.wasm.numThreads = 1; // no cross-origin isolation on dev server
    const opts = { executionProviders: [ep], graphOptimizationLevel: "all" };
    const session = await ort.InferenceSession.create(MODEL_URL, opts);
    return session;
  }

  async function main() {
    try {
      log("onnxruntime-web version: " + (ort.env.versions ? JSON.stringify(ort.env.versions) : ort.version || "unknown"));
      log("fetching + creating InferenceSession from " + MODEL_URL + " …");

      let session = null, usedEP = null;
      for (const ep of ["webgpu", "wasm"]) {
        try {
          const t0 = performance.now();
          session = await tryProvider(ep);
          usedEP = ep;
          log(`InferenceSession created with EP='${ep}' in ${(performance.now() - t0).toFixed(0)}ms`, "ok");
          break;
        } catch (e) {
          log(`EP '${ep}' unavailable: ${e.message || e}`, "warn");
        }
      }
      if (!session) throw new Error("could not create session with any EP (webgpu/wasm)");

      log("input names: " + JSON.stringify(session.inputNames));
      log("output names: " + JSON.stringify(session.outputNames));

      // one valid 768x768x3 input tensor (NCHW)
      const n = 1 * 3 * RES * RES;
      const data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = (Math.random() - 0.5); // arbitrary; model untrained
      const input = new ort.Tensor("float32", data, [1, 3, RES, RES]);
      log("built input tensor shape=[1,3,768,768] dtype=float32");

      log("running ONE forward pass …");
      const t1 = performance.now();
      const results = await session.run({ [session.inputNames[0]]: input });
      const dt = (performance.now() - t1).toFixed(0);
      const out = results[session.outputNames[0]];
      log(`forward pass OK in ${dt}ms`, "ok");
      log("OUTPUT name=" + session.outputNames[0] + " dtype=" + out.type + " shape=[" + out.dims.join(",") + "]", "ok");

      // logits -> binary roof/bg mask (argmax over channel dim=1) — downstream compatibility
      const [B, C, H, W] = out.dims;
      const od = out.data;
      let roof = 0;
      const plane = H * W;
      for (let p = 0; p < plane; p++) {
        const c0 = od[p];          // channel 0 (bg)
        const c1 = od[plane + p];  // channel 1 (roof)
        if (c1 > c0) roof++;
      }
      log(`argmax -> binary mask ${H}x${W}, roof_pixels=${roof} (values in {0,1})`, "ok");

      const summary = {
        browser_export: "PASS",
        execution_provider: usedEP,
        input_shape: [1, 3, RES, RES],
        input_dtype: "float32",
        output_name: session.outputNames[0],
        output_shape: out.dims,
        output_dtype: out.type,
        forward_ms: Number(dt),
        binary_mask: { h: H, w: W, roof_pixels: roof, values: "{0,1}" },
        onnxruntime_web: ort.env.versions || ort.version || "unknown",
      };
      window.__ONNX_RESULT__ = summary;
      log("\nRESULT_JSON " + JSON.stringify(summary));
      setStatus("BROWSER EXPORT: PASS — EP=" + usedEP + " · output [" + out.dims.join(",") + "] " + out.type, "ok");
      window.__ONNX_DONE__ = "PASS";
    } catch (e) {
      const msg = (e && (e.stack || e.message)) || String(e);
      log("FATAL: " + msg, "fail");
      window.__ONNX_RESULT__ = { browser_export: "FAIL", error: msg };
      setStatus("BROWSER EXPORT: FAIL — " + (e.message || e), "fail");
      window.__ONNX_DONE__ = "FAIL";
    }
  }
  main();
})();
