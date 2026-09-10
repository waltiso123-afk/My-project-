"""Export UNTRAINED SegFormer-B0 (binary roof/bg, 768x768) to ONNX for onnxruntime-web.
No training. No GPU. Records full metadata + runs onnx.checker + op inventory + a Python
onnxruntime parity forward (browser test is separate, in a real browser)."""
import os, sys, json, time, subprocess
import torch, torch.nn as nn, torch.nn.functional as F
import transformers, onnx, onnxruntime as ort
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
import numpy as np

OUT = Path("/app/reports/milestone1_v2_browser_export_validation")
OUT.mkdir(parents=True, exist_ok=True)
ONNX_PATH = OUT / "segformer_b0_768_binary.onnx"
LOG = open(OUT / "export.log", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()

RES = 768

class SegB0Binary(nn.Module):
    """Production wrapper: SegFormer-B0 -> logits (H/4) -> bilinear upsample to 768 -> logits[1,2,768,768].
    Downstream takes argmax(dim=1) -> binary roof/bg mask."""
    def __init__(self):
        super().__init__()
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True)
    def forward(self, pixel_values):
        logits = self.net(pixel_values=pixel_values).logits
        return F.interpolate(logits, size=(RES, RES), mode="bilinear", align_corners=False)

def main():
    meta = {}
    log("=== SegFormer-B0 -> ONNX export ===", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    meta["python"] = sys.version.split()[0]
    meta["torch"] = torch.__version__
    meta["transformers"] = transformers.__version__
    meta["onnx"] = onnx.__version__
    meta["onnxruntime_python"] = ort.__version__
    log("versions:", json.dumps({k: meta[k] for k in ["python","torch","transformers","onnx","onnxruntime_python"]}))

    model = SegB0Binary().eval()
    n = sum(p.numel() for p in model.parameters())
    log(f"SegFormer-B0 (nvidia/mit-b0) num_labels=2, params={n/1e6:.2f}M — UNTRAINED (random head)")

    dummy = torch.randn(1, 3, RES, RES, dtype=torch.float32)
    OPSET = 17
    meta["opset"] = OPSET
    meta["input_name"] = "pixel_values"
    meta["output_name"] = "logits"
    export_cmd = (
        "torch.onnx.export(model, dummy, ONNX_PATH, opset_version=17, "
        "input_names=['pixel_values'], output_names=['logits'], "
        "dynamic_axes=None  # STATIC 1x3x768x768 for browser)")
    meta["export_command"] = export_cmd
    log("export command:", export_cmd)

    t0 = time.time()
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        opset_version=OPSET,
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes=None,  # static shapes -> simplest/most compatible for onnxruntime-web
        do_constant_folding=True,
    )
    log(f"exported in {time.time()-t0:.1f}s -> {ONNX_PATH}")
    meta["onnx_file_size_bytes"] = ONNX_PATH.stat().st_size
    meta["onnx_file_size_mb"] = round(ONNX_PATH.stat().st_size / 1e6, 2)
    log("onnx file size:", meta["onnx_file_size_mb"], "MB")

    # ---- onnx.checker validation
    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)
    log("onnx.checker.check_model: OK (valid graph)")

    # ---- input/output tensor specs
    def spec(vi):
        t = vi.type.tensor_type
        dt = onnx.TensorProto.DataType.Name(t.elem_type)
        shape = [(d.dim_value if d.HasField("dim_value") else d.dim_param) for d in t.shape.dim]
        return {"name": vi.name, "dtype": dt, "shape": shape}
    meta["input"] = spec(m.graph.input[0])
    meta["output"] = spec(m.graph.output[0])
    log("INPUT :", json.dumps(meta["input"]))
    log("OUTPUT:", json.dumps(meta["output"]))
    meta["dynamic_axes"] = "none (static 1x3x768x768 in / 1x2x768x768 out)"

    # ---- operator inventory
    ops = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    meta["operators"] = ops
    log("operator inventory:", json.dumps(ops, indent=2))

    # onnxruntime-web WASM EP supported-ops (common set). Flag anything unusual.
    WASM_SUPPORTED = {
        "Add","Sub","Mul","Div","MatMul","Gemm","Conv","Relu","Gelu","Erf","Tanh","Sigmoid",
        "Softmax","Transpose","Reshape","Concat","Slice","Gather","Unsqueeze","Squeeze","Shape",
        "Cast","Constant","ConstantOfShape","Resize","Pad","ReduceMean","Sqrt","Pow","Sub",
        "Where","Expand","Range","Split","Flatten","Identity","Equal","Not","ArgMax","BatchNormalization",
        "InstanceNormalization","LayerNormalization","ReduceSum","Exp","Log","Min","Max","Clip",
        "Neg","Reciprocal","GlobalAveragePool","AveragePool","MaxPool","Dropout","ReduceMax",
    }
    unknown = sorted(set(ops) - WASM_SUPPORTED)
    meta["operators_not_in_known_wasm_set"] = unknown
    log("operators NOT in known-wasm-supported set:", unknown if unknown else "NONE")

    # ---- Python onnxruntime parity forward (sanity; browser test is authoritative)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inp = np.random.randn(1, 3, RES, RES).astype(np.float32)
    out = sess.run(None, {"pixel_values": inp})[0]
    meta["py_ort_output_shape"] = list(out.shape)
    meta["py_ort_output_dtype"] = str(out.dtype)
    log("python onnxruntime forward OK -> output", out.shape, out.dtype)
    # output -> binary mask parity
    binm = out.argmax(axis=1)[0]  # [768,768] in {0,1}
    log("argmax -> binary mask shape", binm.shape, "unique", np.unique(binm).tolist(),
        "roof_px", int((binm == 1).sum()))
    meta["binary_mask_from_logits"] = {"shape": list(binm.shape), "values": np.unique(binm).tolist()}

    (OUT / "export_meta.json").write_text(json.dumps(meta, indent=2))
    log("wrote export_meta.json")
    LOG.close()
    return meta

if __name__ == "__main__":
    main()
