"""ONNX export using the EXACT validated configuration: static [1,3,768,768] ->
[1,2,768,768], opset 17, bilinear upsample INSIDE the graph. Works for SegFormer-B0 and B1.
Self-contained (no Mongo/backend deps) so it runs on RunPod."""
import argparse
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

RES = 768

class SegBinaryONNX(nn.Module):
    """SegFormer -> logits(H/4) -> bilinear upsample to 768 -> logits[1,2,768,768]."""
    def __init__(self, hf_name):
        super().__init__()
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            hf_name, num_labels=2, ignore_mismatched_sizes=True)
    def forward(self, pixel_values):
        logits = self.net(pixel_values=pixel_values).logits
        return F.interpolate(logits, size=(RES, RES), mode="bilinear", align_corners=False)

def export_onnx(hf_name, state_dict_path, out_path, opset=17):
    wrapper = SegBinaryONNX(hf_name).eval()
    sd = torch.load(state_dict_path, map_location="cpu")
    wrapper.net.load_state_dict(sd)
    dummy = torch.randn(1, 3, RES, RES, dtype=torch.float32)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        opset_version=opset,
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes=None,               # STATIC 1x3x768x768 in / 1x2x768x768 out
        do_constant_folding=True,
    )
    # validate
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))
    return str(out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-name", required=True)     # nvidia/mit-b0 | nvidia/mit-b1
    ap.add_argument("--weights", required=True)      # .pt state_dict of the trained Segformer
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    p = export_onnx(a.hf_name, a.weights, a.out, a.opset)
    print("ONNX exported + checked:", p)
