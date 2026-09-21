"""Round-trip test: save a MiniMax H3-style AV latent and load it back.

Run it with the same Python that runs ComfyUI, with ComfyUI on the path:

    COMFYUI_PATH=/path/to/ComfyUI python tests/test_roundtrip.py

By default it assumes this repo lives in `ComfyUI/custom_nodes/`, i.e. ComfyUI is two
directories up. It touches nothing but a temp directory.
"""
import importlib.util
import os
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFYUI_PATH") or os.path.abspath(os.path.join(REPO, "..", ".."))


def _load_nested_tensor_class(comfy_root):
    """Import comfy.nested_tensor without pulling the whole comfy package in."""
    pkg = types.ModuleType("comfy")
    pkg.__path__ = [os.path.join(comfy_root, "comfy")]
    sys.modules["comfy"] = pkg
    path = os.path.join(comfy_root, "comfy", "nested_tensor.py")
    spec = importlib.util.spec_from_file_location("comfy.nested_tensor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["comfy.nested_tensor"] = module
    spec.loader.exec_module(module)
    return module.NestedTensor


def main():
    if not os.path.isfile(os.path.join(COMFY, "comfy", "nested_tensor.py")):
        print(f"ComfyUI not found at {COMFY}: set COMFYUI_PATH")
        return 2

    tmp = tempfile.mkdtemp(prefix="latent_av_test_")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: tmp
    sys.modules["folder_paths"] = folder_paths

    NestedTensor = _load_nested_tensor_class(COMFY)

    import torch

    sys.path.insert(0, REPO)
    node = importlib.import_module("__init__")

    video = torch.randn(1, 24, 42, 36, 54)
    audio = torch.randn(1, 32, 2, 235)
    latent = NestedTensor([video, audio])
    print(f"input: {type(latent).__name__}, is_nested={latent.is_nested}")

    # this is the core SaveLatent call that fails on H3 latents
    try:
        latent.contiguous()
        print("core SaveLatent should NOT be able to do this")
        return 1
    except AttributeError as exc:
        print(f"core SaveLatent would fail with: {exc}")

    print("save ->", node.SaveLatentAV().save({"samples": latent}, "test_latent")["ui"]["text"])
    print("list ->", node.LoadLatentAV.INPUT_TYPES()["required"]["latent"][0])

    loaded = node.LoadLatentAV().load("test_latent.pt")[0]
    samples = loaded["samples"]           # the node input/output is the LATENT dict
    print(f"loaded: LATENT -> {type(samples).__name__}, is_nested={samples.is_nested}")

    assert samples.is_nested, "the H3 VAE requires is_nested"
    assert type(samples) is type(latent)
    for got, want in zip(samples.tensors, (video, audio)):
        assert got.shape == want.shape and torch.equal(got, want), "tensors differ"
    print("round trip identical:", [tuple(t.shape) for t in samples.tensors])
    print("temp dir:", tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
