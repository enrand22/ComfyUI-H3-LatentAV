"""Save and load MiniMax H3 audio+video latents in ComfyUI.

Why these nodes exist
---------------------
The core `SaveLatent` node calls `samples["samples"].contiguous()`, but the MiniMax H3
sampler emits a `comfy.nested_tensor.NestedTensor` — a small container that holds the video
and audio latents as a list — and it has no `.contiguous()`:

    AttributeError: 'NestedTensor' object has no attribute 'contiguous'

The load side has the mirror problem: the H3 VAE wants that nested object itself (it
checks `.is_nested`), so a flat reconstruction of the parts is rejected with:

    'list' object has no attribute 'is_nested'

So `SaveLatentAV` stores the object as-is (`torch.save` serializes the NestedTensor fine;
what breaks in the core node is its `.contiguous()` call, not the serialization) and
`LoadLatentAV` hands it back untouched. No core ComfyUI file is modified.

These two nodes are the bridge between the halves of a two-pass H3 render, which is how
a 5-10 s clip fits on an 8 GB card:

    pass 1  (server started with --cpu-vae)  : sampling with the VAE on the CPU, then SaveLatentAV
    restart  (server without --cpu-vae)      : VAE on the GPU, nothing else resident
    pass 2  (server without --cpu-vae)       : LoadLatentAV + VAEDecodeTiled + VAEDecodeAudio

Keeping the two halves in separate server runs is not a style choice: `--cpu-vae` and a
GPU-resident VAE are mutually exclusive per run, and `VAELoader` exposes no `device`
input, so the sampler cannot have the VAE on the CPU while the decoder has it on the GPU
in the same process.
"""

import os
import time
import torch

import folder_paths

LATENT_DIRNAME = "latent_av"


def _latent_dir():
    d = os.path.join(folder_paths.get_output_directory(), LATENT_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def _plain(obj, depth=0):
    """Serialize without assuming the shape of the latent (legacy format reader)."""
    if depth > 6:
        raise TypeError("SaveLatentAV: structure too deep")
    if torch.is_tensor(obj):
        return {"k": "tensor", "v": obj.detach().cpu()}
    if isinstance(obj, dict):
        return {"k": "dict", "v": {str(a): _plain(b, depth + 1) for a, b in obj.items()}}
    if isinstance(obj, (list, tuple)):
        return {"k": "list", "v": [_plain(x, depth + 1) for x in obj]}
    for attr in ("unbind", "tensors"):
        if hasattr(obj, attr):
            try:
                parts = list(getattr(obj, attr)())
            except Exception:
                continue
            if parts:
                return {"k": "nested", "attr": attr, "v": [_plain(p, depth + 1) for p in parts],
                        "cls": type(obj).__name__}
    raise TypeError(f"SaveLatentAV: don't know how to serialize {type(obj)}")


def _unplain(d):
    kind = d["k"]
    if kind == "tensor":
        return d["v"]
    if kind == "dict":
        return {a: _unplain(b) for a, b in d["v"].items()}
    if kind == "list":
        return [_unplain(x) for x in d["v"]]
    if kind == "nested":
        parts = [_unplain(x) for x in d["v"]]
        try:
            from comfy.nested_tensor import NestedTensor
            return NestedTensor(parts)
        except Exception:
            # If it cannot be regrouped, return the sequence: the H3 VAE consumes it in
            # practice, but `.is_nested` checks will fail downstream.
            return parts if len(parts) > 1 else parts[0]
    raise TypeError("LoadLatentAV: unreadable payload")


def _list_files():
    import glob
    files = sorted(glob.glob(os.path.join(_latent_dir(), "*.pt")))
    names = [os.path.basename(f) for f in files if not f.endswith(".tmp")]
    return names or ["(empty)"]


class SaveLatentAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",),
                             "filename_prefix": ("STRING", {"default": "h3_latent"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "latent"

    def save(self, samples, filename_prefix="h3_latent"):
        payload = {
            "prefix": filename_prefix,
            "created": time.time(),
            "notes": "AV latent (NestedTensor, video+audio) written by h3_latent_av",
            "samples": samples,
        }
        name = f"{filename_prefix}_{time.strftime('%Y%m%d-%H%M%S')}.pt"
        path = os.path.join(_latent_dir(), name)
        torch.save(payload, path)
        # Stable alias so pass 2 can reference the latent without guessing the timestamp.
        # Do not rely on it when several jobs may share a prefix: point pass 2 at the exact
        # timestamped file instead, or ComfyUI's execution cache can serve the previous decode.
        stable = os.path.join(_latent_dir(), f"{filename_prefix}.pt")
        try:
            tmp = f"{stable}.{os.getpid()}.tmp"
            if os.path.lexists(tmp):
                os.remove(tmp)
            os.symlink(name, tmp)
            os.replace(tmp, stable)
        except OSError as exc:
            print(f"[h3_latent_av] could not create the alias {stable}: {exc}")
        size_mb = os.path.getsize(path) / 1048576
        print(f"[h3_latent_av] saved {path} ({size_mb:.1f} MB)")
        return {"ui": {"text": [f"{name} ({size_mb:.1f} MB)"]}}


class LoadLatentAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": (_list_files(),)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "latent"

    def load(self, latent):
        path = os.path.join(_latent_dir(), latent)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        if "samples" in payload:
            samples = payload["samples"]
        else:                      # legacy _plain payloads: only reconstructable if nested
            samples = _unplain(payload["data"])
        print(f"[h3_latent_av] loaded {path} (created {payload.get('created')}, "
              f"{type(samples).__name__})")
        return (samples,)


NODE_CLASS_MAPPINGS = {"SaveLatentAV": SaveLatentAV, "LoadLatentAV": LoadLatentAV}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLatentAV": "Save Latent AV (H3)",
    "LoadLatentAV": "Load Latent AV (H3)",
}
