import sys
from pathlib import Path
from typing import Optional

DISTILDIRE_ROOT = Path("/home/divc_col2/Group_3/distilDIRE/DistilDIRE-master")

if str(DISTILDIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(DISTILDIRE_ROOT))

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from guided_diffusion.compute_dire_eps import ( # type: ignore
    dire_get_first_step_noise,
    create_argparser,
) 
from guided_diffusion.guided_diffusion.script_util import ( # type: ignore
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    dict_parse,
)
from networks.distill_model import DistilDIRE # type: ignore


DEFAULT_CKPT_PATH = DISTILDIRE_ROOT / "weights" / "260326final2.pth"
DEFAULT_ADM_PATH = DISTILDIRE_ROOT / "models" / "256x256-adm.pt"
DEFAULT_OUTPUT_DIR = DISTILDIRE_ROOT / "output"


_MODEL = None
_ADM_MODEL = None
_DIFFUSION = None
_ADM_ARGS = None
_DEVICE = None
_LOADED_CKPT = None
_LOADED_ADM = None


def _load_distildire(ckpt_path: Path, device: str) -> DistilDIRE:
    model = DistilDIRE(device).to(device)

    raw = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _load_adm(adm_path: Path, device: str):
    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        args = create_argparser()
    finally:
        sys.argv = old_argv

    args["timestep_respacing"] = "ddim20"
    args["model_path"] = str(adm_path)

    adm_model, diffusion = create_model_and_diffusion(
        **dict_parse(args, model_and_diffusion_defaults().keys())
    )

    adm_model.load_state_dict(torch.load(args["model_path"], map_location="cpu"))
    adm_model.to(device)
    adm_model.eval()

    return adm_model, diffusion, args


def _ensure_models_loaded(ckpt_path: Path, adm_path: Path, device: str):
    global _MODEL, _ADM_MODEL, _DIFFUSION, _ADM_ARGS, _DEVICE, _LOADED_CKPT, _LOADED_ADM

    need_reload = (
        _MODEL is None
        or _ADM_MODEL is None
        or _DIFFUSION is None
        or _ADM_ARGS is None
        or _DEVICE != device
        or _LOADED_CKPT != str(ckpt_path)
        or _LOADED_ADM != str(adm_path)
    )

    if need_reload:
        _MODEL = _load_distildire(ckpt_path, device)
        _ADM_MODEL, _DIFFUSION, _ADM_ARGS = _load_adm(adm_path, device)
        _DEVICE = device
        _LOADED_CKPT = str(ckpt_path)
        _LOADED_ADM = str(adm_path)


def _preprocess_for_model(img_path: Path, device: str):
    trans = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop((256, 256)),
    ])

    pil = Image.open(img_path).convert("RGB")

    model_img = TF.to_tensor(pil) * 2 - 1
    model_img = trans(model_img).to(device).unsqueeze(0)

    vis_img = TF.to_tensor(pil)
    vis_img = trans(vis_img)
    vis_img = vis_img.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    vis_img = np.clip(vis_img, 0.0, 1.0)

    return model_img, vis_img


class _DistilDIRECamWrapper(nn.Module):
    def __init__(self, model: DistilDIRE, eps: torch.Tensor):
        super().__init__()
        self.model = model
        self.eps = eps

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.model(img, self.eps)
        return out["logit"].view(-1)


class _FakeLogitTarget:
    def __call__(self, model_output: torch.Tensor):
        if model_output.ndim == 0:
            return model_output
        return model_output[0]


def grad_cam(
    input_path: str,
    output_path: Optional[str] = None,
    ckpt_path: str = str(DEFAULT_CKPT_PATH),
    adm_path: str = str(DEFAULT_ADM_PATH),
    device: Optional[str] = None,
) -> str:
    input_path = Path(input_path)
    ckpt_path = Path(ckpt_path)
    adm_path = Path(adm_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到輸入圖片：{input_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 DistilDIRE 權重：{ckpt_path}")

    if not adm_path.exists():
        raise FileNotFoundError(f"找不到 ADM 權重：{adm_path}")

    _ensure_models_loaded(
        ckpt_path=ckpt_path,
        adm_path=adm_path,
        device=device,
    )

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_cam.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    img, vis_img = _preprocess_for_model(input_path, device)

    with torch.no_grad():
        eps = dire_get_first_step_noise(img, _ADM_MODEL, _DIFFUSION, _ADM_ARGS, device)

    cam_model = _DistilDIRECamWrapper(_MODEL, eps).to(device).eval()
    target_layers = [cam_model.model.student_backbone[-1][-1]]
    targets = [_FakeLogitTarget()]

    with GradCAM(model=cam_model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img, targets=targets)[0]

    vis_img = np.clip(vis_img.astype(np.float32), 0.0, 1.0)
    cam_image = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)

    Image.fromarray(cam_image).save(output_path)
    return str(output_path)