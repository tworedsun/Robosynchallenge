"""OpenAI 兼容的 VLM 客户端（仅 HTTP，无任何 CLI 调用）。

本模块从 ``_stage/harness/handle_basket_clickbell_20260924/milk_vision.py`` 移植，
但**彻底删除**了外部命令行（CLI）视觉分支：只保留 OpenAI 兼容的 HTTP 端点。

设计要点
--------
* 全部配置来自 :func:`robosyn_hybrid.settings.get_settings`（``vlm_url`` /
  ``vlm_model`` / ``vlm_timeout_s`` / ``vlm_enabled``）。
* 端点未配置（``vlm_url == ""`` 或 ``vlm_enabled is False``）时，所有公开函数
  **立即返回** ``{"ok": False, "error": "vlm_disabled", ...}``：不发网络请求、
  不阻塞、不抛异常。
* 网络异常 / 超时 / JSON 解析失败一律返回 ``{"ok": False, "mode": "http",
  "error": "...", ...}``，**绝不抛异常**——下游状态机依赖返回值做降级。
* prompt 文本与原始 ``milk_vision.py`` 逐字一致（prompt 是实验的一部分）。

返回字段约定
------------
* :func:`vlm_chat`：``{"ok", "mode": "http", "text", "error", "latency_s"}``
* :func:`locate_milk`：``ok`` 表示**定位可用**（HTTP 成功且 JSON 可解析且可见），
  另含 ``visible / uv / bbox_xyxy / yaw_deg / p1_uv / p2_uv / confidence / note``。
* :func:`judge_place_release`：``place_ready / place_relation /
  milk_inside_basket / over_basket_opening / gripper_clear / confidence /
  reason``。
* :func:`judge_handle_grasp`：``handle_between_fingers / handle_visible /
  confidence / reason``。
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:  # 作为包导入（正常路径）
    from .settings import Settings, get_settings
except ImportError:  # 直接以脚本/扁平目录方式导入时的兜底
    from settings import Settings, get_settings  # type: ignore[no-redef]

__all__ = [
    "vlm_chat",
    "vlm_available",
    "is_vlm_available",
    "locate_milk",
    "judge_place_release",
    "judge_basket_release",
    "judge_handle_grasp",
]

# 生成式端点返回的最大 token 数（与原实现一致）。
_MAX_TOKENS = 256
# 数据 URL 前缀。
_DATA_URL_PREFIX = "data:image/png;base64,"


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _uint8(image: Any) -> np.ndarray:
    """把任意图像类对象转成 uint8 ndarray（与原实现一致）。"""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _json_from_text(text: str) -> dict | None:
    """从模型回复里容错地抠出 JSON 对象（容忍 markdown 围栏）。

    与原始实现逐字一致：去掉 ````` ```json ```` 围栏后取第一个 ``{`` 到最后一个
    ``}`` 之间的内容。
    """
    value = str(text or "").strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:].lstrip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(value[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _error_text(exc: BaseException, timeout_s: int) -> str:
    """把异常整理成人类可读的 ``error`` 字符串（区分超时）。"""
    name = type(exc).__name__
    if "Timeout" in name or "timed out" in str(exc).lower():
        return f"timeout after {timeout_s}s ({name}: {exc})"
    return f"{name}: {exc}"


def _resolve_config() -> tuple[Settings | None, str | None]:
    """读取 settings；返回 ``(settings, 错误串)``，绝不抛异常。"""
    try:
        return get_settings(), None
    except Exception as exc:  # pragma: no cover - 仅在环境异常时触发
        return None, f"settings_unavailable: {type(exc).__name__}: {exc}"


def _base_info(settings: Settings | None = None) -> dict:
    """构造统一的返回字典骨架（保证字段齐全，下游可直接 ``.get``）。"""
    info: dict = {
        "mode": "http",
        "ok": False,
        "text": "",
        "error": None,
        "latency_s": 0.0,
        "http_ok": False,
    }
    if settings is not None:
        info["url"] = settings.vlm_url
        info["model"] = settings.vlm_model
    return info


def _disabled_info(settings: Settings | None) -> dict:
    """端点未配置时的标准返回：不发请求、不阻塞、不抛异常。"""
    info = _base_info(settings)
    info["error"] = "vlm_disabled"
    if settings is not None:
        info["vlm_enabled"] = bool(settings.vlm_enabled)
    return info


def _import_requests() -> tuple[Any | None, str | None]:
    """延迟导入 ``requests``；缺失时返回可读错误而不是抛异常。"""
    try:
        import requests  # noqa: PLC0415 - 延迟导入，缺失时走降级
    except ImportError:
        return None, "requests_missing"
    return requests, None


def _png_bytes(image: Any) -> tuple[bytes | None, str | None]:
    """把图像编码为 PNG 字节；``PIL`` 缺失时返回错误串而不是抛异常。"""
    try:
        from PIL import Image  # noqa: PLC0415 - 延迟导入，缺失时走降级
    except ImportError:
        return None, "pillow_missing"
    try:
        buffer = io.BytesIO()
        Image.fromarray(_uint8(image)).save(buffer, format="PNG")
        return buffer.getvalue(), None
    except Exception as exc:
        return None, f"image_encode_failed: {type(exc).__name__}: {exc}"


def _data_url(png: bytes) -> str:
    """把 PNG 字节封装成 base64 data URL。"""
    return _DATA_URL_PREFIX + base64.b64encode(png).decode("ascii")


def _compose_panels(high_image: Any, wrist_image: Any) -> tuple[np.ndarray | None, str | None]:
    """按原始实现横向拼接两张图：裁到相同高度后沿 axis=1 拼接。

    返回 ``(panel, 错误串)``。
    """
    try:
        high = _uint8(high_image)
        wrist = _uint8(wrist_image)
        height = min(high.shape[0], wrist.shape[0])
        panel = np.concatenate([high[:height], wrist[:height]], axis=1)
    except Exception as exc:
        return None, f"image_compose_failed: {type(exc).__name__}: {exc}"
    return panel, None


def _dump_debug(output_dir: Any, filename: str, panel: np.ndarray, text: str | None) -> None:
    """把送入 VLM 的图（及原始回复）落盘，便于事后排查；失败时静默忽略。

    ``output_dir is None`` 时不写任何文件（默认不产生临时文件）。
    """
    if output_dir is None:
        return
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        png, error = _png_bytes(panel)
        if png is not None:
            (out / filename).write_bytes(png)
        if text:
            (out / "vlm_text.txt").write_text(str(text), encoding="utf-8")
    except Exception:
        return


# ---------------------------------------------------------------------------
# Prompt（与原始 milk_vision.py 逐字一致，不要改写）
# ---------------------------------------------------------------------------


def _prompt(width: int, height: int) -> str:
    return (
        "Return strict JSON only, with no markdown. Locate the milk carton "
        "visible in this wrist RGB frame. Its colour/texture may be randomized. "
        "Report the visible surface centre in pixel coordinates and an axis-aligned "
        "bbox. Also report p1_uv and p2_uv as two points on opposite horizontal "
        "ends of the milk carton's visible front face; they are used to estimate "
        "the carton yaw around the vertical axis. "
        "If not visible set visible=false. Schema: "
        '{"visible":true|false,"u":number|null,"v":number|null,'
        '"bbox_xyxy":[left,top,right,bottom]|null,"yaw_deg":number|null,'
        '"p1_uv":[u,v]|null,"p2_uv":[u,v]|null,'
        '"confidence":0..1,"note":"short reason"}.'
        f" Image size is {width}x{height}."
    )


def _place_release_prompt(width: int, height: int) -> str:
    return (
        "Return strict JSON only, with no markdown. The image is a side-by-side "
        "view of the robot workspace: left panel is cam_high, right panel is the "
        "left wrist camera. Decide whether the left gripper can release the milk "
        "carton into the basket now. Be conservative: release is allowed only if "
        "the milk is visibly inside/over the basket opening and dropping it will "
        "not put it outside the basket. Schema: "
        '{"place_ready":true|false,"place_relation":"inside|edge|outside|unknown",'
        '"milk_inside_basket":true|false,'
        '"over_basket_opening":true|false,"gripper_clear":true|false,'
        '"confidence":0..1,"reason":"short reason"}. '
        f"Image size is {width}x{height}."
    )


def _basket_release_prompt(width: int, height: int) -> str:
    return (
        "Return strict JSON only, with no markdown. The image is a side-by-side "
        "view of the robot workspace: left panel is cam_high, right panel is the "
        "right wrist camera. Decide whether the basket has reached a place where the "
        "right gripper can release it. It is OK if the fingers are still inside "
        "the basket; do not reject solely for that reason. The key question is "
        "whether the basket is supported by the table/target and will not fall or "
        "tip when released. Schema: "
        '{"release_ready":true|false,"basket_supported":true|false,'
        '"basket_stable":true|false,"gripper_clear":true|false,'
        '"confidence":0..1,"reason":"short reason"}. '
        f"Image size is {width}x{height}."
    )


def _handle_grasp_prompt(width: int, height: int) -> str:
    return (
        "Return strict JSON only. The image is a side-by-side view: left panel is "
        "cam_high, right panel is cam_right_wrist. Decide whether the basket "
        "handle/grasp part is visibly positioned between the two right-gripper "
        "fingers, with the fingers open. In random mode the basket/handle color "
        "and texture are randomized, so do not use color as the main cue. The "
        "basket handle is the elongated vertical strap/bar/loop attached to or "
        "crossing the basket opening. Treat that vertical elongated shape as the "
        "handle even if its color differs from the basket. Set "
        "handle_between_fingers=true only when the two open right-gripper fingers "
        "straddle that vertical strap/bar. Schema: "
        '{"handle_between_fingers":true|false,"handle_visible":true|false,'
        '"confidence":0..1,"reason":"short reason"}. '
        f"Image size is {width}x{height}."
    )


def _finish(info: dict, parsed: dict | None, width: int, height: int) -> dict:
    """把牛奶定位的 JSON 结果裁剪/校验进 ``info``（与原始实现逐字一致的逻辑）。"""
    if parsed is None:
        info["error"] = "VLM returned no parseable JSON"
        return info
    info["answer"] = parsed
    if not bool(parsed.get("visible", False)):
        info["visible"] = False
        info["error"] = "VLM reported milk not visible"
        return info
    try:
        u, v = float(parsed["u"]), float(parsed["v"])
    except Exception:
        info["error"] = "VLM returned no usable centre"
        return info
    if not (0 <= u < width and 0 <= v < height):
        info["error"] = "VLM centre outside frame"
        return info
    bbox = parsed.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        r = 18
        bbox = [u - r, v - r, u + r, v + r]
    try:
        bbox = [
            max(0.0, min(width - 1.0, float(bbox[0]))),
            max(0.0, min(height - 1.0, float(bbox[1]))),
            max(0.0, min(width - 1.0, float(bbox[2]))),
            max(0.0, min(height - 1.0, float(bbox[3]))),
        ]
    except Exception:
        bbox = None
    info.update(
        {
            "ok": True,
            "visible": True,
            "uv": [u, v],
            "bbox_xyxy": bbox,
            "yaw_deg": parsed.get("yaw_deg"),
            "p1_uv": parsed.get("p1_uv"),
            "p2_uv": parsed.get("p2_uv"),
            "confidence": float(parsed.get("confidence") or 0.0),
            "note": parsed.get("note"),
        }
    )
    return info


# ---------------------------------------------------------------------------
# 端点可用性
# ---------------------------------------------------------------------------


def vlm_available() -> bool:
    """端点是否可用（``vlm_enabled=True`` 且 ``vlm_url`` 非空）。"""
    settings, error = _resolve_config()
    if error is not None or settings is None:
        return False
    return bool(settings.vlm_enabled and settings.vlm_url.strip())


# 语义化别名，方便下游按需选用。
is_vlm_available = vlm_available


# ---------------------------------------------------------------------------
# 底层 HTTP 调用
# ---------------------------------------------------------------------------


def vlm_chat(prompt: str, images: list[np.ndarray], *, timeout_s: int | None = None) -> dict:
    """向 OpenAI 兼容端点发一次多模态 chat 请求。

    返回 ``{"ok", "mode": "http", "text", "error", "latency_s"}``；端点不可用、
    网络异常、超时或解析失败时 ``ok=False`` 并给出 ``error``，**不抛异常**。
    """
    settings, cfg_error = _resolve_config()
    if cfg_error is not None or settings is None:
        info = _base_info(None)
        info["error"] = cfg_error or "settings_unavailable"
        return info

    info = _base_info(settings)
    # 未配置端点：立即返回，不发网络请求、不阻塞。
    if not settings.vlm_enabled or not settings.vlm_url.strip():
        info["error"] = "vlm_disabled"
        info["vlm_enabled"] = bool(settings.vlm_enabled)
        return info

    timeout_value = int(timeout_s or settings.vlm_timeout_s)
    requests_module, import_error = _import_requests()
    if requests_module is None:
        info["error"] = import_error or "requests_missing"
        return info

    # 组装 OpenAI 兼容 content 列表：先文本，再逐张 base64 data URL 图片。
    content: list[dict] = [{"type": "text", "text": str(prompt)}]
    try:
        for image in images or []:
            png, encode_error = _png_bytes(image)
            if png is None:
                info["error"] = encode_error or "image_encode_failed"
                return info
            content.append({"type": "image_url", "image_url": {"url": _data_url(png)}})
    except Exception as exc:  # 图片本身异常（比如 None）也要降级而不是抛
        info["error"] = f"image_prepare_failed: {type(exc).__name__}: {exc}"
        return info

    body = {
        "model": settings.vlm_model,
        "temperature": 0.0,
        "max_tokens": _MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }

    started = time.perf_counter()
    try:
        reply = requests_module.post(
            settings.vlm_url, json=body, timeout=max(5, timeout_value)
        )
        reply.raise_for_status()
        payload = reply.json()
        info["latency_s"] = round(time.perf_counter() - started, 3)
        if not isinstance(payload, dict):
            info["error"] = "VLM endpoint returned non-object JSON"
            return info
        choices = payload.get("choices") or [{}]
        if not isinstance(choices, list) or not choices:
            choices = [{}]
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        text = (message or {}).get("content") if isinstance(message, dict) else ""
        info["text"] = text or ""
        info["ok"] = True
        info["error"] = None
        return info
    except Exception as exc:
        # 网络 / 超时 / HTTP 状态码 / JSON 解析失败：全部走降级。
        info["latency_s"] = round(time.perf_counter() - started, 3)
        info["error"] = _error_text(exc, timeout_value)
        return info


# ---------------------------------------------------------------------------
# 牛奶定位
# ---------------------------------------------------------------------------


def locate_milk(
    image: Any,
    *,
    camera: str = "cam_left_wrist",
    output_dir: Any = None,
    timeout_s: int | None = None,
) -> dict:
    """用 VLM 在（腕部）RGB 帧里定位牛奶盒。

    ``ok=True`` 表示定位可用；端点不可用时立即返回 ``error="vlm_disabled"``。
    """
    settings, cfg_error = _resolve_config()
    if cfg_error is not None or settings is None:
        info = _base_info(None)
        info["error"] = cfg_error or "settings_unavailable"
        info["camera"] = camera
        return info
    if not settings.vlm_enabled or not settings.vlm_url.strip():
        info = _disabled_info(settings)
        info["camera"] = camera
        return info

    # 图像准备也包在 try 里：非法输入只降级，绝不抛异常。
    try:
        image = _uint8(image)
        height, width = int(image.shape[0]), int(image.shape[1])
    except Exception as exc:
        info = _base_info(settings)
        info["error"] = f"image_prepare_failed: {type(exc).__name__}: {exc}"
        info["camera"] = camera
        return info
    png, encode_error = _png_bytes(image)
    if png is None:
        info = _base_info(settings)
        info["error"] = encode_error or "image_encode_failed"
        info["camera"] = camera
        return info
    _dump_debug(output_dir, "milk_frame.png", image, None)

    reply = vlm_chat(_prompt(width, height), [image], timeout_s=timeout_s)
    # 注意：chat 层的 ok 表示"HTTP 成功"；定位层的 ok 表示"可用"，由 _finish 决定。
    result = _base_info(settings)
    result["http_ok"] = bool(reply.get("ok"))
    result["text"] = reply.get("text", "")
    result["error"] = reply.get("error")
    result["latency_s"] = reply.get("latency_s", 0.0)
    result["camera"] = camera
    parsed = _json_from_text(result["text"]) if result["http_ok"] else None
    transport_error = result["error"]
    if not result["http_ok"] and transport_error is None:
        transport_error = "VLM request failed"
    result = _finish(result, parsed, width, height)
    if parsed is None and transport_error:
        # _finish 会把 error 改写成"无法解析 JSON"；这里保留传输层（网络/超时）的
        # 真实原因，便于下游 summary 与日志区分降级类型。
        result["parse_error"] = result.get("error")
        result["error"] = transport_error
    return result


# ---------------------------------------------------------------------------
# 放置 / 释放判断（high + wrist 双图）
# ---------------------------------------------------------------------------


def judge_place_release(
    high_image: Any,
    wrist_image: Any,
    *,
    output_dir: Any = None,
    timeout_s: int = 60,
) -> dict:
    """询问 VLM：左夹爪现在是否可以把牛奶放进篮子。

    保守默认：不可释放（``place_ready=False`` / ``place_relation="unknown"``）。
    """
    defaults = {
        "place_ready": False,
        "place_relation": "unknown",
        "milk_inside_basket": False,
        "over_basket_opening": False,
        "gripper_clear": False,
        "confidence": 0.0,
        "reason": None,
    }

    settings, cfg_error = _resolve_config()
    if cfg_error is not None or settings is None:
        info = _base_info(None)
        info["error"] = cfg_error or "settings_unavailable"
        info.update(defaults)
        return info
    if not settings.vlm_enabled or not settings.vlm_url.strip():
        info = _disabled_info(settings)
        info.update(defaults)
        return info

    panel, compose_error = _compose_panels(high_image, wrist_image)
    if panel is None:
        info = _base_info(settings)
        info["error"] = compose_error or "image_compose_failed"
        info.update(defaults)
        return info

    reply = vlm_chat(
        _place_release_prompt(panel.shape[1], panel.shape[0]),
        [panel],
        timeout_s=timeout_s,
    )
    _dump_debug(output_dir, "place_release_frame.png", panel, reply.get("text"))

    info = _base_info(settings)
    info["http_ok"] = bool(reply.get("ok"))
    info["latency_s"] = reply.get("latency_s", 0.0)
    info["text"] = reply.get("text", "")
    parsed = _json_from_text(info["text"]) if info["http_ok"] else None
    info["answer"] = parsed
    if parsed is None:
        info["error"] = reply.get("error") or "VLM returned no parseable JSON"
        info.update(defaults)
        return info

    info["ok"] = True
    info["error"] = reply.get("error")
    # 关系字段容错：模型偶尔不给枚举值，就从 reason 的自由文本里猜（与原实现一致）。
    relation = str(parsed.get("place_relation", "") or "").lower()
    reason = str(parsed.get("reason", "") or "").lower()
    if relation not in {"inside", "edge", "outside", "unknown"}:
        if any(word in reason for word in ("edge", "rim", "near the opening", "close to the opening")):
            relation = "edge"
        elif any(word in reason for word in ("inside", "within", "over the opening")):
            relation = "inside"
        elif any(word in reason for word in ("outside", "far", "nowhere near", "beside")):
            relation = "outside"
        else:
            relation = "unknown"
    info["place_ready"] = bool(parsed.get("place_ready", False))
    info["place_relation"] = relation
    info["milk_inside_basket"] = bool(parsed.get("milk_inside_basket", False))
    info["over_basket_opening"] = bool(parsed.get("over_basket_opening", False))
    info["gripper_clear"] = bool(parsed.get("gripper_clear", False))
    info["confidence"] = float(parsed.get("confidence") or 0.0)
    info["reason"] = parsed.get("reason")
    return info


def judge_basket_release(
    high_image: Any,
    wrist_image: Any,
    *,
    output_dir: Any = None,
    timeout_s: int = 60,
) -> dict:
    """询问 VLM：右夹爪现在是否可以把篮子放下（保留原 API 与 prompt）。"""
    defaults = {
        "release_ready": False,
        "basket_supported": False,
        "basket_stable": False,
        "gripper_clear": False,
        "confidence": 0.0,
        "reason": None,
    }

    settings, cfg_error = _resolve_config()
    if cfg_error is not None or settings is None:
        info = _base_info(None)
        info["error"] = cfg_error or "settings_unavailable"
        info.update(defaults)
        return info
    if not settings.vlm_enabled or not settings.vlm_url.strip():
        info = _disabled_info(settings)
        info.update(defaults)
        return info

    panel, compose_error = _compose_panels(high_image, wrist_image)
    if panel is None:
        info = _base_info(settings)
        info["error"] = compose_error or "image_compose_failed"
        info.update(defaults)
        return info

    reply = vlm_chat(
        _basket_release_prompt(panel.shape[1], panel.shape[0]),
        [panel],
        timeout_s=timeout_s,
    )
    _dump_debug(output_dir, "basket_release_frame.png", panel, reply.get("text"))

    info = _base_info(settings)
    info["http_ok"] = bool(reply.get("ok"))
    info["latency_s"] = reply.get("latency_s", 0.0)
    info["text"] = reply.get("text", "")
    parsed = _json_from_text(info["text"]) if info["http_ok"] else None
    info["answer"] = parsed
    if parsed is None:
        info["error"] = reply.get("error") or "VLM returned no parseable JSON"
        info.update(defaults)
        return info

    info["ok"] = True
    info["error"] = reply.get("error")
    info["release_ready"] = bool(
        parsed.get("release_ready", False) or parsed.get("basket_supported", False)
    )
    info["basket_supported"] = bool(parsed.get("basket_supported", False))
    info["basket_stable"] = bool(parsed.get("basket_stable", False))
    info["gripper_clear"] = bool(parsed.get("gripper_clear", False))
    info["confidence"] = float(parsed.get("confidence") or 0.0)
    info["reason"] = parsed.get("reason")
    return info


def judge_handle_grasp(
    high_image: Any,
    wrist_image: Any,
    *,
    output_dir: Any = None,
    timeout_s: int = 60,
) -> dict:
    """询问 VLM：篮子手柄是否已经夹在右夹爪两指之间（保留原字段）。"""
    defaults = {
        "handle_between_fingers": False,
        "handle_visible": False,
        "confidence": 0.0,
        "reason": None,
    }

    settings, cfg_error = _resolve_config()
    if cfg_error is not None or settings is None:
        info = _base_info(None)
        info["error"] = cfg_error or "settings_unavailable"
        info.update(defaults)
        return info
    if not settings.vlm_enabled or not settings.vlm_url.strip():
        info = _disabled_info(settings)
        info.update(defaults)
        return info

    panel, compose_error = _compose_panels(high_image, wrist_image)
    if panel is None:
        info = _base_info(settings)
        info["error"] = compose_error or "image_compose_failed"
        info.update(defaults)
        return info

    reply = vlm_chat(
        _handle_grasp_prompt(panel.shape[1], panel.shape[0]),
        [panel],
        timeout_s=timeout_s,
    )
    _dump_debug(output_dir, "handle_grasp_frame.png", panel, reply.get("text"))

    info = _base_info(settings)
    info["http_ok"] = bool(reply.get("ok"))
    info["latency_s"] = reply.get("latency_s", 0.0)
    info["text"] = reply.get("text", "")
    parsed = _json_from_text(info["text"]) if info["http_ok"] else None
    info["answer"] = parsed
    if parsed is None:
        info["error"] = reply.get("error") or "VLM returned no parseable JSON"
        info.update(defaults)
        return info

    info["ok"] = True
    info["error"] = reply.get("error")
    info["handle_between_fingers"] = bool(parsed.get("handle_between_fingers", False))
    info["handle_visible"] = bool(parsed.get("handle_visible", False))
    info["confidence"] = float(parsed.get("confidence") or 0.0)
    info["reason"] = parsed.get("reason")
    return info
