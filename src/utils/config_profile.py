#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""=================================================
@PROJECT_NAME: agentic_knowledge_system
@File    : config_profile.py
@Author  : caixiongjiang
@Date    : 2026/08/20
@Function:
    配置档案：公共策略在 ``config/config.toml``，模型身份与角色参数在
    ``config/profiles/<name>/``。档案跟随 ``MODEL_GATEWAY_TYPE``（openai* → model_lake）。
@Copyright：Copyright(c) 2024-2026. All Rights Reserved
=================================================="""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
PROFILES_ROOT = CONFIG_ROOT / "profiles"

PROFILE_ALIASES = {
    "openai": "model_lake",
    "openai_compatible": "model_lake",
}
KNOWN_PROFILES = frozenset({"litellm", "model_lake"})
DEFAULT_PROFILE = "litellm"


def resolve_config_profile(explicit: Optional[str] = None) -> str:
    """解析当前配置档案名（默认跟随 MODEL_GATEWAY_TYPE）。"""
    from src.utils.env_manager import get_env_manager

    env = get_env_manager()
    raw = (
        explicit
        or env.get_model_gateway_type()
        or DEFAULT_PROFILE
    )
    name = PROFILE_ALIASES.get(str(raw).strip().lower(), str(raw).strip().lower())
    if name not in KNOWN_PROFILES:
        return DEFAULT_PROFILE
    return name


def get_profile_dir(profile: Optional[str] = None) -> Path:
    return PROFILES_ROOT / resolve_config_profile(profile)


def resolve_profile_file(filename: str, profile: Optional[str] = None) -> Path:
    """档案目录中的文件。"""
    return get_profile_dir(profile) / filename


def load_profile_presets(profile: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """读取 ``models.toml`` 的 ``[presets.<role>]``：模型 id + 采样参数。

    兼容旧写法 ``fast = "model-id"``（视为只有 ``model`` 字段）。
    """
    import toml

    path = resolve_profile_file("models.toml", profile)
    if not path.exists():
        return {}
    data = toml.load(path) or {}
    raw = data.get("presets") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                out[str(key)] = {"model": text}
            continue
        if isinstance(value, dict):
            cleaned = {k: v for k, v in value.items() if v is not None}
            if cleaned:
                out[str(key)] = cleaned
    return out


def load_profile_preset_models(profile: Optional[str] = None) -> Dict[str, str]:
    """读取各角色的 ``model`` 字段（不含采样参数）。"""
    out: Dict[str, str] = {}
    for name, preset in load_profile_presets(profile).items():
        model = preset.get("model")
        if isinstance(model, str) and model.strip():
            out[name] = model.strip()
    return out


def load_profile_visible_models(profile: Optional[str] = None) -> List[str]:
    """读取 ``models.toml`` 的 ``visible``：前端对话主模型白名单（litellm 档案）。

    空列表表示下拉不展示任何模型。匹配时会去掉 ``openai/`` / ``litellm_proxy/``
    调用前缀，但保留 Model Lake 的 ``channel/model``。
    """
    return _load_model_list_section("visible", profile)


def load_profile_hidden_models(profile: Optional[str] = None) -> List[str]:
    """读取 ``models.toml`` 的 ``[hidden]``：前端对话主模型黑名单（model_lake 档案）。

    model_lake（ModelNexus）网关的 ``/v1/models`` 即真相源、默认全部可见，
    档案只登记"不想暴露给前端"的模型做减法；空列表（或缺 section）= 全可见。
    """
    return _load_model_list_section("hidden", profile)


def _load_model_list_section(section: str, profile: Optional[str] = None) -> List[str]:
    """读取 ``models.toml`` 中形如 ``[<section>].models = [...]`` 的模型名列表。"""
    import toml

    path = resolve_profile_file("models.toml", profile)
    if not path.exists():
        return []
    data = toml.load(path) or {}
    raw = data.get(section)
    if isinstance(raw, dict):
        raw = raw.get("models") or raw.get("ids") or []
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "CONFIG_ROOT",
    "DEFAULT_PROFILE",
    "KNOWN_PROFILES",
    "PROFILE_ALIASES",
    "PROFILES_ROOT",
    "get_profile_dir",
    "load_profile_hidden_models",
    "load_profile_preset_models",
    "load_profile_presets",
    "load_profile_visible_models",
    "resolve_config_profile",
    "resolve_profile_file",
]
