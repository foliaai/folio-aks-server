#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""配置档案：公共策略 + profiles/<name> 模型绑定"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.config_profile import (
    load_profile_hidden_models,
    load_profile_preset_models,
    load_profile_presets,
    load_profile_visible_models,
    resolve_config_profile,
    resolve_profile_file,
)
from src.client.llm.registry import LiteLLMRegistry, ThinkingModelSpec


class TestConfigProfile(unittest.TestCase):
    def test_explicit_arg_profile_wins(self):
        with patch.dict(os.environ, {
            "MODEL_GATEWAY_TYPE": "model_lake",
        }, clear=False):
            self.assertEqual(resolve_config_profile("litellm"), "litellm")

    def test_follow_gateway_type(self):
        with patch.dict(os.environ, {
            "MODEL_GATEWAY_TYPE": "model_lake",
        }, clear=False):
            self.assertEqual(resolve_config_profile(), "model_lake")

    def test_openai_alias(self):
        with patch.dict(os.environ, {"MODEL_GATEWAY_TYPE": "openai"}, clear=False):
            self.assertEqual(resolve_config_profile(), "model_lake")

    def test_unknown_falls_back_to_litellm(self):
        with patch.dict(os.environ, {"MODEL_GATEWAY_TYPE": "does-not-exist"}, clear=False):
            self.assertEqual(resolve_config_profile(), "litellm")

    def test_litellm_bindings(self):
        models = load_profile_preset_models("litellm")
        self.assertEqual(models["fast"], "qwen3.7-flash")
        self.assertEqual(models["quality"], "deepseek-v4-pro")

    def test_model_lake_bindings_keep_channel(self):
        models = load_profile_preset_models("model_lake")
        self.assertEqual(models["fast"], "qwen/qwen3.7-flash")
        self.assertIn("/", models["fast"])

    def test_config_manager_loads_profile_preset_table(self):
        with patch.dict(os.environ, {"MODEL_GATEWAY_TYPE": "model_lake"}, clear=False):
            cm = ConfigManager()
            preset = cm.get_llm_preset("fast")
            self.assertEqual(preset["model"], "qwen/qwen3.7-flash")
            self.assertEqual(preset["temperature"], 0.3)
            self.assertEqual(preset["thinking_level"], "off")
            self.assertEqual(preset["timeout"], 60)

        lake = load_profile_presets("model_lake")
        lite = load_profile_presets("litellm")
        self.assertEqual(lake["fast"]["model"], "qwen/qwen3.7-flash")
        self.assertEqual(lite["fast"]["model"], "qwen3.7-flash")
        self.assertIn("temperature", lake["fast"])
        self.assertIn("thinking_level", lake["reasoning"])

    def test_capability_files_come_from_profile(self):
        """litellm 档案继续用 json 声明；model_lake 档案 json 已废弃（能力从接口合成）。"""
        # 显式 patch 档案解析：本地 .env 的 MODEL_GATEWAY_TYPE 经
        # load_dotenv(override=True) 会覆盖进程环境变量，patch os.environ 不可靠
        with patch("src.utils.config_profile.resolve_config_profile", return_value="litellm"):
            path = resolve_profile_file("thinking_models.json")
            self.assertTrue(path.exists())
            self.assertIn("profiles/litellm", str(path).replace("\\", "/"))
            specs = LiteLLMRegistry._load_thinking_models()
            self.assertIn("qwen3.7-flash", specs)

        with patch("src.utils.config_profile.resolve_config_profile", return_value="model_lake"):
            path = resolve_profile_file("thinking_models.json")
            self.assertFalse(path.exists())
            # 启动时静态声明为空；model_lake 的能力声明由 /v1/models 富字段
            # 在 list_models 刷新时合成并回写（见 test_modelnexus_capabilities）
            self.assertEqual(LiteLLMRegistry._load_thinking_models(), {})

    def test_model_lake_thinking_lookup_accepts_channel_route(self):
        """声明 key 双形态（channel/model + 裸名）时，SDK id 也必须能查到。"""
        with patch.dict(os.environ, {"MODEL_GATEWAY_TYPE": "model_lake"}, clear=False):
            reg = LiteLLMRegistry()
            # 模拟接口合成回写后的声明字典（routed + bare 双 key）
            reg._thinking_models = {
                "deepseek/deepseek-v4-flash": ThinkingModelSpec(
                    reasoning=True, supports_thinking_effort=True,
                    default="high",
                    thinking_level_map={"off": "none", "low": "low", "high": "high", "max": "max"},
                ),
                "qwen3.7-flash": ThinkingModelSpec(
                    reasoning=True, supports_thinking_effort=False,
                ),
            }
            official = "openai/deepseek/deepseek-v4-flash"
            qwen = "openai/qwen/qwen3.7-flash"

            self.assertEqual(reg.resolve_reasoning_effort(official, "high"), "high")
            self.assertEqual(reg.clamp_thinking_level(official, "high"), "high")
            self.assertIsNotNone(reg.peek_thinking_spec(official))

            # 只写最后一段的 Qwen 条目仍按裸名命中
            self.assertEqual(reg.resolve_reasoning_effort(qwen, "medium"), "enabled")
            self.assertEqual(reg.clamp_thinking_level(qwen, "high"), "medium")

    def test_thinking_miss_passthrough_keeps_user_level(self):
        """档案对不上时不能把用户打开的思考档位钳成 off / None。"""
        with patch.dict(os.environ, {"MODEL_GATEWAY_TYPE": "model_lake"}, clear=False):
            reg = LiteLLMRegistry()
            unknown = "openai/unknown-channel/deepseek-v4-flash-unknown"
            self.assertEqual(reg.clamp_thinking_level(unknown, "high"), "high")
            self.assertEqual(reg.resolve_reasoning_effort(unknown, "high"), "high")
            self.assertEqual(reg.clamp_thinking_level(unknown, "off"), "off")
            self.assertIsNone(reg.resolve_reasoning_effort(unknown, "off"))

    def test_model_lake_hidden_replaces_visible(self):
        """model_lake 档案：visible 白名单废弃，hidden 黑名单默认含 local/qwen3.6。"""
        presets = load_profile_preset_models("model_lake")
        hidden = load_profile_hidden_models("model_lake")
        self.assertIn("fast", presets)
        self.assertEqual(hidden, ["local/qwen3.6"])
        self.assertEqual(load_profile_visible_models("model_lake"), [])

        litellm_visible = load_profile_visible_models("litellm")
        self.assertIn("qwen3.7-flash", litellm_visible)


if __name__ == "__main__":
    unittest.main()
