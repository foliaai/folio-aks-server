#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""ModelNexus（model_lake）能力自动合成单元测试。

覆盖：
1. infer_thinking_level_map：厂商档位推导（与 ThinkingAdapter 同源）
2. infer_default_thinking_level：默认档位规则
3. _fetch_model_lake：/v1/models 富字段合成（思考/多模态/上下文/deprecated 过滤/spec 回写）
4. _filter_hidden_models：黑名单语义（空=全可见）
5. _validate_presets：preset 失配不抛异常
全部离线（mock payload，不触网）。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.client.llm.registry import LiteLLMRegistry, ThinkingModelSpec, LLMModelInfo
from src.client.llm.thinking_adapter import (
    infer_default_thinking_level,
    infer_thinking_level_map,
)


def _nexus_model(mid, *, modalities=("text",), effort=False, reasoning=True,
                  ctx=1000000, deprecated=None):
    sp = ["temperature", "max_tokens", "tools"]
    if reasoning:
        sp.append("reasoning")
    if effort:
        sp.append("reasoning_effort")
    return {
        "id": mid,
        "name": mid,
        "context_length": ctx,
        "architecture": {"input_modalities": list(modalities)},
        "supported_parameters": sp,
        "top_provider": {"context_length": ctx, "max_completion_tokens": 131072},
        "deprecated_at": deprecated,
    }


# 模拟 ModelNexus /v1/models 富响应（覆盖各能力组合）
_MOCK_PAYLOAD = {
    "data": [
        _nexus_model("deepseek/deepseek-v4.1-flash", modalities=("text", "image"), effort=True),
        _nexus_model("deepseek/deepseek-v4-flash", deprecated="2026-09-29T16:00:00Z"),
        _nexus_model("local/qwen3.6", reasoning=False, ctx=256000),
        _nexus_model("moonshotai/kimi-k3", modalities=("text", "image", "video"), effort=True),
        # 网关误标 effort：本地厂商规则否决，应显示为开关式（仅 off/medium）
        _nexus_model("qwen/qwen3.5-flash", modalities=("text", "image", "video"), effort=True),
        _nexus_model("qwen/qwen3.7-flash", modalities=("text", "image", "video")),
        _nexus_model("qwen/qwen3.8-max", modalities=("text", "image", "video"), effort=True),
        _nexus_model("z-ai/glm-5.3", effort=True),
    ]
}


class TestInferThinkingLevelMap(unittest.TestCase):
    def test_deepseek_native_levels_only(self):
        m = infer_thinking_level_map("deepseek/deepseek-v4.1-flash")
        self.assertEqual(m["off"], "none")
        self.assertEqual(m["low"], "low")
        self.assertEqual(m["high"], "high")
        self.assertEqual(m["max"], "max")
        # 非原生档置 None（前端不展示，传入时 clamp 归位）
        self.assertIsNone(m["minimal"])
        self.assertIsNone(m["medium"])
        self.assertIsNone(m["xhigh"])

    def test_qwen38_native_levels_only(self):
        m = infer_thinking_level_map("qwen/qwen3.8-max")
        self.assertEqual(m["off"], "none")
        self.assertEqual(m["low"], "low")
        self.assertEqual(m["medium"], "medium")
        self.assertEqual(m["xhigh"], "xhigh")
        self.assertIsNone(m["high"])
        self.assertIsNone(m["max"])

    def test_glm53_native_levels_only(self):
        m = infer_thinking_level_map("z-ai/glm-5.3")
        self.assertEqual(m["off"], "none")     # adapter 转 low+disabled
        self.assertEqual(m["low"], "low")
        self.assertEqual(m["high"], "high")
        self.assertEqual(m["max"], "max")
        self.assertIsNone(m["medium"])
        self.assertIsNone(m["xhigh"])

    def test_openai_cannot_turn_off(self):
        m = infer_thinking_level_map("gpt-5")
        self.assertIsNone(m["off"])            # 推理不可完全关闭
        self.assertIsNone(m["minimal"])
        self.assertEqual(m["high"], "high")
        self.assertIsNone(m["max"])

    def test_kimi_native_levels(self):
        """Kimi K3：始终推理（off=null 归位 low），原生档 low/high/max"""
        m = infer_thinking_level_map("moonshotai/kimi-k3")
        self.assertIsNone(m["off"])
        self.assertEqual(m["low"], "low")
        self.assertEqual(m["high"], "high")
        self.assertEqual(m["max"], "max")
        self.assertIsNone(m["medium"])
        self.assertIsNone(m["xhigh"])

    def test_unknown_vendor_passthrough(self):
        m = infer_thinking_level_map("unknown-vendor/some-model-x")
        self.assertEqual(m["off"], "none")
        self.assertEqual(m["max"], "max")      # 全档透传

    def test_local_rules_override_toggle_only(self):
        """本地规则明确判定仅开关的模型返回 None（优先于网关 effort 声明）"""
        self.assertIsNone(infer_thinking_level_map("qwen/qwen3.5-flash"))
        self.assertIsNone(infer_thinking_level_map("qwen/qwen3.7-flash"))
        self.assertIsNone(infer_thinking_level_map("z-ai/glm-5.1"))
        self.assertIsNone(infer_thinking_level_map("xiaomi-mimo/xiaomi-mimo-v2-5"))
        # 3.8 / 5.3 / 5.2 等强度模型不受影响
        self.assertIsNotNone(infer_thinking_level_map("qwen/qwen3.8-max"))
        self.assertIsNotNone(infer_thinking_level_map("z-ai/glm-5.3"))

    def test_prefix_stripped(self):
        self.assertEqual(
            infer_thinking_level_map("openai/deepseek/deepseek-v4.1-flash"),
            infer_thinking_level_map("deepseek/deepseek-v4.1-flash"),
        )

    def test_default_level_rule(self):
        self.assertEqual(infer_default_thinking_level(
            infer_thinking_level_map("deepseek/deepseek-v4.1-flash")), "high")
        self.assertEqual(infer_default_thinking_level(
            infer_thinking_level_map("qwen/qwen3.8-max")), "medium")
        self.assertIsNone(infer_default_thinking_level({"off": "none"}))


class TestFetchModelLake(unittest.TestCase):
    def setUp(self):
        # 显式锁定 model_lake 档案（_fetch_model_lake 执行期间读取 hidden/presets
        # 也会走档案解析），避免受本地 .env 的 MODEL_GATEWAY_TYPE 影响——
        # load_dotenv(override=True) 会覆盖进程环境变量
        patcher = patch(
            "src.utils.config_profile.resolve_config_profile",
            return_value="model_lake",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _registry(self):
        return LiteLLMRegistry()

    def test_synthesize_and_writeback(self):
        reg = self._registry()
        visible = reg._fetch_model_lake(_MOCK_PAYLOAD, {}, "model_lake")

        # 8 个 mock 模型 - 1 弃用 - 1 命中档案黑名单(local/qwen3.6) = 6 个可见
        ids = [m.id for m in visible]
        self.assertEqual(len(visible), 6)
        self.assertNotIn("openai/deepseek/deepseek-v4-flash", ids)
        self.assertNotIn("openai/local/qwen3.6", ids)
        self.assertIn("openai/deepseek/deepseek-v4.1-flash", ids)
        self.assertIn("openai/z-ai/glm-5.3", ids)

        # label 去掉 channel 前缀；排序先按 provider 分组、组内按名称
        by_id = {m.id: m for m in visible}
        self.assertEqual(by_id["openai/z-ai/glm-5.3"].label, "glm-5.3")
        self.assertEqual(by_id["openai/deepseek/deepseek-v4.1-flash"].label, "deepseek-v4.1-flash")
        order = [m.provider for m in visible]
        self.assertEqual(order, sorted(order, key=str.lower))
        # 同 provider（qwen）的模型相邻
        qwen_idx = [i for i, m in enumerate(visible) if m.provider == "qwen"]
        self.assertEqual(qwen_idx, list(range(qwen_idx[0], qwen_idx[0] + len(qwen_idx))))

        # 思考声明：effort 型（glm-5.3：low/high/max + off）
        glm = by_id["openai/z-ai/glm-5.3"]
        self.assertTrue(glm.supports_thinking)
        self.assertIn("high", glm.thinking_levels)
        self.assertNotIn("medium", glm.thinking_levels)
        self.assertEqual(glm.default_thinking_level, "high")

        # 开关式（qwen3.7-flash：off/medium 两档）
        qwen = by_id["openai/qwen/qwen3.7-flash"]
        self.assertTrue(qwen.supports_thinking)
        self.assertEqual(qwen.thinking_levels, ["off", "medium"])

        # 本地规则否决网关 effort 声明：qwen3.5 网关标了 effort，
        # 但 Qwen 3.5 官方仅思考开关——展示必须与 adapter 行为一致
        qwen35 = by_id["openai/qwen/qwen3.5-flash"]
        self.assertTrue(qwen35.supports_thinking)
        self.assertEqual(qwen35.thinking_levels, ["off", "medium"])
        self.assertEqual(reg.resolve_reasoning_effort("openai/qwen/qwen3.5-flash", "medium"), "enabled")

        # Kimi K3：始终推理不可关（前端无 off 档），low/high/max
        kimi = by_id["openai/moonshotai/kimi-k3"]
        self.assertTrue(kimi.supports_thinking)
        self.assertEqual(kimi.thinking_levels, ["low", "high", "max"])
        self.assertEqual(kimi.default_thinking_level, "high")
        self.assertEqual(reg.resolve_reasoning_effort("openai/moonshotai/kimi-k3", "low"), "low")

        # 多模态：input_modalities 含 image
        self.assertTrue(by_id["openai/qwen/qwen3.7-flash"].supports_multimodal)
        self.assertTrue(by_id["openai/moonshotai/kimi-k3"].supports_multimodal)
        self.assertFalse(glm.supports_multimodal)

        # 上下文长度来自 context_length
        self.assertEqual(glm.max_context, 1000000)
        # 黑名单模型（local/qwen3.6）前端不可见，但能力合成照常（回写字典仍有它）
        self.assertNotIn("qwen3.6", [m.label for m in visible])
        self.assertEqual(reg._long_context_map.get("qwen3.6"), 256000)

        # 运行时翻译字典已回写：SDK id / channel 路由 / 裸名 都能命中
        self.assertIsNotNone(reg.peek_thinking_spec("openai/z-ai/glm-5.3"))
        self.assertIsNotNone(reg.peek_thinking_spec("z-ai/glm-5.3"))
        self.assertIsNotNone(reg.peek_thinking_spec("glm-5.3"))
        self.assertEqual(reg.resolve_reasoning_effort("openai/z-ai/glm-5.3", "off"), "none")
        self.assertEqual(reg.resolve_reasoning_effort("openai/z-ai/glm-5.3", "xhigh"), "max")
        self.assertEqual(reg.resolve_reasoning_effort("openai/qwen/qwen3.7-flash", "medium"), "enabled")

        # 多模态集合与长上下文回写
        self.assertIn("qwen/qwen3.7-flash", reg._multimodal_models)
        self.assertEqual(reg._long_context_map.get("glm-5.3"), 1000000)

    def test_hidden_blacklist(self):
        models = [
            LLMModelInfo(id="openai/a/one", label="one", provider="a"),
            LLMModelInfo(id="openai/a/two", label="two", provider="a"),
        ]
        # 空黑名单 = 全可见
        self.assertEqual(len(LiteLLMRegistry._filter_hidden_models(models, [])), 2)
        self.assertEqual(len(LiteLLMRegistry._filter_hidden_models(models, None)), 2)
        # 黑名单按去前缀形态匹配
        out = LiteLLMRegistry._filter_hidden_models(models, ["a/one", "openai/a/two"])
        self.assertEqual([m.id for m in out], [])

    def test_validate_presets_no_raise_on_miss(self):
        reg = self._registry()
        # 只有一个网关模型，presets 大多失配——校验只打日志不抛异常
        reg._validate_presets(["qwen/qwen3.7-flash"])
        # 命中时不报错
        reg._validate_presets(["qwen/qwen3.7-flash", "z-ai/glm-5.3", "deepseek/deepseek-v4.1-flash"])


if __name__ == "__main__":
    unittest.main()
