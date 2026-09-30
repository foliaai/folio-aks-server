#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
模型网关切换与多网关适配单元测试
验证：
1. MODEL_GATEWAY_TYPE 控制 LLM 网关类型（litellm / model_lake / openai_compatible）
2. LLM 网关 URL / KEY 环境变量优先级及 fallback 行为
3. Embedding / Reranker 网关保持独立配置并 fallback 到 LITELLM_PROXY_*
4. LLMClient 前缀自动路由（_ensure_gateway_routable）
5. ThinkingAdapter 去除前缀并正确适配
6. ConfigManager 各网关独立配置解析
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.env_manager import EnvManager
from src.utils.config_manager import ConfigManager
from src.client.llm.client import _ensure_gateway_routable
from src.client.llm.registry import LiteLLMRegistry
from src.client.llm.thinking_adapter import get_thinking_adapter, QwenThinkingAdapter, GLMThinkingAdapter


class TestModelGatewaySwitching(unittest.TestCase):
    def test_ensure_gateway_routable_model_lake(self):
        """测试 model_lake / openai 网关下的模型前缀路由"""
        # 裸名 -> 添加 openai/ 前缀
        self.assertEqual(_ensure_gateway_routable("qwen-plus", "model_lake"), "openai/qwen-plus")
        self.assertEqual(_ensure_gateway_routable("gpt-4o", "openai"), "openai/gpt-4o")
        self.assertEqual(_ensure_gateway_routable("glm-4-plus", "openai_compatible"), "openai/glm-4-plus")

        # 原带 litellm_proxy/ 前缀 -> 剥离并替换为 openai/
        self.assertEqual(_ensure_gateway_routable("litellm_proxy/qwen-plus", "model_lake"), "openai/qwen-plus")
        self.assertEqual(_ensure_gateway_routable("litellm_proxy/openai/gpt-4o", "model_lake"), "openai/gpt-4o")

        # 已带 openai/ 前缀 -> 保持 openai/
        self.assertEqual(_ensure_gateway_routable("openai/qwen-plus", "model_lake"), "openai/qwen-plus")

    def test_ensure_gateway_routable_litellm(self):
        """测试 litellm 网关下的模型前缀路由"""
        # 裸名 -> 添加 litellm_proxy/ 前缀
        self.assertEqual(_ensure_gateway_routable("qwen-plus", "litellm"), "litellm_proxy/qwen-plus")
        self.assertEqual(_ensure_gateway_routable("gpt-4o", "litellm"), "litellm_proxy/gpt-4o")

        # 已带 provider -> 保持原样
        self.assertEqual(_ensure_gateway_routable("openai/gpt-4o", "litellm"), "openai/gpt-4o")
        self.assertEqual(_ensure_gateway_routable("anthropic/claude-3-5-sonnet", "litellm"), "anthropic/claude-3-5-sonnet")
        self.assertEqual(_ensure_gateway_routable("litellm_proxy/glm-4", "litellm"), "litellm_proxy/glm-4")

    def test_env_manager_gateway_hierarchy(self):
        """测试环境变量优先级与 fallback 机制"""
        env = EnvManager()

        # 1. 默认 fallback 到 LITELLM_PROXY_*
        with patch.dict(os.environ, {
            "LITELLM_PROXY_URL": "http://litellm.internal:4000",
            "LITELLM_PROXY_KEY": "sk-litellm-default",
        }, clear=True):
            self.assertEqual(env.get_model_gateway_type(), "litellm")
            self.assertEqual(env.get_model_gateway_url(), "http://litellm.internal:4000")
            self.assertEqual(env.get_model_gateway_key(), "sk-litellm-default")
            self.assertEqual(env.get_embedding_gateway_url(), "http://litellm.internal:4000")
            self.assertEqual(env.get_reranker_gateway_url(), "http://litellm.internal:4000")

        # 1b. 切回 litellm 时，残留 MODEL_LAKE_BASE 不能抢走 URL
        with patch.dict(os.environ, {
            "MODEL_GATEWAY_TYPE": "litellm",
            "LITELLM_PROXY_URL": "http://localhost:4000",
            "LITELLM_PROXY_KEY": "sk-litellm-default",
            "MODEL_LAKE_BASE": "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
        }, clear=True):
            self.assertEqual(env.get_model_gateway_type(), "litellm")
            self.assertEqual(env.get_model_gateway_url(), "http://localhost:4000")
            self.assertEqual(env.get_model_gateway_key(), "sk-litellm-default")

        # 2. 切换 LLM 到 model_lake，Embedding / Reranker 仍走 litellm
        with patch.dict(os.environ, {
            "MODEL_GATEWAY_TYPE": "model_lake",
            "MODEL_LAKE_BASE": "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
            "MODEL_LAKE_API_KEY": "mn-sk-test-key",
            "LITELLM_PROXY_URL": "http://litellm.internal:4000",
            "LITELLM_PROXY_KEY": "sk-litellm-key",
        }, clear=True):
            self.assertEqual(env.get_model_gateway_type(), "model_lake")
            self.assertEqual(
                env.get_model_gateway_url(),
                "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
            )
            self.assertEqual(env.get_model_gateway_key(), "mn-sk-test-key")
            # Embedding 和 Reranker 走 LiteLLM
            self.assertEqual(env.get_embedding_gateway_url(), "http://litellm.internal:4000")
            self.assertEqual(env.get_embedding_gateway_key(), "sk-litellm-key")
            self.assertEqual(env.get_reranker_gateway_url(), "http://litellm.internal:4000")
            self.assertEqual(env.get_reranker_gateway_key(), "sk-litellm-key")

    def test_normalize_model_lake_api_base(self):
        """ModelNexus base 原样使用：仅容忍首尾空白与末尾斜杠，不做路径拼接"""
        from src.utils.env_manager import EnvManager

        with patch.dict(os.environ, {
            "MODEL_GATEWAY_TYPE": "model_lake",
            "MODEL_LAKE_BASE": "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
            "LITELLM_PROXY_URL": "http://litellm.internal:4000",
        }, clear=True):
            env = EnvManager.__new__(EnvManager)
            self.assertEqual(
                env.get_model_gateway_url(),
                "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
            )
        # 末尾斜杠被清掉
        with patch.dict(os.environ, {
            "MODEL_GATEWAY_TYPE": "model_lake",
            "MODEL_LAKE_BASE": "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1/",
        }, clear=True):
            env = EnvManager.__new__(EnvManager)
            self.assertEqual(
                env.get_model_gateway_url(),
                "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
            )

    def test_config_manager_gateway_configs(self):
        """测试 ConfigManager 获取各独立网关配置"""
        cm = ConfigManager()

        with patch.dict(os.environ, {
            "MODEL_GATEWAY_TYPE": "model_lake",
            "MODEL_LAKE_BASE": "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
            "MODEL_LAKE_API_KEY": "mn-sk-test-key",
            "LITELLM_PROXY_URL": "http://litellm.internal:4000",
            "LITELLM_PROXY_KEY": "sk-litellm-key",
        }, clear=True):
            # LLM 网关
            llm_cfg = cm.get_llm_gateway_full_config()
            self.assertEqual(llm_cfg["gateway_type"], "model_lake")
            self.assertEqual(
                llm_cfg["api_base"],
                "https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
            )
            self.assertEqual(llm_cfg["api_key"], "mn-sk-test-key")

            # Embedding 网关（走 litellm）
            emb_cfg = cm.get_embedding_gateway_full_config()
            self.assertEqual(emb_cfg["api_base"], "http://litellm.internal:4000")
            self.assertEqual(emb_cfg["api_key"], "sk-litellm-key")

            # Reranker 网关（走 litellm）
            rerank_cfg = cm.get_reranker_gateway_full_config()
            self.assertEqual(rerank_cfg["api_base"], "http://litellm.internal:4000")
            self.assertEqual(rerank_cfg["api_key"], "sk-litellm-key")

        with patch.dict(os.environ, {
            "MODEL_GATEWAY_TIMEOUT": "90",
            "MODEL_GATEWAY_MAX_RETRIES": "5",
        }, clear=False):
            llm_cfg = cm.get_llm_gateway_full_config()
            self.assertEqual(llm_cfg["default_timeout"], 90.0)
            self.assertEqual(llm_cfg["default_max_retries"], 5)

    def test_thinking_adapter_prefix_handling(self):
        """测试 ThinkingAdapter 在不同前缀下的适配能力"""
        # qwen3.7-plus 带 openai/ 前缀 (model_lake)
        adapter1 = get_thinking_adapter("openai/qwen3.7-plus")
        self.assertIsInstance(adapter1, QwenThinkingAdapter)
        extra1 = adapter1.adapt("openai/qwen3.7-plus", "high")
        self.assertNotIn("reasoning_effort", extra1)
        self.assertTrue(extra1.get("extra_body", {}).get("enable_thinking"))
        self.assertNotIn("thinking_budget", extra1.get("extra_body", {}))

        # qwen3.7-plus 带 litellm_proxy/ 前缀 (litellm)
        adapter2 = get_thinking_adapter("litellm_proxy/qwen3.7-plus")
        self.assertIsInstance(adapter2, QwenThinkingAdapter)
        extra2 = adapter2.adapt("litellm_proxy/qwen3.7-plus", "high")
        self.assertNotIn("reasoning_effort", extra2)
        self.assertTrue(extra2.get("extra_body", {}).get("enable_thinking"))
        self.assertNotIn("thinking_budget", extra2.get("extra_body", {}))

        # glm-5.1 带 openai/ 前缀 (model_lake)
        adapter3 = get_thinking_adapter("openai/glm-5.1")
        self.assertIsInstance(adapter3, GLMThinkingAdapter)
        extra3 = adapter3.adapt("openai/glm-5.1", "high")
        self.assertIn("thinking", extra3.get("extra_body", {}))

    def test_registry_normalize_proxy_id(self):
        """模型清单 id 按网关类型归一化，展示用裸名"""
        sdk_id, label, provider = LiteLLMRegistry._normalize_proxy_id(
            "qwen3.7-flash", gateway_type="model_lake"
        )
        self.assertEqual(sdk_id, "openai/qwen3.7-flash")
        self.assertEqual(label, "qwen3.7-flash")
        self.assertEqual(provider, "qwen")

        sdk_id, label, provider = LiteLLMRegistry._normalize_proxy_id(
            "litellm_proxy/deepseek-v4-flash", gateway_type="litellm"
        )
        self.assertEqual(sdk_id, "litellm_proxy/deepseek-v4-flash")
        self.assertEqual(label, "deepseek-v4-flash")
        self.assertEqual(provider, "deepseek")

        models = LiteLLMRegistry._parse_models_response(
            {"data": [{"id": "qwen3.7-flash"}, {"id": "bge-m3-embedding"}]},
            gateway_type="model_lake",
        )
        ids = [m.id for m in models]
        self.assertIn("openai/qwen3.7-flash", ids)
        self.assertIn("openai/bge-m3-embedding", ids)

        visible = LiteLLMRegistry._filter_visible_models(
            models, ["qwen3.7-flash"]
        )
        self.assertEqual([m.id for m in visible], ["openai/qwen3.7-flash"])
        self.assertEqual(LiteLLMRegistry._filter_visible_models(models, []), [])

    def test_infer_provider_keyword_table(self):
        """_infer_provider 关键词表：老厂商保持原判定，新厂商正确归组，未知落兜底"""
        cases = {
            # 原有厂商（顺序敏感：deepseek 必须先于其他命中）
            "deepseek-v4-flash": "deepseek",
            "qwen3.7-flash": "qwen",
            "qwq-32b": "qwen",
            "glm-5.1": "glm",
            "chatglm3-6b": "glm",
            "zhipu-glm-5-2": "glm",
            "claude-sonnet-4": "anthropic",
            "gpt-4o-mini": "openai",
            "o3-mini": "openai",
            "gemini-2.5-pro": "gemini",
            "kimi-k2": "moonshot",
            "moonshot-v1-8k": "moonshot",
            "baichuan2-13b": "baichuan",
            "minimax-text-01": "minimax",
            "abab6.5s": "minimax",
            # 国内其他厂商
            "mimo-v2.5": "mimo",
            "mimo-v2.5-pro": "mimo",
            "ernie-4.5": "ernie",
            "hunyuan-turbo": "hunyuan",
            "doubao-pro-32k": "doubao",
            "internlm2-20b": "internlm",
            "yi-lightning": "yi",
            "step-2-16k": "step",
            "spark-max": "spark",
            "sensenova-v6": "sensenova",
            # 国际其他厂商
            "mistral-large": "mistral",
            "mixtral-8x7b": "mistral",
            "codestral-latest": "mistral",
            "llama-3.3-70b": "meta",
            "grok-3": "xai",
            "command-r-plus": "cohere",
            "phi-4": "microsoft",
            "nemotron-70b": "nvidia",
            "jamba-1.5": "ai21",
            "minicpm-v": "openbmb",
            # 未命中 → 兜底
            "small-32k": "litellm_proxy",
            "bge-m3-embedding": "litellm_proxy",
            "": "litellm_proxy",
        }
        for name, want in cases.items():
            self.assertEqual(
                LiteLLMRegistry._infer_provider(name), want,
                f"{name!r} 应归入 {want}",
            )
        # model_lake 兜底值
        self.assertEqual(
            LiteLLMRegistry._infer_provider("unknown-model", default="model_lake"),
            "model_lake",
        )

    def test_gateway_type_comes_from_env_only(self):
        """未设置 MODEL_GATEWAY_TYPE 时默认 litellm，不读 [proxy]"""
        cm = ConfigManager()
        with patch.dict(os.environ, {
            "LITELLM_PROXY_URL": "http://litellm.internal:4000",
            "LITELLM_PROXY_KEY": "sk-litellm-key",
        }, clear=False):
            for key in (
                "MODEL_GATEWAY_TYPE",
                "MODEL_LAKE_BASE",
                "MODEL_GATEWAY_TIMEOUT",
                "MODEL_GATEWAY_MAX_RETRIES",
            ):
                os.environ.pop(key, None)
            llm_cfg = cm.get_llm_gateway_full_config()
            self.assertEqual(llm_cfg["gateway_type"], "litellm")
            self.assertEqual(llm_cfg["api_base"], "http://litellm.internal:4000")
            self.assertEqual(llm_cfg["default_timeout"], 60)
            self.assertEqual(llm_cfg["default_max_retries"], 2)

    def test_model_lake_allowed_openai_params_with_reasoning_effort(self):
        """测试 ModelNexus 网关模式下，带思考强度参数构建请求不会被 LiteLLM 抛出 UnsupportedParamsError"""
        from src.client.llm.client import LLMClient, LLMClientConfig
        import litellm

        # 模拟 ModelNexus 客户端
        client = LLMClient(
            LLMClientConfig(
                model="openai/deepseek-official/deepseek-v4-flash",
                api_base="https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
                api_key="mn-sk-test-token",
            )
        )
        params = client._build_params(
            [{"role": "user", "content": "你好"}],
            reasoning_effort="high",
        )

        # 验证 allowed_openai_params 已注入
        self.assertIn("allowed_openai_params", params)
        self.assertIn("reasoning_effort", params["allowed_openai_params"])
        self.assertIn("thinking", params["allowed_openai_params"])
        self.assertEqual(params.get("drop_params"), False)
        self.assertEqual(params.get("reasoning_effort"), "high")

        # 验证 litellm 的 get_optional_params 处理此参数组合时不会抛出 UnsupportedParamsError
        optional = litellm.get_optional_params(
            model=params["model"],
            custom_llm_provider="openai",
            reasoning_effort=params["reasoning_effort"],
            drop_params=params["drop_params"],
            allowed_openai_params=params["allowed_openai_params"],
        )
        self.assertEqual(optional.get("reasoning_effort"), "high")

    def test_assistant_message_tool_calls_content_sanitization(self):
        """测试 assistant 消息带 tool_calls 且无正文时 content 被规范化为 None（JSON null）"""
        import json
        from src.client.llm.client import LLMClient, LLMClientConfig
        from src.client.llm.types import LLMResponse, ToolCall
        from src.service.chat.chat_service import _assistant_message
        from src.prompts.chat.context_builder import rebuild_messages_from_history
        from unittest.mock import MagicMock

        # 1. 测试 _assistant_message
        tc = ToolCall(id="call_123", name="test_tool", arguments={"q": "hello"})
        resp_no_text = LLMResponse(model="test-model", content="", tool_calls=[tc], finish_reason="tool_calls")
        msg1 = _assistant_message(resp_no_text)
        self.assertIsNone(msg1["content"])
        self.assertIn("tool_calls", msg1)
        self.assertIn('"content": null', json.dumps(msg1))

        resp_with_text = LLMResponse(model="test-model", content="Thinking result", tool_calls=[tc], finish_reason="tool_calls")
        msg2 = _assistant_message(resp_with_text)
        self.assertEqual(msg2["content"], "Thinking result")

        # 2. 测试 rebuild_messages_from_history
        fake_msg = MagicMock()
        fake_msg.role = "assistant"
        fake_msg.content = ""
        fake_msg.tool_calls = [tc]
        rebuilt = rebuild_messages_from_history([fake_msg])
        self.assertEqual(len(rebuilt), 1)
        self.assertIsNone(rebuilt[0]["content"])
        self.assertIn('"content": null', json.dumps(rebuilt[0]))

        # 3. 测试 client._build_params 兜底清洗
        client = LLMClient(
            LLMClientConfig(
                model="openai/deepseek-official/deepseek-v4-flash",
                api_base="https://modelnexus.jiepeix.com:20443/model-nexus/api/v1",
                api_key="mn-sk-test-token",
            )
        )
        params = client._build_params(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
            ]
        )
        msgs = params["messages"]
        self.assertEqual(msgs[0]["content"], "hello")
        self.assertIsNone(msgs[1]["content"])
        self.assertIn('"content": null', json.dumps(msgs[1]))


if __name__ == "__main__":
    unittest.main()
