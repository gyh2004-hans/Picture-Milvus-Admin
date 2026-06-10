"""共享 LLM 调用工具 —— OpenAI 兼容 chat/completions API.

所有 LLM 调用通过此模块统一发出：prompt 分解、prompt 优化、VLM 视觉复核等。

提供:
  - chat_completion(): 纯文本 LLM 调用
  - vision_completion(): 视觉 LLM 调用（支持图片 + 文本）
  - retry_call(): 通用重试装饰器
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    VLM_VISION_MODEL,
)

logger = logging.getLogger(__name__)

# ── 摘要工具 ────────────────────────────────────────────


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── LLM 调用 ────────────────────────────────────────────


async def chat_completion(
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> str:
    """调用 OpenAI 兼容 chat/completions 接口，返回模型文本输出.

    Args:
        system: 系统 prompt.
        user: 用户消息.
        model: 模型 ID，默认使用 config.LLM_MODEL.
        temperature: 采样温度.
        max_tokens: 最大输出 token 数.
        timeout: HTTP 超时秒数.

    Returns:
        str: 模型输出的纯文本.

    Raises:
        RuntimeError: HTTP 错误或 API 返回错误时抛出.
        ValueError: API Key 未配置时抛出.
    """
    model_id = model or LLM_MODEL
    api_key = LLM_API_KEY

    if not api_key:
        raise ValueError("LLM API Key 未配置，请在 .env 中设置 LLM_API_KEY 或 ARK_API_KEY")

    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    input_digest = _digest(system + user)
    return await _make_openai_request(
        url, headers, body,
        model_id=model_id,
        input_digest=input_digest,
        timeout=timeout,
        label="llm",
    )


def parse_json_from_llm(raw: str, context: str = "") -> list[str]:
    """从 LLM 输出中提取 JSON 数组.

    兼容模型输出带 markdown 代码块或前后说明文字的情况.

    Args:
        raw: LLM 原始输出文本.
        context: 上下文描述（用于错误信息）.

    Returns:
        list[str]: 解析出的字符串列表.

    Raises:
        ValueError: 无法提取有效 JSON 数组时抛出.
    """
    text = raw.strip()

    # 尝试提取 ```json ... ``` 代码块
    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end > start:
            text = text[start + 3 : end].strip()
            # 去掉可能的 "json" 语言标记
            if text.startswith("json"):
                text = text[4:].strip()

    # 尝试找到 JSON 数组边界
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        text = text[bracket_start : bracket_end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("llm_utils.parse_json.error | context=%s error=%s raw_len=%d", context, e, len(raw))
        raise ValueError(f"LLM 输出无法解析为 JSON ({context}): {e}") from e

    if not isinstance(parsed, list):
        raise ValueError(f"LLM 输出不是 JSON 数组 ({context}): {type(parsed)}")

    # 确保所有元素为字符串
    result = [str(item) for item in parsed]
    return result


# ── 重试工具 ────────────────────────────────────────────

async def _retry_call(
    fn: Callable[[], Awaitable[str]],
    max_attempts: int = 3,
    retryable_errors: tuple = (httpx.TimeoutException, httpx.HTTPStatusError),
) -> str:
    """通用重试包装器（异步）.

    遇到可重试错误时指数退避重试，不可重试错误直接抛.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except retryable_errors as exc:
            last_exc = exc
            if attempt < max_attempts:
                wait_s = 2 ** (attempt - 1)  # 1s, 2s, 4s
                logger.warning(
                    "llm_utils.retry | attempt=%d/%d wait_s=%d error=%s",
                    attempt,
                    max_attempts,
                    wait_s,
                    exc,
                )
                await asyncio.sleep(wait_s)
            else:
                logger.error(
                    "llm_utils.retry.exhausted | attempts=%d error=%s",
                    max_attempts,
                    exc,
                )
        except Exception as exc:
            logger.error(
                "llm_utils.retry.non_retryable | error=%s",
                exc,
            )
            raise
    raise RuntimeError(f"LLM 调用失败（已重试 {max_attempts} 次）") from last_exc


async def _make_openai_request(
    url: str,
    headers: dict,
    body: dict,
    *,
    model_id: str,
    input_digest: str,
    timeout: float = 120.0,
    label: str = "llm",
) -> str:
    """统一的 OpenAI 兼容 API 请求 + 日志（异步）.

    Returns:
        str: 模型输出文本内容.
    """
    t0 = time.perf_counter()
    logger.info(
        "%s.call.start | provider=openai_compat model=%s "
        "sys_len=%d user_repr=%s temp=%.2f input_digest=%s",
        label,
        model_id,
        len(body.get("messages", [{}])[0].get("content", "")) if body.get("messages") else 0,
        _summarize_body(body),
        body.get("temperature", 0),
        input_digest,
    )

    # 显式分层超时：connect 不宜太长（服务不可达快速失败），
    # read 按参数 timeout 设置（大图/长文本需要更长时间）
    _connect_timeout = min(15.0, timeout * 0.3)
    _read_timeout = timeout
    _write_timeout = max(30.0, timeout * 0.5)
    client_timeout = httpx.Timeout(
        connect=_connect_timeout,
        read=_read_timeout,
        write=_write_timeout,
        pool=_connect_timeout,
    )

    try:
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "%s.call.error | model=%s status=%d duration_ms=%d body=%s "
            "error_kind=http_status retryable=%s",
            label,
            model_id,
            exc.response.status_code,
            duration_ms,
            exc.response.text[:500],
            "true" if exc.response.status_code >= 500 else "false",
        )
        raise
    except httpx.TimeoutException as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "%s.call.error | model=%s duration_ms=%d error_kind=timeout retryable=true",
            label,
            model_id,
            duration_ms,
        )
        raise

    duration_ms = int((time.perf_counter() - t0) * 1000)
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})

    logger.info(
        "%s.call.end | model=%s duration_ms=%d output_len=%d output_digest=%s "
        "usage_prompt=%s usage_completion=%s usage_total=%s",
        label,
        model_id,
        duration_ms,
        len(content),
        _digest(content),
        usage.get("prompt_tokens", "?"),
        usage.get("completion_tokens", "?"),
        usage.get("total_tokens", "?"),
    )
    return content


def _summarize_body(body: dict) -> str:
    """Summarize request body for logging (avoid full content)."""
    msgs = body.get("messages", [])
    total_len = sum(len(str(m.get("content", ""))) for m in msgs)
    return f"msgs={len(msgs)} total_chars={total_len}"


# ── Vision LLM 调用 ─────────────────────────────────────

async def vision_completion(
    *,
    system: str,
    user_text: str,
    image_b64: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> str:
    """调用视觉 LLM（OpenAI 兼容 vision API），返回模型文本输出.

    统一复用 llm_utils 的日志、错误处理、重试模式。
    VLM 视觉复核由 vlm_verifier 调用此函数。

    Args:
        system: 系统 prompt.
        user_text: 用户问题文本.
        image_b64: base64 data URI 编码的图片.
        model: 模型 ID，默认使用 config.VLM_VISION_MODEL.
        temperature: 采样温度.
        max_tokens: 最大输出 token 数.
        timeout: HTTP 超时秒数.
        max_retries: 最大重试次数（遇到 5xx / timeout 时重试）.

    Returns:
        str: 模型输出的纯文本.

    Raises:
        RuntimeError: HTTP 错误或 API 返回错误，已重试仍失败时抛出.
        ValueError: API Key 未配置时抛出.
    """
    model_id = model or VLM_VISION_MODEL
    api_key = LLM_API_KEY

    if not api_key:
        raise ValueError("VLM API Key 未配置，请在 .env 中设置 LLM_API_KEY 或 ARK_API_KEY")

    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    input_digest = _digest(system + user_text + image_b64[:50])

    async def _call() -> str:
        return await _make_openai_request(
            url, headers, body,
            model_id=model_id,
            input_digest=input_digest,
            timeout=timeout,
            label="llm",
        )

    return await _retry_call(_call, max_attempts=max_retries + 1)
