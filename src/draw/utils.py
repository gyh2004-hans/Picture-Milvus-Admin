"""Draw 模块共享工具 —— 图片保存、API 响应解析."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# DashScope 文生图异步任务轮询配置
_DASHSCOPE_POLL_INTERVAL = 2.0   # 秒
_DASHSCOPE_POLL_TIMEOUT = 180.0  # 秒


def prompt_digest(prompt: str) -> str:
    """prompt 指纹，用于日志回溯."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def save_image_bytes(image_bytes: bytes, directory: Path, prefix: str) -> Path:
    """将图片字节写入 storage/images/ 并返回路径."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.png"
    filepath = directory / filename
    filepath.write_bytes(image_bytes)
    return filepath


async def decode_image_from_response(data: dict) -> bytes:
    """从 OpenAI 兼容 images/generations 响应中提取图片字节."""
    items = data.get("data") or []
    if not items:
        raise ValueError("API 响应缺少 data 字段")

    item = items[0]
    if "b64_json" in item and item["b64_json"]:
        return base64.b64decode(item["b64_json"])

    url = item.get("url")
    if not url:
        raise ValueError("API 响应既无 b64_json 也无 url")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def _dashscope_endpoint(model: str) -> str:
    """根据模型名判断应使用哪个 DashScope 端点."""
    if model.startswith("qwen-image"):
        return "multimodal"
    return "text2image"


def _dashscope_extract_image_url(response_data: dict, endpoint: str) -> str | None:
    """从 DashScope 响应中提取图片下载 URL."""
    output = response_data.get("output", {})

    if endpoint == "multimodal":
        # qwen-image: output.choices[0].message.content[0].image
        choices = output.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", [])
            if content:
                return content[0].get("image")
        return None

    # text2image: output.results[0].url
    results = output.get("results", [])
    if results:
        return results[0].get("url")
    return None


async def _download_image(image_url: str, t0: float) -> bytes:
    """下载图片并返回字节."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            img_resp = await client.get(image_url)
            img_resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "draw.api.error | provider=tongyi phase=download "
            "status=%d duration_ms=%d",
            exc.response.status_code,
            duration_ms,
        )
        raise RuntimeError(
            f"tongyi 生图结果下载失败 (HTTP {exc.response.status_code})"
        ) from exc
    return img_resp.content


async def post_dashscope_image_generation(
    *,
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    timeout: float = 180.0,
) -> bytes:
    """DashScope 原生文生图接口.

    自动根据模型名选择端点:
      - qwen-image-* → multimodal-generation (messages 格式)
      - wanx* 等      → text2image/image-synthesis (异步提交 + 轮询)

    DashScope 兼容模式 (compatible-mode) 不支持 images/generations 端点.
    """
    if not api_key:
        raise ValueError("DashScope API Key 未配置，请在 .env 中设置 DASHSCOPE_API_KEY")

    size_normalized = size.replace("x", "*")
    endpoint = _dashscope_endpoint(model)
    base_url = "https://dashscope.aliyuncs.com"
    t0 = time.perf_counter()

    # ── 构建请求 ──
    if endpoint == "multimodal":
        submit_url = f"{base_url}/api/v1/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: dict = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
            "parameters": {
                "size": size_normalized,
                "n": 1,
                "watermark": False,
            },
        }
        is_async = False
    else:
        submit_url = f"{base_url}/api/v1/services/aigc/text2image/image-synthesis"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        body = {
            "model": model,
            "input": {"prompt": prompt},
            "parameters": {
                "size": size_normalized,
                "n": 1,
            },
        }
        is_async = True

    logger.info(
        "draw.api.start | provider=tongyi model=%s endpoint=%s prompt_len=%d "
        "prompt_digest=%s async=%s",
        model,
        endpoint,
        len(prompt),
        prompt_digest(prompt),
        is_async,
    )

    # ── 提交请求 ──
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(submit_url, json=body, headers=headers)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "draw.api.error | provider=tongyi phase=submit endpoint=%s "
            "status=%d duration_ms=%d body=%s",
            endpoint,
            exc.response.status_code,
            duration_ms,
            exc.response.text[:500],
        )
        raise RuntimeError(
            f"tongyi 生图任务提交失败 (HTTP {exc.response.status_code}): "
            f"{exc.response.text[:200]}"
        ) from exc

    resp_data = resp.json()

    # ── 同步模式 (multimodal): 直接提取图片 URL ──
    if not is_async:
        image_url = _dashscope_extract_image_url(resp_data, endpoint)
        if not image_url:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "draw.api.error | provider=tongyi endpoint=%s phase=no_url "
                "duration_ms=%d body=%s",
                endpoint,
                duration_ms,
                resp.text[:500],
            )
            raise RuntimeError(
                f"tongyi 生图响应未包含图片 URL: {resp.text[:200]}"
            )

        image_bytes = await _download_image(image_url, t0)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "draw.api.end | provider=tongyi endpoint=%s duration_ms=%d "
            "image_bytes=%d",
            endpoint,
            duration_ms,
            len(image_bytes),
        )
        return image_bytes

    # ── 异步模式 (text2image): 轮询任务 → 获取 URL → 下载 ──
    task_id = resp_data.get("output", {}).get("task_id")
    if not task_id:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "draw.api.error | provider=tongyi endpoint=%s phase=submit "
            "duration_ms=%d error=missing_task_id body=%s",
            endpoint,
            duration_ms,
            resp.text[:500],
        )
        raise RuntimeError(f"tongyi 生图未返回 task_id: {resp.text[:200]}")

    logger.info(
        "draw.api.task_submitted | provider=tongyi task_id=%s model=%s",
        task_id,
        model,
    )

    task_url = f"{base_url}/api/v1/tasks/{task_id}"
    deadline = time.perf_counter() + timeout

    while True:
        await asyncio.sleep(_DASHSCOPE_POLL_INTERVAL)

        if time.perf_counter() > deadline:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "draw.api.error | provider=tongyi phase=poll task_id=%s "
                "duration_ms=%d error=timeout",
                task_id,
                duration_ms,
            )
            raise TimeoutError(
                f"tongyi 生图任务 {task_id} 轮询超时 ({timeout}s)"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                poll_resp = await client.get(task_url, headers=headers)
                poll_resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "draw.api.error | provider=tongyi phase=poll task_id=%s "
                "status=%d duration_ms=%d body=%s",
                task_id,
                exc.response.status_code,
                duration_ms,
                exc.response.text[:500],
            )
            raise RuntimeError(
                f"tongyi 生图任务查询失败 (HTTP {exc.response.status_code})"
            ) from exc

        poll_data = poll_resp.json()
        task_status = poll_data.get("output", {}).get("task_status", "")

        if task_status == "SUCCEEDED":
            break
        if task_status == "FAILED":
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "draw.api.error | provider=tongyi phase=task_failed task_id=%s "
                "duration_ms=%d body=%s",
                task_id,
                duration_ms,
                poll_resp.text[:500],
            )
            raise RuntimeError(
                f"tongyi 生图任务 {task_id} 失败: {poll_resp.text[:200]}"
            )

        logger.debug(
            "draw.api.polling | provider=tongyi task_id=%s status=%s",
            task_id,
            task_status,
        )

    image_url = _dashscope_extract_image_url(poll_data, endpoint)
    if not image_url:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "draw.api.error | provider=tongyi phase=no_url task_id=%s "
            "duration_ms=%d body=%s",
            task_id,
            duration_ms,
            poll_resp.text[:500],
        )
        raise RuntimeError(f"tongyi 生图任务 {task_id} 返回结果缺少图片 URL")

    image_bytes = await _download_image(image_url, t0)

    duration_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "draw.api.end | provider=tongyi endpoint=%s duration_ms=%d "
        "image_bytes=%d task_id=%s",
        endpoint,
        duration_ms,
        len(image_bytes),
        task_id,
    )
    return image_bytes


async def post_image_generation(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    provider: str,
    timeout: float = 120.0,
) -> bytes:
    """调用 OpenAI 兼容文生图接口."""
    if not api_key:
        raise ValueError(f"{provider} API Key 未配置，请在 .env 中设置对应密钥")

    url = f"{base_url.rstrip('/')}/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
        "n": 1,
        "watermark": False,
    }

    t0 = time.perf_counter()
    logger.info(
        "draw.api.start | provider=%s model=%s prompt_len=%d prompt_digest=%s",
        provider,
        model,
        len(prompt),
        prompt_digest(prompt),
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            image_bytes = await decode_image_from_response(resp.json())
    except httpx.HTTPStatusError as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "draw.api.error | provider=%s status=%d duration_ms=%d body=%s",
            provider,
            exc.response.status_code,
            duration_ms,
            exc.response.text[:500],
        )
        raise RuntimeError(
            f"{provider} 生图失败 (HTTP {exc.response.status_code}): "
            f"{exc.response.text[:200]}"
        ) from exc
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "draw.api.error | provider=%s duration_ms=%d error=%s",
            provider,
            duration_ms,
            exc,
        )
        raise

    duration_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "draw.api.end | provider=%s duration_ms=%d image_bytes=%d",
        provider,
        duration_ms,
        len(image_bytes),
    )
    return image_bytes
