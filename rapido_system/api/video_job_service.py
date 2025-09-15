#!/usr/bin/env python3
"""
Video Job Service: fetches video job data by ID from Creatium API and stores it locally.

Usage:
    await fetch_and_store_video_job(video_job_id: str) -> str
        - Returns the absolute path to the saved JSON file.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import aiohttp


CREATIUM_BASE_URL = (
    "https://creatium-node-e9dxh4hxgge6dsct.eastus-01.azurewebsites.net/api/v1"
)


def _get_slide_data_dir() -> Path:
    """Return the slideData directory path, ensuring it exists."""
    # This file lives at rapido_system/api/video_job_service.py
    api_dir = Path(__file__).resolve().parent
    rapido_system_dir = api_dir.parent
    data_dir = rapido_system_dir / "data" / "slideData"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _get_parsed_slide_data_dir() -> Path:
    """Return the parsed_slideData directory path, ensuring it exists."""
    api_dir = Path(__file__).resolve().parent
    rapido_system_dir = api_dir.parent
    data_dir = rapido_system_dir / "data" / "parsed_slideData"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


async def fetch_video_job(video_job_id: str, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
    """Fetch video job JSON from Creatium API.

    Args:
        video_job_id: The video job identifier.
        session: Optional shared aiohttp session.

    Returns:
        Parsed JSON dictionary.

    Raises:
        aiohttp.ClientResponseError: For HTTP errors
        aiohttp.ClientError: For request/connection errors
        ValueError: If response is not JSON
    """
    url = f"{CREATIUM_BASE_URL}/video-jobs/{video_job_id}"

    async def _request(sess: aiohttp.ClientSession) -> Dict[str, Any]:
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    if session is not None:
        return await _request(session)

    async with aiohttp.ClientSession() as local_session:
        return await _request(local_session)


def _extract_stringified_doc_json(raw: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the stringified document JSON from the fetched payload."""
    if not isinstance(raw, dict):
        return None
    if "stringifiedDocJson" in raw and isinstance(raw["stringifiedDocJson"], str):
        return raw["stringifiedDocJson"]
    data_obj = raw.get("data")
    if isinstance(data_obj, dict) and isinstance(data_obj.get("stringifiedDocJson"), str):
        return data_obj["stringifiedDocJson"]
    # Some APIs may nest it under payload/document
    payload_obj = raw.get("payload")
    if isinstance(payload_obj, dict) and isinstance(payload_obj.get("stringifiedDocJson"), str):
        return payload_obj["stringifiedDocJson"]
    document_obj = raw.get("document")
    if isinstance(document_obj, dict) and isinstance(document_obj.get("stringifiedDocJson"), str):
        return document_obj["stringifiedDocJson"]
    return None


def _parse_slides_from_doc(doc_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform slides to the required simplified schema."""
    slides_input = []
    if isinstance(doc_json, dict):
        maybe_slides = doc_json.get("slides")
        if isinstance(maybe_slides, list):
            slides_input = maybe_slides

    parsed_slides: List[Dict[str, Any]] = []
    for idx, slide in enumerate(slides_input, start=1):
        content_type = (slide or {}).get("contentType")
        slide_type = 1 if content_type == "canvas" else 2

        narration = (slide or {}).get("narrationData") or {}
        narration_text = narration.get("text") if slide_type == 1 else ""
        narration_duration = narration.get("totalDuration") if slide_type == 1 else ""

        presenter_config = slide.get("presenterConfig") if slide_type == 1 else ""

        # Prefer explicit embedId when present for non-canvas slides; fallback to slide id
        embed_id: Any = ""
        if slide_type == 2:
            embed_id = slide.get("embedId") or slide.get("id") or ""

        parsed_slides.append({
            "slideInfo": {
                "position": idx,
                "slideType": slide_type,
                "narrationData": narration_text if narration_text is not None else "",
                "narrationDuration": narration_duration if narration_duration is not None else "",
                "slideId": slide.get("id") if slide_type == 1 else "",
                "embedId": embed_id,
                "presenterConfig": presenter_config if presenter_config is not None else ""
            }
        })

    return parsed_slides


def _build_parsed_payload(doc_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "slides": _parse_slides_from_doc(doc_json)
    }


async def parse_and_store_parsed_slide_data(video_job_id: str, raw_payload: Dict[str, Any]) -> Path:
    """Parse the fetched payload and persist the simplified JSON to parsed_slideData.

    Returns the absolute path to the saved file.
    """
    stringified = _extract_stringified_doc_json(raw_payload)
    if not stringified:
        # If we cannot extract, write an empty structure for visibility
        doc_json: Dict[str, Any] = {}
    else:
        try:
            doc_json = json.loads(stringified)
        except Exception:
            doc_json = {}

    parsed_payload = _build_parsed_payload(doc_json)

    out_dir = _get_parsed_slide_data_dir()
    out_path = out_dir / f"{video_job_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(parsed_payload, f, ensure_ascii=False, indent=2)

    return out_path.resolve()


async def fetch_and_store_video_job(video_job_id: str) -> Path:
    """Fetch video job data and persist it under data/slideData/{video_job_id}.json.

    Args:
        video_job_id: The video job identifier.

    Returns:
        Absolute Path to the saved JSON file.
    """
    slide_dir = _get_slide_data_dir()
    target_path = slide_dir / f"{video_job_id}.json"

    data = await fetch_video_job(video_job_id)

    # Write pretty JSON for readability
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Immediately parse and store simplified slide data
    try:
        await parse_and_store_parsed_slide_data(video_job_id, data)
    except Exception:
        # Keep raw file even if parsing fails; caller can inspect logs
        pass

    return target_path.resolve()


