"""LiteLLM provider profile.

LiteLLM proxy exposes model metadata at ``/v1/model/info`` instead of the
standard ``/v1/models``.  This plugin overrides ``fetch_models`` and context
length resolution to use the LiteLLM-specific endpoint.

Response format::

    {"data": [{"model_name": "gpt-4", "model_info": {"max_input_tokens": 8192, "max_tokens": 4096}}]}
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)


class LiteLLMProfile(ProviderProfile):
    """LiteLLM proxy — model info from ``/v1/model/info``."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch model list from LiteLLM's ``/v1/model/info`` endpoint."""
        models_url = getattr(self, "models_url", "")
        if not models_url and self.base_url:
            models_url = self.base_url.rstrip("/") + "/v1/model/info"
        if not models_url:
            return None

        req = urllib.request.Request(models_url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        for k, v in self.default_headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["model_name"] for m in items if isinstance(m, dict) and "model_name" in m]
        except Exception as exc:
            logger.debug("fetch_models(litellm): %s", exc)
            return None


litellm = LiteLLMProfile(
    name="litellm",
    aliases=(),
    display_name="LiteLLM",
    description="LiteLLM proxy — unified OpenAI-compatible gateway for 100+ LLMs",
    signup_url="",
    env_vars=("LITELLM_API_KEY",),
    base_url="",
    hostname="",
    auth_type="api_key",
    default_aux_model="",
    fallback_models=(),
)

register_provider(litellm)
