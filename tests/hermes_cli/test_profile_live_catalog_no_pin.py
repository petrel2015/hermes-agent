"""A transient live-fetch failure must never be pinned into the provider models disk cache.

Contract (matches the cache's own header comment, ``only NON-EMPTY results are cached
so a transient failure is never pinned``):

* When a provider's live ``/models`` fetch fails, ``provider_model_ids`` still returns
  a usable list (curated static list / models.dev merge / profile fallback_models — the
  picker never goes blank), BUT
* ``cached_provider_model_ids`` and every other cache writer must not persist that
  degraded result: the disk cache row is what later picker opens serve (1h TTL + up to
  7d stale-serve), so a pinned fallback list makes new subscription models (e.g.
  glm-5.3 on a z.ai coding plan) disappear from ``/model`` long after the network blip
  healed.
"""

import json
from unittest.mock import MagicMock, patch

from hermes_cli.models import cached_provider_model_ids
from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel


def _profile_with(models=None, fallback=("glm-old-1", "glm-old-2"), base_url="https://api.example.com/v1"):
    p = MagicMock()
    p.auth_type = "api_key"
    p.base_url = base_url
    p.fetch_models.return_value = models  # None simulates timeout/HTTP failure
    p.fallback_models = fallback
    return p


def _no_cache_row(tmp_path):
    cache_file = tmp_path / "cache.json"
    return not cache_file.exists() or "zai" not in json.loads(cache_file.read_text())


class TestLiveFetchFailureNotPinned:
    """A failed live fetch serves a degraded list but never writes the disk cache."""

    def test_failed_fetch_does_not_write_cache_row(self, tmp_path):
        """Live fetch fails -> the picker gets the profile fallback floor, and the cache
        file gets NO row for the provider."""
        with (
            patch("providers.get_provider_profile", return_value=_profile_with(models=None)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch("hermes_cli.models._PROVIDER_MODELS", {"zai": []}),
            patch("hermes_cli.models._MODELS_DEV_PREFERRED", frozenset()),
            patch("hermes_cli.models._provider_models_cache_path", return_value=tmp_path / "cache.json"),
        ):
            result = cached_provider_model_ids("zai")

        # The caller still gets a usable list this open (profile fallback_models floor).
        assert result == ["glm-old-1", "glm-old-2"]
        # But nothing was persisted: a transient failure is never pinned.
        assert _no_cache_row(tmp_path)

    def test_failed_fetch_falls_back_to_curated_before_profile_models(self, tmp_path):
        """With a curated static list, a failed live fetch prefers curated (#46309
        curated-first intent) over the profile's stale fallback_models; the disk cache
        is still not written."""
        curated = ["glm-5.3", "glm-5.3-flash", "glm-5.2"]
        with (
            patch("providers.get_provider_profile", return_value=_profile_with(models=None)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch("hermes_cli.models._PROVIDER_MODELS", {"zai": curated}),
            patch("hermes_cli.models._MODELS_DEV_PREFERRED", frozenset()),
            patch("hermes_cli.models._provider_models_cache_path", return_value=tmp_path / "cache.json"),
        ):
            result = cached_provider_model_ids("zai")

        assert result[: len(curated)] == curated
        assert _no_cache_row(tmp_path)

    def test_degraded_stale_row_beats_static_floor(self, tmp_path):
        """A failed live fetch must not shadow a REAL earlier live result already on
        disk (stale-serve during outage), and must not overwrite it either."""
        import time

        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps({
            "zai": {"fp": "fp", "at": time.time() - 3600, "models": ["glm-5.3", "glm-5.3-flash"]},
        }))
        with (
            patch("providers.get_provider_profile", return_value=_profile_with(models=None)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch("hermes_cli.models._PROVIDER_MODELS", {"zai": []}),
            patch("hermes_cli.models._MODELS_DEV_PREFERRED", frozenset()),
            patch("hermes_cli.models._provider_models_cache_path", return_value=cache_file),
            patch("hermes_cli.models._credential_fingerprint", return_value="fp"),
        ):
            result = cached_provider_model_ids("zai", force_refresh=True)

        assert result == ["glm-5.3", "glm-5.3-flash"]  # stale real row beats the floor
        assert json.loads(cache_file.read_text())["zai"]["models"] == ["glm-5.3", "glm-5.3-flash"]

    def test_successful_fetch_still_writes_cache(self, tmp_path):
        """Live fetch succeeds -> cache row IS written (regression guard for the fix
        over-correcting into never caching)."""
        with (
            patch("providers.get_provider_profile", return_value=_profile_with(models=["glm-5.3"])),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch("hermes_cli.models._PROVIDER_MODELS", {"zai": ["glm-5.2"]}),
            patch("hermes_cli.models._provider_models_cache_path", return_value=tmp_path / "cache.json"),
        ):
            cached_provider_model_ids("zai")

        data = json.loads((tmp_path / "cache.json").read_text())
        assert "zai" in data
        assert set(data["zai"]["models"]) >= {"glm-5.3", "glm-5.2"}

    def test_prefetch_parallel_path_does_not_pin_failure(self, tmp_path):
        """The parallel prefetch writer (update_provider_cache_entry) must not re-pin a
        degraded list either: a failed fetch returns the floor, but no row is persisted."""
        with (
            patch("providers.get_provider_profile", return_value=_profile_with(models=None)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch("hermes_cli.models._PROVIDER_MODELS", {"zai": []}),
            patch("hermes_cli.models._MODELS_DEV_PREFERRED", frozenset()),
            patch("hermes_cli.models._provider_models_cache_path", return_value=tmp_path / "cache.json"),
            patch("hermes_cli.models._credential_fingerprint", return_value="fp"),
        ):
            _prefetch_provider_models_parallel(["zai"])

        assert _no_cache_row(tmp_path)
