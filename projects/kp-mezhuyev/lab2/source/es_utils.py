"""
Утилиты для подключения к Elasticsearch с автоопределением HTTPS/HTTP.
"""

from typing import Any, Tuple

from elasticsearch import Elasticsearch
from elasticsearch import AuthenticationException, AuthorizationException


def _build_es_params(
    url: str,
    es_config: dict[str, Any],
    use_ssl: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "hosts": [url],
        "request_timeout": es_config.get("request_timeout", 10),
    }

    if use_ssl:
        params["verify_certs"] = es_config.get("verify_certs", False)
        ca_certs = es_config.get("ca_certs")
        if ca_certs:
            params["ca_certs"] = ca_certs
        if not params["verify_certs"]:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Auth
    if es_config.get("cloud_id"):
        params["cloud_id"] = es_config["cloud_id"]
    if es_config.get("api_key"):
        params["api_key"] = es_config["api_key"]
    if es_config.get("username") and es_config.get("password"):
        params["basic_auth"] = (es_config["username"], es_config["password"])

    return params


def get_es_client(es_config: dict[str, Any]) -> Tuple[Elasticsearch, str]:
    """Создает Elasticsearch клиент с автоопределением протокола.

    Returns:
        (client, url)
    """
    host = es_config.get("host", "localhost")
    port = es_config.get("port", 9200)
    use_ssl = es_config.get("use_ssl", None)  # None = auto

    if use_ssl is True:
        protocols = ["https"]
    elif use_ssl is False:
        protocols = ["http"]
    else:
        protocols = ["https", "http"]

    last_error: Exception | None = None

    for scheme in protocols:
        url = f"{scheme}://{host}:{port}"
        use_https = scheme == "https"
        params = _build_es_params(url, es_config, use_https)

        try:
            client = Elasticsearch(**params)
            # Проверяем доступ
            client.info()
            return client, url
        except (AuthenticationException, AuthorizationException) as e:
            raise ConnectionError(
                "Elasticsearch требует аутентификацию. "
                "Укажите username/password или api_key в config.yaml."
            ) from e
        except Exception as e:
            last_error = e
            continue

    raise ConnectionError(
        f"Cannot connect to Elasticsearch. Last error: {last_error}"
    )
{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}