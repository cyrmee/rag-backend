"""Signs the HTTP check scripts in as a test user of the docker compose
Keycloak (password grant on the rag-frontend client - enabled in the test
realm only). Use as the `auth` of an httpx client:

    async with httpx.AsyncClient(base_url=..., auth=KeycloakAuth()) as client:

It fetches a token on the first request and a fresh one whenever the
current one is about to expire, so long runs don't hit a 401 halfway.
Defaults to alice/alice at http://localhost:8180; override with
KEYCLOAK_URL, KEYCLOAK_REALM, KEYCLOAK_CLIENT_ID, RAG_TEST_USER and
RAG_TEST_PASSWORD."""

import os
import time

import httpx


class KeycloakAuth(httpx.Auth):
    def __init__(self, username: str | None = None, password: str | None = None) -> None:
        base = os.environ.get("KEYCLOAK_URL", "http://localhost:8180").rstrip("/")
        realm = os.environ.get("KEYCLOAK_REALM", "rag")
        self.token_url = f"{base}/realms/{realm}/protocol/openid-connect/token"
        self.client_id = os.environ.get("KEYCLOAK_CLIENT_ID", "rag-frontend")
        self.username = username or os.environ.get("RAG_TEST_USER", "alice")
        self.password = password or os.environ.get("RAG_TEST_PASSWORD", self.username)
        self._token: str | None = None
        self._expires_at = 0.0

    def _token_request(self) -> httpx.Request:
        return httpx.Request("POST", self.token_url, data={
            "grant_type": "password",
            "client_id": self.client_id,
            "username": self.username,
            "password": self.password,
        })

    def _store(self, response: httpx.Response) -> None:
        if response.status_code != 200:
            raise RuntimeError(f"Keycloak login as {self.username} failed: {response.status_code} {response.text}")
        body = response.json()
        self._token = body["access_token"]
        self._expires_at = time.monotonic() + body["expires_in"] - 30

    def auth_flow(self, request: httpx.Request):
        if self._token is None or time.monotonic() >= self._expires_at:
            response = yield self._token_request()
            response.read()
            self._store(response)
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        if self._token is None or time.monotonic() >= self._expires_at:
            response = yield self._token_request()
            await response.aread()
            self._store(response)
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request
