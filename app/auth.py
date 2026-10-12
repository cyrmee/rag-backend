"""Bearer-token authentication against Keycloak.

The backend never sees passwords: the frontend logs users in at Keycloak
and sends the resulting access token with every request. This module
checks the token's signature against the realm's published keys, its
expiry, issuer and audience, and turns it into a User. Account management
(password, profile, sessions) stays in Keycloak's own account console."""

import asyncio
import time
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.db import get_connection

_bearer = HTTPBearer(auto_error=False)
_jwks = jwt.PyJWKClient(
    settings.keycloak_jwks_url or f"{settings.keycloak_issuer}/protocol/openid-connect/certs",
    cache_keys=True,
    lifespan=3600,
)

# Users seen recently - their row is only rewritten every few minutes, not
# on every request.
_USER_REFRESH_SECONDS = 300
_last_upsert: dict[str, float] = {}


@dataclass(frozen=True)
class User:
    id: str  # Keycloak's "sub" - stable across username/email changes
    username: str
    email: str | None
    name: str | None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def _decode(token: str) -> dict:
    # Blocking on a key-cache miss (an HTTP fetch of the realm's JWKS);
    # callers run it in a thread.
    signing_key = _jwks.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256", "RS384", "RS512", "PS256", "ES256"],
        audience=settings.keycloak_audience,
        issuer=settings.keycloak_issuer,
        options={"require": ["exp", "iat", "sub"]},
        leeway=30,
    )


async def _remember_user(user: User) -> None:
    now = time.monotonic()
    if now - _last_upsert.get(user.id, -_USER_REFRESH_SECONDS) < _USER_REFRESH_SECONDS:
        return
    async with get_connection() as conn:
        await conn.execute(
            """
            insert into users (id, username, email, name, last_seen_at)
            values (%s, %s, %s, %s, now())
            on conflict (id) do update set
                username = excluded.username, email = excluded.email,
                name = excluded.name, last_seen_at = now()
            """,
            (user.id, user.username, user.email, user.name),
        )
        await conn.commit()
    _last_upsert[user.id] = now


async def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User:
    """FastAPI dependency: the signed-in user, or 401."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Not signed in")
    try:
        claims = await asyncio.to_thread(_decode, credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Session expired - sign in again")
    except jwt.PyJWKClientConnectionError:
        raise HTTPException(status_code=503, detail="The sign-in server (Keycloak) is unreachable")
    except jwt.PyJWTError:
        raise _unauthorized("Invalid access token")

    user = User(
        id=claims["sub"],
        username=claims.get("preferred_username") or claims["sub"],
        email=claims.get("email"),
        name=claims.get("name"),
    )
    await _remember_user(user)
    return user
