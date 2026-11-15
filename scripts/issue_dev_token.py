"""Make an HS256 access token for local work with `AUTH_ENABLED=true` and `AUTH_JWT_SECRET`.

    PYTHONPATH=src python scripts/issue_dev_token.py --sub dev --scope "query threads:read"

The token uses the configured issuer and audience. Production uses tokens from an identity
provider (JWKS), so this script refuses to run with `APP_ENV=production`.
"""

import argparse
import sys
import time

import jwt

from insightflow.core.config import get_settings, secret_text
from insightflow.security.auth import ALL_SCOPES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sub", required=True, help="The token subject (a user identifier).")
    parser.add_argument(
        "--scope",
        default="query threads:read threads:delete",
        help=f"Space-separated scopes from: {' '.join(sorted(ALL_SCOPES))}.",
    )
    parser.add_argument("--minutes", type=int, default=60, help="Lifetime of the token.")
    args = parser.parse_args()

    settings = get_settings()
    if settings.is_production:
        print("refused: APP_ENV=production uses identity-provider tokens", file=sys.stderr)
        return 1
    secret = secret_text(settings.auth_jwt_secret)
    if secret is None:
        print("AUTH_JWT_SECRET is not set", file=sys.stderr)
        return 1
    unknown = set(args.scope.split()) - ALL_SCOPES
    if unknown:
        print(f"unknown scopes: {' '.join(sorted(unknown))}", file=sys.stderr)
        return 1
    now = int(time.time())
    claims = {"sub": args.sub, "scope": args.scope, "iat": now, "exp": now + args.minutes * 60}
    if settings.auth_jwt_issuer:
        claims["iss"] = settings.auth_jwt_issuer
    if settings.auth_jwt_audience:
        claims["aud"] = settings.auth_jwt_audience
    algorithm = settings.jwt_algorithms[0] if settings.jwt_algorithms else "HS256"
    print(jwt.encode(claims, secret, algorithm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
