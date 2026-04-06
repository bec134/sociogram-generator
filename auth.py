# auth.py
"""
Authentication module — Google OAuth 2.0.

Any teacher with a Google account (DoE-managed Workspace or personal/
organisational) can sign in. No domain restriction is applied.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ENABLE GOOGLE AUTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Create an OAuth 2.0 Client ID in Google Cloud Console:
   - Type: Web application
   - Authorised redirect URI: <your-app-url>/  (Streamlit handles the callback
     at the root path via st.query_params)

2. Add the following to Streamlit secrets (never hard-code):
       [google_oauth]
       AUTH_ENABLED  = true
       CLIENT_ID     = "<OAuth 2.0 client ID>"
       CLIENT_SECRET = "<OAuth 2.0 client secret>"
       REDIRECT_URI  = "<your app's public URL — must match Google Console>"

3. For local development over HTTP, also set:
       OAUTHLIB_INSECURE_TRANSPORT = "1"
   (This module sets it automatically when REDIRECT_URI starts with http://)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User dict shape (preserved by all implementations):
    {
        "display_name": str,    # e.g. "Jane Smith"
        "email":        str,    # e.g. "jane.smith@det.nsw.edu.au"
        "roles":        list,   # always ["teacher"] — no role claims from Google
        "school_code":  str,    # always "" — not available from Google
    }
"""

import os
import secrets
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────────

def _cfg() -> dict:
    """Return the google_oauth secrets dict, or {} if not configured."""
    if hasattr(st, "secrets"):
        return st.secrets.get("google_oauth", {})
    return {}


_AUTH_ENABLED: bool = (
    _cfg().get("AUTH_ENABLED", os.environ.get("AUTH_ENABLED", "false")).lower() == "true"
)

_SESSION_KEY = "auth_user"
_OAUTH_STATE_KEY = "oauth_state"

_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def require_auth() -> dict:
    """
    Call at the top of the app, before any other UI code.

    Returns the authenticated user dict if the user is signed in.
    Otherwise renders the Google sign-in screen and halts (st.stop()).

    No-op gate when AUTH_ENABLED is false (dev mode).
    """
    if not _AUTH_ENABLED:
        return _dev_user()

    user = st.session_state.get(_SESSION_KEY)
    if user is None:
        _google_oauth_login_page()
        st.stop()

    return user


def get_current_user() -> dict | None:
    """Returns the current user dict, or None if not authenticated."""
    if not _AUTH_ENABLED:
        return _dev_user()
    return st.session_state.get(_SESSION_KEY)


def logout():
    """Clear the authenticated session and return to the login screen."""
    st.session_state.pop(_SESSION_KEY, None)
    st.session_state.pop(_OAUTH_STATE_KEY, None)
    st.rerun()


# ── Dev bypass (AUTH_ENABLED=false only) ───────────────────────────────────────

def _dev_user() -> dict:
    """Synthetic user for local development. Never reachable when AUTH_ENABLED=true."""
    return {
        "display_name": "Dev User",
        "email": "dev@det.nsw.edu.au",
        "roles": ["teacher"],
        "school_code": "0000",
    }


# ── Google OAuth 2.0 login flow ────────────────────────────────────────────────

def _google_oauth_login_page():
    """
    Handles the full Google OAuth 2.0 flow within Streamlit.

    First visit:  generates a CSRF state token, builds the Google auth URL,
                  and redirects the browser via a meta-refresh tag.
    Callback:     Google redirects back with ?code=...&state=...
                  We validate state, exchange the code for tokens, verify the
                  ID token, and store the user dict in session state.
    """
    from google_auth_oauthlib.flow import Flow
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests

    cfg = _cfg()
    if not cfg.get("CLIENT_ID") or not cfg.get("CLIENT_SECRET") or not cfg.get("REDIRECT_URI"):
        st.error(
            "Google OAuth is not configured. "
            "Set CLIENT_ID, CLIENT_SECRET, and REDIRECT_URI in Streamlit secrets."
        )
        st.stop()

    # Allow HTTP for local dev (oauthlib enforces HTTPS by default)
    if cfg["REDIRECT_URI"].startswith("http://"):
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    client_config = {
        "web": {
            "client_id": cfg["CLIENT_ID"],
            "client_secret": cfg["CLIENT_SECRET"],
            "redirect_uris": [cfg["REDIRECT_URI"]],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    params = st.query_params

    # ── Callback: Google has redirected back with a code ───────────────────────
    if "code" in params:
        # Validate CSRF state
        returned_state = params.get("state", "")
        expected_state = st.session_state.get(_OAUTH_STATE_KEY, "")
        if not expected_state or returned_state != expected_state:
            st.error("Login failed: invalid state parameter. Please try signing in again.")
            st.session_state.pop(_OAUTH_STATE_KEY, None)
            st.query_params.clear()
            st.stop()

        try:
            flow = Flow.from_client_config(
                client_config,
                scopes=_SCOPES,
                redirect_uri=cfg["REDIRECT_URI"],
                state=expected_state,
            )
            flow.fetch_token(code=params["code"])
            credentials = flow.credentials

            # Verify the ID token with Google's public keys
            id_info = google_id_token.verify_oauth2_token(
                credentials.id_token,
                google_requests.Request(),
                cfg["CLIENT_ID"],
            )
        except Exception as exc:
            st.error(f"Login failed: {exc}")
            st.session_state.pop(_OAUTH_STATE_KEY, None)
            st.query_params.clear()
            st.stop()

        st.session_state[_SESSION_KEY] = {
            "display_name": id_info.get("name", id_info.get("email", "Unknown")),
            "email": id_info.get("email", ""),
            "roles": ["teacher"],
            "school_code": "",
        }
        st.session_state.pop(_OAUTH_STATE_KEY, None)
        st.query_params.clear()
        st.rerun()

    # ── First visit: redirect to Google ───────────────────────────────────────
    state = secrets.token_urlsafe(32)
    st.session_state[_OAUTH_STATE_KEY] = state

    flow = Flow.from_client_config(
        client_config,
        scopes=_SCOPES,
        redirect_uri=cfg["REDIRECT_URI"],
    )
    auth_url, _ = flow.authorization_url(
        state=state,
        access_type="online",
        include_granted_scopes="true",
        prompt="select_account",
    )

    st.markdown(
        """
        <style>
        .login-wrap {
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; min-height: 60vh; gap: 1.5rem;
        }
        .login-title { font-size: 1.6rem; font-weight: 700; color: #002664; }
        .login-sub   { color: #555; font-size: 0.95rem; text-align: center; max-width: 380px; }
        .google-btn  {
            display: inline-flex; align-items: center; gap: 10px;
            background: #fff; border: 1px solid #dadce0; border-radius: 4px;
            padding: 10px 24px; font-size: 0.95rem; font-weight: 500;
            color: #3c4043; text-decoration: none; cursor: pointer;
            box-shadow: 0 1px 3px rgba(0,0,0,.12);
        }
        .google-btn:hover { background: #f8f8f8; box-shadow: 0 2px 6px rgba(0,0,0,.15); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="login-wrap">
          <div class="login-title">Sociogram Generator</div>
          <div class="login-sub">
            Sign in with your Google account to continue.<br>
            NSW DoE, other education departments, and personal Google accounts are all supported.
          </div>
          <a class="google-btn" href="{auth_url}">
            <svg width="18" height="18" viewBox="0 0 48 48">
              <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
              <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
              <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
              <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.31-8.16 2.31-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
              <path fill="none" d="M0 0h48v48H0z"/>
            </svg>
            Sign in with Google
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Azure AD skeleton (preserved for future DoE SSO integration) ───────────────
#
# def _azure_ad_login_page():
#     """
#     Redirects to the DoE Azure AD login page and handles the OAuth callback.
#
#     Prerequisites:
#       - msal installed (pip install msal)
#       - Streamlit secrets configured:
#           [azure_ad]
#           AUTH_ENABLED = true
#           CLIENT_ID    = "<app registration client ID>"
#           TENANT_ID    = "<DoE Azure AD tenant ID>"
#           CLIENT_SECRET = "<client secret>"
#           REDIRECT_URI  = "<your app's callback URL>"
#           AUTHORITY     = "https://login.microsoftonline.com/<TENANT_ID>"
#     """
#     import msal
#
#     cfg = st.secrets["azure_ad"]
#     app = msal.ConfidentialClientApplication(
#         client_id=cfg["CLIENT_ID"],
#         client_credential=cfg["CLIENT_SECRET"],
#         authority=cfg["AUTHORITY"],
#     )
#
#     params = st.query_params
#
#     if "code" in params:
#         result = app.acquire_token_by_authorization_code(
#             code=params["code"],
#             scopes=["openid", "profile", "email"],
#             redirect_uri=cfg["REDIRECT_URI"],
#         )
#         if "error" in result:
#             st.error(f"Login failed: {result.get('error_description', result['error'])}")
#             return
#
#         claims = result.get("id_token_claims", {})
#         st.session_state[_SESSION_KEY] = {
#             "display_name": claims.get("name", "Unknown"),
#             "email":        claims.get("preferred_username", ""),
#             "roles":        claims.get("roles", ["teacher"]),
#             "school_code":  claims.get("extension_SchoolCode", ""),
#         }
#         st.query_params.clear()
#         st.rerun()
#     else:
#         auth_url = app.get_authorization_request_url(
#             scopes=["openid", "profile", "email"],
#             redirect_uri=cfg["REDIRECT_URI"],
#         )
#         st.markdown(
#             f'<meta http-equiv="refresh" content="0; url={auth_url}">',
#             unsafe_allow_html=True,
#         )
#         st.info("Redirecting to NSW DoE sign-in...")
#         st.stop()
