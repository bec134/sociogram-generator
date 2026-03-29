# auth.py
"""
Authentication module — stub implementation.

INTEGRATION POINT: Replace this stub with DoE Azure AD SSO (OAuth2 / OIDC).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ENABLE REAL AUTH (Azure AD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Register the app in the DoE Azure AD tenant (via DoE ICT or your Azure admin).
   Set the Redirect URI to: <your-app-url>/oauth2/callback

2. Add the msal package:
       pip install msal

3. Set the following in Streamlit secrets (never hard-code):
       [azure_ad]
       AUTH_ENABLED      = true
       CLIENT_ID         = "<app registration client ID>"
       TENANT_ID         = "<DoE Azure AD tenant ID>"
       CLIENT_SECRET     = "<client secret>"
       REDIRECT_URI      = "<your app's callback URL>"
       AUTHORITY         = "https://login.microsoftonline.com/<TENANT_ID>"

4. Replace _stub_login_page() with _azure_ad_login_page() — see the skeleton
   provided at the bottom of this file.

5. Replace _extract_user_from_session() with real MSAL token validation.

Reference: https://learn.microsoft.com/en-us/azure/active-directory/develop/
           msal-overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User dict shape (must be preserved by any real implementation):
    {
        "display_name": str,    # e.g. "Jane Smith"
        "email":        str,    # e.g. "jane.smith@det.nsw.edu.au"
        "roles":        list,   # e.g. ["teacher"] or ["admin"]
        "school_code":  str,    # e.g. "8765" — from Azure AD group/claim
    }
"""

import os
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────────────

# Set AUTH_ENABLED=true in your environment or Streamlit secrets to activate
# the real auth flow. Defaults to False so local dev works without credentials.
_AUTH_ENABLED: bool = (
    st.secrets.get("azure_ad", {}).get("AUTH_ENABLED", "false").lower() == "true"
    if hasattr(st, "secrets")
    else os.environ.get("AUTH_ENABLED", "false").lower() == "true"
)

_SESSION_KEY = "auth_user"


# ── Public API ─────────────────────────────────────────────────────────────────

def require_auth() -> dict:
    """
    Call at the top of the app, before any other UI code.

    Returns the authenticated user dict if the user is signed in.
    Otherwise renders the login screen and halts execution (st.stop()).

    This is a no-op gate when AUTH_ENABLED is false (dev mode).
    """
    if not _AUTH_ENABLED:
        return _dev_user()

    user = st.session_state.get(_SESSION_KEY)
    if user is None:
        _stub_login_page()
        st.stop()

    return user


def get_current_user() -> dict | None:
    """
    Returns the current user dict, or None if not authenticated.
    Useful for conditional UI (e.g. showing admin controls).
    """
    if not _AUTH_ENABLED:
        return _dev_user()
    return st.session_state.get(_SESSION_KEY)


def logout():
    """
    Clear the authenticated session and return to the login screen.

    INTEGRATION POINT: Also call msal app.remove_account() here to invalidate
    the Azure AD token cache when real auth is enabled.
    """
    st.session_state.pop(_SESSION_KEY, None)
    st.rerun()


# ── Dev bypass (AUTH_ENABLED=false only) ───────────────────────────────────────

def _dev_user() -> dict:
    """
    Returns a synthetic user for local development.
    NEVER reachable when AUTH_ENABLED=true.
    """
    return {
        "display_name": "Dev User",
        "email": "dev@det.nsw.edu.au",
        "roles": ["teacher"],
        "school_code": "0000",
    }


# ── Stub login UI ──────────────────────────────────────────────────────────────

def _stub_login_page():
    """
    Placeholder login screen shown when AUTH_ENABLED=true but no real
    OAuth flow is wired up yet.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  REPLACE THIS ENTIRE FUNCTION with _azure_ad_login_page()       ║
    ║  (skeleton below) once the Azure AD app registration is ready.  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    st.set_page_config(page_title="Sign in — Sociogram Generator", layout="centered")
    st.title("Sign in")
    st.info(
        "This tool is restricted to NSW Department of Education staff. "
        "Please sign in with your DoE account to continue."
    )

    st.warning(
        "**Auth stub active.** "
        "Azure AD integration is not yet configured. "
        "Click below to simulate a successful DoE login for testing."
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Sign in with NSW DoE (stub)", use_container_width=True):
            st.session_state[_SESSION_KEY] = {
                "display_name": "Test Teacher",
                "email": "test.teacher@det.nsw.edu.au",
                "roles": ["teacher"],
                "school_code": "1234",
            }
            st.rerun()


# ── Azure AD skeleton (implement when ready) ───────────────────────────────────
#
# def _azure_ad_login_page():
#     """
#     Redirects to the DoE Azure AD login page and handles the OAuth callback.
#
#     Prerequisites:
#       - msal installed (pip install msal)
#       - Streamlit secrets configured (see top of file)
#
#     Streamlit note: st.query_params is used to detect the OAuth callback code.
#     For production, consider a reverse-proxy callback handler instead.
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
#         # ── Exchange auth code for token ───────────────────────────────
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
#         # Clear the code from the URL
#         st.query_params.clear()
#         st.rerun()
#     else:
#         # ── Redirect to Azure AD login ─────────────────────────────────
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
