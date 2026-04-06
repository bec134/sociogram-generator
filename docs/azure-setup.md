# Azure Deployment Setup Guide

This document covers the one-time infrastructure setup required to deploy
Sociogram Generator to Azure App Service (Australia East).

Hand this document to your Azure administrator or DoE ICT contact.

---

## Prerequisites

- An Azure subscription (DoE ICT can provide this under the DoE Azure tenancy)
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- Owner or Contributor role on the target subscription

---

## 1. Log in to Azure

```bash
az login
az account set --subscription "<your subscription ID>"
```

---

## 2. Create a resource group in Australia East

All resources must be in `australiaeast` to satisfy NSW DoE data residency requirements.

```bash
az group create \
  --name sociogram-rg \
  --location australiaeast
```

---

## 3. Create an Azure Container Registry (ACR)

The container registry stores the built Docker images.

```bash
az acr create \
  --resource-group sociogram-rg \
  --name sociogramacr \
  --sku Basic \
  --location australiaeast \
  --admin-enabled true
```

Note the login server — it will be `sociogramacr.azurecr.io`.

Get the admin credentials:

```bash
az acr credential show --name sociogramacr
```

Save the `username` and one of the `passwords` — you'll need them for GitHub secrets.

---

## 4. Create an App Service plan and web app

```bash
# App Service plan (Linux, Basic tier — ~$15 AUD/month)
az appservice plan create \
  --name sociogram-plan \
  --resource-group sociogram-rg \
  --is-linux \
  --sku B1 \
  --location australiaeast

# Web app (container-based)
az webapp create \
  --name sociogram-generator \
  --resource-group sociogram-rg \
  --plan sociogram-plan \
  --deployment-container-image-name sociogramacr.azurecr.io/sociogram-generator:latest
```

Configure the app to pull from ACR:

```bash
az webapp config container set \
  --name sociogram-generator \
  --resource-group sociogram-rg \
  --docker-custom-image-name sociogramacr.azurecr.io/sociogram-generator:latest \
  --docker-registry-server-url https://sociogramacr.azurecr.io \
  --docker-registry-server-user sociogramacr \
  --docker-registry-server-password "<acr admin password>"
```

Set the port (Streamlit runs on 8501):

```bash
az webapp config appsettings set \
  --name sociogram-generator \
  --resource-group sociogram-rg \
  --settings WEBSITES_PORT=8501
```

---

## 5. Add Streamlit app secrets

These are the runtime secrets your app reads from `st.secrets`. Set them as
App Service environment variables (they are encrypted at rest):

```bash
az webapp config appsettings set \
  --name sociogram-generator \
  --resource-group sociogram-rg \
  --settings \
    AUTH_ENABLED=true \
    GOOGLE_OAUTH_CLIENT_ID="<your Google OAuth client ID>" \
    GOOGLE_OAUTH_CLIENT_SECRET="<your Google OAuth client secret>" \
    GOOGLE_OAUTH_REDIRECT_URI="https://sociogram-generator.azurewebsites.net/"
```

> **Note:** Streamlit reads secrets from `.streamlit/secrets.toml` or from
> environment variables. For Azure, environment variables are the right approach
> — never commit secrets to the repository.

---

## 6. Create a GitHub Actions service principal

This allows GitHub Actions to deploy to Azure on your behalf.

```bash
az ad sp create-for-rbac \
  --name sociogram-github-actions \
  --role contributor \
  --scopes /subscriptions/<subscription-id>/resourceGroups/sociogram-rg \
  --sdk-auth
```

Copy the entire JSON output — you'll need it in the next step.

---

## 7. Add GitHub repository secrets

In the GitHub repository, go to **Settings → Secrets and variables → Actions**
and add the following secrets:

| Secret name | Value |
|---|---|
| `AZURE_CREDENTIALS` | The full JSON from step 6 |
| `REGISTRY_LOGIN_SERVER` | `sociogramacr.azurecr.io` |
| `REGISTRY_USERNAME` | ACR admin username from step 3 |
| `REGISTRY_PASSWORD` | ACR admin password from step 3 |
| `AZURE_WEBAPP_NAME` | `sociogram-generator` |

---

## 8. Trigger the first deployment

Push to `main` (or merge the feature branch into `main`). GitHub Actions will:

1. Build the Docker image
2. Push it to ACR
3. Deploy it to App Service

The app will be live at:
```
https://sociogram-generator.azurewebsites.net
```

You can also trigger a deploy manually from the **Actions** tab in GitHub.

---

## 9. Add the redirect URI to Google Cloud Console

Once the app is live, go to your Google Cloud Console → APIs & Services →
Credentials → your OAuth client, and add:

```
https://sociogram-generator.azurewebsites.net/
```

as an authorised redirect URI.

---

## Ongoing costs (approximate AUD/month)

| Resource | Tier | Cost |
|---|---|---|
| App Service (B1) | Basic | ~$22 |
| Container Registry | Basic | ~$6 |
| **Total** | | **~$28/month** |

Costs can be reduced by scaling down to Free/Shared tiers during development,
or by stopping the App Service when not in use.

---

## Troubleshooting

**App won't start:** Check App Service logs via Azure Portal → App Service →
Log stream, or run `az webapp log tail --name sociogram-generator --resource-group sociogram-rg`.

**Image pull failure:** Confirm the ACR credentials are correctly set in the
App Service container configuration (step 4).

**Auth redirect mismatch:** Ensure the `REDIRECT_URI` environment variable
exactly matches the URI registered in Google Cloud Console, including the
trailing slash.
