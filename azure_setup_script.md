# Azure Setup Script for Rapido Proxy

## Prerequisites
1. Azure account with active subscription
2. GitHub repository with the proxy files
3. Custom domain (optional but recommended)

## Step 1: Create Azure Resources

### Using Azure CLI
```bash
# Login to Azure
az login

# Set variables (customize these)
RESOURCE_GROUP="rapido-proxy-rg"
APP_NAME="rapido-api-proxy"  # Must be globally unique
LOCATION="eastus"
SKU="B1"  # Basic tier

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service plan (Linux)
az appservice plan create \
  --name "${APP_NAME}-plan" \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku $SKU \
  --is-linux

# Create Web App with Python 3.12 runtime
az webapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --plan "${APP_NAME}-plan" \
  --runtime "PYTHON:3.12"

# Set startup command
az webapp config set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --startup-file "python azure_proxy_wrapper.py"

# Configure app settings
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    AWS_RAPIDO_BASE_URL="http://54.82.5.74:8080" \
    PORT="8000" \
    PYTHONUNBUFFERED="1"

echo "✅ Azure Web App created: https://${APP_NAME}.azurewebsites.net"
```

## Step 2: Set up GitHub Actions Secrets

In your GitHub repository, go to Settings > Secrets and variables > Actions, and add:

```
AZURE_CLIENT_ID=your-service-principal-client-id
AZURE_TENANT_ID=your-azure-tenant-id  
AZURE_SUBSCRIPTION_ID=your-azure-subscription-id
```

### Create Service Principal for GitHub Actions
```bash
# Create service principal (replace with your subscription ID)
SUBSCRIPTION_ID="your-subscription-id"
SP_NAME="rapido-github-actions"

az ad sp create-for-rbac \
  --name $SP_NAME \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID \
  --json-auth

# Copy the output values to GitHub Secrets:
# clientId -> AZURE_CLIENT_ID
# tenantId -> AZURE_TENANT_ID  
# subscriptionId -> AZURE_SUBSCRIPTION_ID
```

## Step 3: Update Workflow File

In `.github/workflows/deploy-azure-proxy.yml`, update:
- `app-name: 'rapido-api-proxy'` → Your actual app name
- Branches to deploy from

## Step 4: Custom Domain (Optional)

### Add Custom Domain
```bash
# Add custom domain (replace with your domain)
DOMAIN="rapido-api.yourdomain.com"

az webapp config hostname add \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --hostname $DOMAIN

# Enable HTTPS
az webapp config ssl bind \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --certificate-thumbprint auto \
  --ssl-type SNI \
  --hostname $DOMAIN
```

### DNS Configuration
Create CNAME record in your DNS provider:
```
rapido-api.yourdomain.com → rapido-api-proxy.azurewebsites.net
```

## Step 5: Test Deployment

1. Push code to your repository
2. GitHub Actions will automatically deploy
3. Test endpoints:
   - Health: `https://your-app.azurewebsites.net/health`
   - API: `https://your-app.azurewebsites.net/api/v1/livekit/token`

## Step 6: Monitor

- **Logs**: Azure Portal > App Service > Log stream
- **Metrics**: Azure Portal > App Service > Metrics
- **Health**: Built-in health endpoint at `/health`

## Troubleshooting

### Common Issues:
1. **502 Bad Gateway**: Check if AWS server (54.82.5.74:8080) is running
2. **CORS Errors**: Verify proxy is running and configured correctly
3. **Build Failures**: Check requirements.txt and Python version

### Useful Commands:
```bash
# View app logs
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP

# Restart app
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

# Update app settings
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings AWS_RAPIDO_BASE_URL="http://new-ip:8080"
```

## Final Architecture

```
Frontend (https://test.creatium.com)
    ↓ HTTPS requests
Azure App Service (https://rapido-api.yourdomain.com)
    ↓ HTTP proxy
AWS EC2 Server (http://54.82.5.74:8080)
    ↑ HTTP response
Azure App Service (adds CORS headers)
    ↑ HTTPS response  
Frontend (receives response)
```

**Benefits:**
- ✅ Stable HTTPS URL that never changes
- ✅ Automatic SSL certificates from Azure
- ✅ Professional domain name
- ✅ Auto-deployment from GitHub
- ✅ Built-in monitoring and logging
- ✅ No tunnel complexity
