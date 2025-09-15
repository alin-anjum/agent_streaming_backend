# Azure Proxy Wrapper Deployment Guide

## Architecture
```
Frontend (test.creatium.com) 
    ↓ HTTPS
Azure App Service (your-domain.azurewebsites.net or CNAME)
    ↓ HTTP  
AWS EC2 Server (54.82.5.74:8080)
```

**NO TUNNELS! Direct connection to AWS server.**

## Step 1: Prepare Files for Azure
Copy these 3 files to your Azure deployment:
- `azure_proxy_wrapper.py` 
- `azure_requirements.txt`
- This deployment guide

## Step 2: Deploy to Azure App Service

### Option A: Azure Portal
1. Create **App Service** (Python 3.11)
2. Upload files via **Advanced Tools > Kudu**
3. Set startup command: `python azure_proxy_wrapper.py`

### Option B: Azure CLI
```bash
# Login to Azure
az login

# Create resource group
az group create --name rapido-proxy --location eastus

# Create app service plan
az appservice plan create --name rapido-plan --resource-group rapido-proxy --sku B1 --is-linux

# Create web app
az webapp create --name rapido-api-proxy --resource-group rapido-proxy --plan rapido-plan --runtime "PYTHON:3.11"

# Deploy code
az webapp deploy --name rapido-api-proxy --resource-group rapido-proxy --src-path ./ --type zip

# Set startup command
az webapp config set --name rapido-api-proxy --resource-group rapido-proxy --startup-file "python azure_proxy_wrapper.py"
```

## Step 3: Configure Environment Variable
In Azure App Service > Configuration > Application Settings:
```
AWS_RAPIDO_BASE_URL = http://54.82.5.74:8080
```

**Static IP - NEVER CHANGES!**

## Step 4: Set Up Custom Domain (CNAME)
1. Azure App Service > Custom domains
2. Add domain: `rapido-api.yourdomain.com`
3. Create CNAME record in your DNS:
   ```
   rapido-api.yourdomain.com → your-app.azurewebsites.net
   ```
4. Enable SSL certificate (free with Azure)

## Step 5: Update Frontend
Change frontend API base URL to:
```
https://rapido-api.yourdomain.com
```

## Benefits
✅ **Stable URL** - Never changes  
✅ **Professional SSL** - Azure managed certificates
✅ **CORS Handled** - Proxy handles all CORS issues
✅ **Scalable** - Azure auto-scaling
✅ **Health Monitoring** - Built-in health checks
✅ **Easy Updates** - Just update AWS_RAPIDO_BASE_URL when serveo URL changes

## Monitoring
- Health endpoint: `https://rapido-api.yourdomain.com/health`
- Azure Application Insights for monitoring
- Logs available in Azure portal

## Benefits of Direct Connection
✅ **No Tunnels** - Direct HTTP connection to AWS
✅ **Never Changes** - Static IP address
✅ **Simple** - No complex tunnel management  
✅ **Fast** - One less network hop
✅ **Reliable** - No tunnel disconnections

## No URL Changes Ever!
The AWS server IP (54.82.5.74:8080) is static - nothing to update!
