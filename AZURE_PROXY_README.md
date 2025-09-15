# Rapido Azure Proxy 🚀

A lightweight Azure-hosted proxy that provides stable HTTPS endpoints for your AWS Rapido API server.

## Architecture

```
Frontend (HTTPS) → Azure App Service (Proxy) → AWS EC2 (Rapido API)
```

## ⚡ Quick Deploy

### Option 1: One-Click Deploy to Azure

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fyour-username%2Fyour-repo%2Fmain%2Fazuredeploy.json)

### Option 2: GitHub Actions Auto-Deploy

1. Fork this repository
2. Set up GitHub secrets (see `azure_setup_script.md`)
3. Push to trigger auto-deployment

### Option 3: Manual Azure CLI

```bash
# Clone and deploy
git clone your-repo
cd your-repo
az group create --name rapido-proxy --location eastus
az deployment group create \
  --resource-group rapido-proxy \
  --template-file azuredeploy.json \
  --parameters azuredeploy.parameters.json
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `azure_proxy_wrapper.py` | Main proxy application |
| `azure_requirements.txt` | Python dependencies |
| `.github/workflows/deploy-azure-proxy.yml` | GitHub Actions workflow |
| `azuredeploy.json` | Azure ARM template |
| `azure_setup_script.md` | Detailed setup instructions |

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_RAPIDO_BASE_URL` | `http://54.82.5.74:8080` | Your AWS Rapido API server |
| `PORT` | `8000` | Azure App Service port |
| `PYTHONUNBUFFERED` | `1` | Python logging |

### Custom Domain Setup

1. **Azure Portal** → App Service → Custom domains
2. **Add domain**: `rapido-api.yourdomain.com`
3. **DNS CNAME**: `rapido-api.yourdomain.com` → `your-app.azurewebsites.net`
4. **Enable SSL**: Free Azure-managed certificate

## 🚦 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (proxy + target) |
| `/api/v1/livekit/token` | POST | LiveKit token generation |
| `/api/v1/presentations/start` | POST | Start presentation |
| `/api/v1/sessions/{id}/status` | GET | Session status |

## 🔍 Monitoring

### Health Check
```bash
curl https://your-app.azurewebsites.net/health
```

### Logs
- **Azure Portal** → App Service → Log stream
- **Azure CLI**: `az webapp log tail --name your-app --resource-group your-rg`

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| 502 Bad Gateway | Check if AWS server is running |
| CORS errors | Verify proxy configuration |
| Build failures | Check Python version and requirements |

### Test CORS
```bash
curl -X OPTIONS https://your-app.azurewebsites.net/api/v1/livekit/token \
  -H "Origin: https://your-frontend.com" \
  -H "Access-Control-Request-Method: POST"
```

## 🎯 Benefits

✅ **Stable URL** - Never changes  
✅ **Professional SSL** - Azure-managed certificates  
✅ **CORS Handled** - Built-in CORS support  
✅ **Auto-scaling** - Azure handles traffic spikes  
✅ **Monitoring** - Built-in logs and metrics  
✅ **No Tunnels** - Direct HTTP connection to AWS  

## 📊 Cost Estimation

| Tier | Monthly Cost | Features |
|------|--------------|----------|
| Free (F1) | $0 | 1GB storage, 60 mins/day |
| Basic (B1) | ~$13 | 1.75GB RAM, always on |
| Standard (S1) | ~$56 | Auto-scaling, custom domains |

## 🔄 Updates

When your AWS server IP changes, just update the environment variable:

```bash
az webapp config appsettings set \
  --name your-app \
  --resource-group your-rg \
  --settings AWS_RAPIDO_BASE_URL="http://new-ip:8080"
```

## 📞 Support

- **Issues**: Create GitHub issue
- **Docs**: See `azure_setup_script.md`
- **Azure Docs**: [App Service Python](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python)
