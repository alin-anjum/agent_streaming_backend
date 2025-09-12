# Rapido Production Deployment Guide

## Overview

This guide covers deploying the refactored Rapido system in a production environment with proper security, monitoring, and scalability considerations.

## 🏗️ Infrastructure Requirements

### Minimum System Requirements
- **CPU**: 4+ cores (8+ recommended for concurrent processing)
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 50GB+ SSD (100GB+ for production)
- **Network**: High-bandwidth connection for streaming
- **OS**: Ubuntu 20.04+ LTS or similar Linux distribution

### Recommended Production Setup
```bash
# Server specifications
CPU: 8 cores (Intel Xeon or AMD EPYC)
Memory: 32GB RAM
Storage: 200GB NVMe SSD
Network: 1Gbps dedicated connection
OS: Ubuntu 22.04 LTS
```

## 🔧 Pre-deployment Setup

### 1. System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    pkg-config \
    nginx \
    supervisor \
    redis-server \
    postgresql \
    git

# Create rapido user
sudo useradd -m -s /bin/bash rapido
sudo usermod -aG sudo rapido
```

### 2. Application Setup
```bash
# Switch to rapido user
sudo su - rapido

# Create application directory
mkdir -p /opt/rapido
cd /opt/rapido

# Clone repository
git clone <repository-url> .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Directory Structure
```bash
# Create required directories
sudo mkdir -p /var/log/rapido
sudo mkdir -p /var/lib/rapido
sudo mkdir -p /opt/rapido/data/{presentation_frames,lessons,output}
sudo mkdir -p /etc/rapido

# Set ownership
sudo chown -R rapido:rapido /var/log/rapido
sudo chown -R rapido:rapido /var/lib/rapido
sudo chown -R rapido:rapido /opt/rapido
```

## 🔐 Security Configuration

### 1. SSL/TLS Certificates
```bash
# Install certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot certonly --nginx -d your-domain.com

# Certificate will be stored in /etc/letsencrypt/live/your-domain.com/
```

### 2. Environment Variables
```bash
# Create production environment file
sudo tee /etc/rapido/production.env << EOF
# Security
JWT_SECRET=$(openssl rand -base64 32)
RAPIDO_ENV=production

# API Keys (replace with your actual keys)
ELEVENLABS_API_KEY=your_elevenlabs_api_key
LIVEKIT_URL=your_livekit_server_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Database
DATABASE_URL=postgresql://rapido:password@localhost/rapido

# Services
SYNCTALK_SERVER_URL=http://35.172.212.10:8000
REDIS_URL=redis://localhost:6379

# Paths
RAPIDO_LOG_DIR=/var/log/rapido
SLIDE_FRAMES_PATH=/opt/rapido/data/presentation_frames
INPUT_DATA_PATH=/opt/rapido/data/lessons
OUTPUT_PATH=/opt/rapido/data/output

# Performance
RAPIDO_WORKERS=4
RAPIDO_MAX_CONCURRENT_LESSONS=3
EOF

# Secure the environment file
sudo chmod 600 /etc/rapido/production.env
sudo chown rapido:rapido /etc/rapido/production.env
```

### 3. Production Configuration
```bash
# Copy production config
sudo cp /opt/rapido/config/production.json /etc/rapido/config.json

# Edit configuration for your environment
sudo vim /etc/rapido/config.json
```

## 🚀 Application Deployment

### 1. Systemd Service Configuration
```bash
# Create systemd service file
sudo tee /etc/systemd/system/rapido-api.service << EOF
[Unit]
Description=Rapido Avatar Presentation API
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=rapido
Group=rapido
WorkingDirectory=/opt/rapido
Environment=RAPIDO_CONFIG=/etc/rapido/config.json
EnvironmentFile=/etc/rapido/production.env
ExecStart=/opt/rapido/venv/bin/python src/rapido_api_refactored.py --host 127.0.0.1 --port 8080
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/rapido /var/lib/rapido /opt/rapido/data

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl enable rapido-api
sudo systemctl start rapido-api

# Check status
sudo systemctl status rapido-api
```

### 2. Nginx Configuration
```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/rapido << EOF
upstream rapido_backend {
    server 127.0.0.1:8080;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Client upload limits
    client_max_body_size 100M;
    
    # Timeouts for long-running requests
    proxy_connect_timeout 600;
    proxy_send_timeout 600;
    proxy_read_timeout 600;

    location / {
        proxy_pass http://rapido_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://rapido_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://rapido_backend;
        access_log off;
    }
}
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/rapido /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 📊 Monitoring Setup

### 1. Log Management with Logrotate
```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/rapido << EOF
/var/log/rapido/*.jsonl {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 rapido rapido
    postrotate
        systemctl reload rapido-api
    endscript
}
EOF
```

### 2. Monitoring with Supervisor (Alternative to systemd)
```bash
# Install supervisor if preferred over systemd
sudo apt install supervisor

# Create supervisor configuration
sudo tee /etc/supervisor/conf.d/rapido.conf << EOF
[program:rapido-api]
command=/opt/rapido/venv/bin/python src/rapido_api_refactored.py --host 127.0.0.1 --port 8080
directory=/opt/rapido
user=rapido
environment=RAPIDO_CONFIG="/etc/rapido/config.json"
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/rapido/supervisor.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
EOF

# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start rapido-api
```

### 3. Health Checks
```bash
# Create health check script
sudo tee /opt/rapido/scripts/health_check.sh << EOF
#!/bin/bash
set -e

# Check API health
response=\$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
if [ \$response -ne 200 ]; then
    echo "Health check failed with status \$response"
    exit 1
fi

# Check log file exists and is recent
log_file="/var/log/rapido/rapido_\$(date +%Y-%m-%d).jsonl"
if [ ! -f "\$log_file" ]; then
    echo "Log file not found: \$log_file"
    exit 1
fi

# Check if log file was modified in the last 5 minutes
if [ \$(find "\$log_file" -mmin -5 | wc -l) -eq 0 ]; then
    echo "Log file not updated recently: \$log_file"
    exit 1
fi

echo "Health check passed"
EOF

chmod +x /opt/rapido/scripts/health_check.sh

# Add to cron for regular checks
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/rapido/scripts/health_check.sh >> /var/log/rapido/health_check.log 2>&1") | crontab -
```

## 🔄 Database Setup (Optional)

### 1. PostgreSQL Configuration
```bash
# Configure PostgreSQL
sudo -u postgres createuser rapido
sudo -u postgres createdb rapido -O rapido

# Set password
sudo -u postgres psql -c "ALTER USER rapido PASSWORD 'secure_password';"

# Create tables (if using database for session management)
sudo -u postgres psql -d rapido -f /opt/rapido/sql/schema.sql
```

### 2. Redis Configuration
```bash
# Configure Redis
sudo vim /etc/redis/redis.conf
# Update settings:
# bind 127.0.0.1
# requirepass your_redis_password
# maxmemory 2gb
# maxmemory-policy allkeys-lru

sudo systemctl restart redis-server
```

## 🚦 Load Balancing (Multi-instance Setup)

### 1. Multiple Instance Configuration
```bash
# Create multiple service instances
for i in {1..4}; do
    sudo cp /etc/systemd/system/rapido-api.service /etc/systemd/system/rapido-api-$i.service
    sudo sed -i "s/port 8080/port 808$i/" /etc/systemd/system/rapido-api-$i.service
done

# Start all instances
for i in {1..4}; do
    sudo systemctl enable rapido-api-$i
    sudo systemctl start rapido-api-$i
done
```

### 2. Updated Nginx Configuration for Load Balancing
```bash
# Update upstream configuration
sudo tee -a /etc/nginx/sites-available/rapido << EOF
upstream rapido_backend {
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
    server 127.0.0.1:8083;
    server 127.0.0.1:8084;
}
EOF
```

## 📦 Backup and Recovery

### 1. Backup Scripts
```bash
# Create backup script
sudo tee /opt/rapido/scripts/backup.sh << EOF
#!/bin/bash
BACKUP_DIR="/var/backups/rapido"
DATE=\$(date +%Y%m%d_%H%M%S)

mkdir -p \$BACKUP_DIR

# Backup configuration
tar -czf \$BACKUP_DIR/config_\$DATE.tar.gz /etc/rapido/

# Backup data
tar -czf \$BACKUP_DIR/data_\$DATE.tar.gz /opt/rapido/data/

# Backup logs (last 7 days)
find /var/log/rapido -name "*.jsonl" -mtime -7 | tar -czf \$BACKUP_DIR/logs_\$DATE.tar.gz -T -

# Clean old backups (keep 30 days)
find \$BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: \$DATE"
EOF

chmod +x /opt/rapido/scripts/backup.sh

# Schedule daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/rapido/scripts/backup.sh >> /var/log/rapido/backup.log 2>&1") | crontab -
```

### 2. Recovery Procedures
```bash
# Create recovery script
sudo tee /opt/rapido/scripts/recover.sh << EOF
#!/bin/bash
BACKUP_FILE=\$1

if [ -z "\$BACKUP_FILE" ]; then
    echo "Usage: \$0 <backup_file.tar.gz>"
    exit 1
fi

echo "Stopping services..."
sudo systemctl stop rapido-api

echo "Restoring from \$BACKUP_FILE..."
tar -xzf \$BACKUP_FILE -C /

echo "Starting services..."
sudo systemctl start rapido-api

echo "Recovery completed"
EOF

chmod +x /opt/rapido/scripts/recover.sh
```

## 🔧 Performance Tuning

### 1. System Optimizations
```bash
# Create system optimization script
sudo tee /etc/sysctl.d/99-rapido.conf << EOF
# Network optimizations
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 65535

# Memory optimizations
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File system optimizations
fs.file-max = 65535
EOF

# Apply optimizations
sudo sysctl -p /etc/sysctl.d/99-rapido.conf
```

### 2. Application Performance
```bash
# Create performance monitoring script
sudo tee /opt/rapido/scripts/performance_monitor.sh << EOF
#!/bin/bash
LOG_FILE="/var/log/rapido/performance_\$(date +%Y-%m-%d).log"

while true; do
    echo "\$(date): System Performance Check" >> \$LOG_FILE
    
    # CPU usage
    echo "CPU: \$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}' | cut -d'%' -f1)" >> \$LOG_FILE
    
    # Memory usage
    echo "Memory: \$(free -m | awk 'NR==2{printf "%.1f%%", \$3*100/\$2}')" >> \$LOG_FILE
    
    # Disk usage
    echo "Disk: \$(df -h /opt/rapido | awk 'NR==2{print \$5}')" >> \$LOG_FILE
    
    # Active connections
    echo "Connections: \$(ss -tuln | wc -l)" >> \$LOG_FILE
    
    echo "---" >> \$LOG_FILE
    
    sleep 300  # 5 minutes
done
EOF

chmod +x /opt/rapido/scripts/performance_monitor.sh

# Run as background service
nohup /opt/rapido/scripts/performance_monitor.sh &
```

## 🚨 Troubleshooting

### 1. Common Issues

#### Service Won't Start
```bash
# Check service status
sudo systemctl status rapido-api

# Check logs
sudo journalctl -u rapido-api -f

# Check configuration
python -c "import json; json.load(open('/etc/rapido/config.json'))"
```

#### High Memory Usage
```bash
# Monitor memory usage
htop
sudo journalctl -u rapido-api | grep -i memory

# Restart service if needed
sudo systemctl restart rapido-api
```

#### Network Connection Issues
```bash
# Check port binding
sudo netstat -tlnp | grep 8080

# Test API connectivity
curl -v http://localhost:8080/health

# Check nginx logs
sudo tail -f /var/log/nginx/error.log
```

### 2. Maintenance Commands
```bash
# Graceful restart
sudo systemctl reload rapido-api

# View real-time logs
sudo journalctl -u rapido-api -f

# Check disk space
df -h /var/log/rapido
df -h /opt/rapido

# Clear old logs manually
find /var/log/rapido -name "*.jsonl" -mtime +7 -delete
```

## 🔄 Updates and Upgrades

### 1. Application Updates
```bash
# Create update script
sudo tee /opt/rapido/scripts/update.sh << EOF
#!/bin/bash
set -e

echo "Starting Rapido update..."

# Backup current version
/opt/rapido/scripts/backup.sh

# Stop services
sudo systemctl stop rapido-api

# Update code
cd /opt/rapido
git pull origin main

# Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Run database migrations if needed
# python scripts/migrate.py

# Start services
sudo systemctl start rapido-api

# Health check
sleep 10
/opt/rapido/scripts/health_check.sh

echo "Update completed successfully"
EOF

chmod +x /opt/rapido/scripts/update.sh
```

### 2. Zero-downtime Deployment
```bash
# For load-balanced setups
sudo tee /opt/rapido/scripts/rolling_update.sh << EOF
#!/bin/bash
set -e

INSTANCES=(1 2 3 4)

for instance in "\${INSTANCES[@]}"; do
    echo "Updating instance \$instance..."
    
    # Stop instance
    sudo systemctl stop rapido-api-\$instance
    
    # Update would happen here (code is shared)
    
    # Start instance
    sudo systemctl start rapido-api-\$instance
    
    # Health check
    sleep 5
    curl -f http://localhost:808\$instance/health
    
    echo "Instance \$instance updated successfully"
done

echo "Rolling update completed"
EOF

chmod +x /opt/rapido/scripts/rolling_update.sh
```

This deployment guide provides comprehensive instructions for setting up Rapido in a production environment with proper security, monitoring, and scalability considerations.
