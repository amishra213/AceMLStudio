# 🐳 Docker Deployment Guide for AceML Studio

## Rancher Desktop Deployment

This guide will help you deploy AceML Studio on **Rancher Desktop**, a lightweight Kubernetes and container management platform.

---

## 📋 Prerequisites

1. **Rancher Desktop** installed and running
   - Download from: https://rancherdesktop.io/
   - Configure to use **dockerd (moby)** as the container runtime
   - Recommended: 4GB+ RAM allocated

2. **Basic knowledge** of Docker and Docker Compose

---

## 🚀 Quick Start

### Step 1: Clone or Navigate to Project Directory

```bash
cd D:/Projects/AceMLStudio
```

### Step 2: Configure Environment Variables

Create a `.env` file from the example:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your credentials
notepad .env
```

**Important**: Set at least your LLM API key:
- For OpenAI: Set `OPENAI_API_KEY`
- For DeepSeek: Set `DEEPSEEK_API_KEY`
- For Anthropic: Set `ANTHROPIC_API_KEY`

### Step 3: Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 4: Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🛠️ Docker Commands Reference

### Building

```bash
# Build the image
docker-compose build

# Build without cache (clean build)
docker-compose build --no-cache

# Build specific service
docker-compose build aceml-studio
```

### Running

```bash
# Start in detached mode
docker-compose up -d

# Start and view logs
docker-compose up

# Start with build
docker-compose up -d --build
```

### Managing

```bash
# Stop the application
docker-compose down

# Stop and remove volumes (clears data)
docker-compose down -v

# Restart the application
docker-compose restart

# View running containers
docker-compose ps

# View logs
docker-compose logs -f aceml-studio
```

### Maintenance

```bash
# Execute commands in running container
docker-compose exec aceml-studio bash

# View container resource usage
docker stats aceml-studio

# Inspect container
docker inspect aceml-studio
```

---

## 📦 Volume Management

AceML Studio uses Docker volumes for persistent data:

| Volume | Purpose | Path in Container |
|--------|---------|-------------------|
| `aceml_uploads` | Uploaded datasets | `/app/uploads` |
| `aceml_experiments` | Experiment tracking | `/app/experiments` |
| `aceml_logs` | Application logs | `/app/logs` |
| `aceml_data` | Sample data files | `/app/Data` |

### Backup Volumes

```bash
# Create backup directory
mkdir -p backups

# Backup uploads volume
docker run --rm -v aceml_uploads:/data -v ${PWD}/backups:/backup alpine tar czf /backup/uploads-backup.tar.gz -C /data .

# Backup experiments volume
docker run --rm -v aceml_experiments:/data -v ${PWD}/backups:/backup alpine tar czf /backup/experiments-backup.tar.gz -C /data .
```

### Restore Volumes

```bash
# Restore uploads
docker run --rm -v aceml_uploads:/data -v ${PWD}/backups:/backup alpine sh -c "cd /data && tar xzf /backup/uploads-backup.tar.gz"

# Restore experiments
docker run --rm -v aceml_experiments:/data -v ${PWD}/backups:/backup alpine sh -c "cd /data && tar xzf /backup/experiments-backup.tar.gz"
```

---

## 🔧 Configuration Options

### Using Environment Variables (Recommended)

Edit `.env` file and restart:

```bash
docker-compose down
# Edit .env file
docker-compose up -d
```

### Using config.properties File

Alternatively, mount a config file:

1. Create `config.properties` from `config.properties.example`
2. Uncomment the volume mount in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./config.properties:/app/config.properties:ro
   ```
3. Restart the container

---

## 🌐 Rancher Desktop Specific Tips

### Port Forwarding

By default, the app runs on `localhost:5000`. To access from other machines:

1. Change the port mapping in `docker-compose.yml`:
   ```yaml
   ports:
     - "0.0.0.0:5000:5000"  # Accessible from network
   ```

2. Restart the container

### Resource Limits

Add resource limits in `docker-compose.yml`:

```yaml
services:
  aceml-studio:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Using Rancher Desktop Kubernetes

To deploy on Kubernetes instead of Docker Compose:

```bash
# Create Kubernetes deployment (coming soon)
kubectl apply -f kubernetes/
```

---

## 🔍 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs aceml-studio

# Check container status
docker-compose ps

# Inspect detailed error
docker inspect aceml-studio
```

### Permission Issues

```bash
# Fix volume permissions
docker-compose exec aceml-studio chown -R aceml:aceml /app/uploads /app/experiments /app/logs
```

### Out of Memory

1. Increase Rancher Desktop memory allocation:
   - Open Rancher Desktop settings
   - Increase Memory (recommend 4GB minimum)
   - Restart Rancher Desktop

2. Or reduce dataset size in app settings

### Port Already in Use

```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Change port in docker-compose.yml
ports:
  - "5001:5000"  # Use port 5001 instead
```

### Health Check Failing

```bash
# Check if app is responding
docker-compose exec aceml-studio curl http://localhost:5000

# Disable health check temporarily (docker-compose.yml)
# Comment out the healthcheck section
```

---

## 🔐 Security Best Practices

1. **Never commit `.env` file** - It contains secrets
2. **Change SECRET_KEY** in production
3. **Set DEBUG=False** in production
4. **Use HTTPS** with a reverse proxy (nginx, traefik)
5. **Regularly update** the Docker image:
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

---

## 📊 Monitoring

### View Real-time Logs

```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100 -f

# Specific service
docker-compose logs -f aceml-studio
```

### Resource Usage

```bash
# Container stats
docker stats aceml-studio

# Disk usage
docker system df
```

### Health Status

```bash
# Check health
docker inspect --format='{{json .State.Health}}' aceml-studio | python -m json.tool
```

---

## 🔄 Updating the Application

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up -d --build

# Or force recreation
docker-compose up -d --force-recreate
```

---

## 🗑️ Cleanup

### Remove Everything

```bash
# Stop and remove containers, networks
docker-compose down

# Also remove volumes (CAUTION: deletes data)
docker-compose down -v

# Remove images
docker rmi aceml-studio:latest
```

### Clean Docker System

```bash
# Remove unused containers, networks, images
docker system prune

# Remove ALL volumes
docker volume prune
```

---

## 📝 Additional Resources

- [Rancher Desktop Documentation](https://docs.rancherdesktop.io/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [AceML Studio Repository](https://github.com/your-repo)

---

## 💡 Tips for Best Performance

1. **Use SSD** for Docker volumes
2. **Allocate enough RAM** (4GB+ recommended)
3. **Enable WSL2** backend if on Windows
4. **Use BuildKit** for faster builds:
   ```bash
   export DOCKER_BUILDKIT=1
   export COMPOSE_DOCKER_CLI_BUILD=1
   ```

---

## 🆘 Getting Help

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Review this troubleshooting section
3. Open an issue on GitHub with:
   - Error logs
   - Your configuration (without secrets)
   - Steps to reproduce

---

**Happy Machine Learning! 🚀**
