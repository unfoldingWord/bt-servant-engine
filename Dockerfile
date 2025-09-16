#FROM python:3.12-slim
FROM cgr.dev/chainguard/wolfi-base

# Set Python version
ARG version=3.12

# Set working directory
WORKDIR /app

# Install required libraries
RUN apk update && apk add --no-cache \
    python-${version}=3.12.11-r6 \
    py${version}-pip \
    py${version}-setuptools

# Set ownership to nonroot
RUN chown -R nonroot:nonroot /app/

# Copy application code and entrypoint script
COPY . .
COPY entrypoint.sh /app/entrypoint.sh

# Ensure entrypoint.sh is executable
RUN chmod +x /app/entrypoint.sh

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to nonroot user
USER nonroot

# Set default environment variable (can be overridden at runtime)
ENV BT_SERVANT_LOG_LEVEL=info

# Expose port
EXPOSE 8080

# Use entrypoint script to start Uvicorn
ENTRYPOINT ["/app/entrypoint.sh"]
