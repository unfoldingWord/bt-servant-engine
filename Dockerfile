FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy application code and entrypoint script
COPY . .
COPY entrypoint.sh /app/entrypoint.sh

# Ensure entrypoint.sh is executable
RUN chmod +x /app/entrypoint.sh

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default environment variable (can be overridden at runtime)
ENV BT_SERVANT_LOG_LEVEL=info

# Expose port
EXPOSE 8080

# Use entrypoint script to start Uvicorn
ENTRYPOINT ["/app/entrypoint.sh"]
