#!/usr/bin/env bash
set -e

echo "Initializing Airflow DB..."
airflow db migrate

echo "Creating admin user (if not exists)..."
airflow users create \
  --username "${AIRFLOW_ADMIN_USER:-admin}" \
  --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
  --firstname Admin \
  --lastname Admin \
  --role Admin \
  --email "${AIRFLOW_ADMIN_EMAIL:-admin@example.com}" \
  || true

echo "Starting Airflow..."
exec "$@"
