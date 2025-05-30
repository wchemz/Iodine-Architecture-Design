#!/bin/bash

# Database connection details
ENDPOINT="aurora-postgres-cluster.cluster-c5mxhlhzitip.us-east-1.rds.amazonaws.com"
PORT=5432
DB_NAME="mydb"
MASTER_USER="dbadmin"
IAM_USER="wchemz+demo@amazon.com"

# Prompt for master password
echo -n "Enter master password for $MASTER_USER: "
read -s MASTER_PASSWORD
echo

# Create IAM user
echo "Creating IAM user in PostgreSQL..."
PGPASSWORD=$MASTER_PASSWORD /opt/homebrew/opt/postgresql@15/bin/psql \
    "host=$ENDPOINT \
    port=$PORT \
    dbname=$DB_NAME \
    user=$MASTER_USER \
    sslmode=verify-full" \
    -c "CREATE USER \"$IAM_USER\" WITH LOGIN;" \
    -c "GRANT rds_iam TO \"$IAM_USER\";" \
    -c "ALTER USER \"$IAM_USER\" WITH LOGIN PASSWORD 'dummy_password' VALID UNTIL 'infinity';" \
    -c "GRANT ALL PRIVILEGES ON DATABASE \"$DB_NAME\" TO \"$IAM_USER\";"

echo "IAM user setup complete!"
