#!/bin/bash

# Database connection details
ENDPOINT="aurora-postgres-cluster.cluster-c5mxhlhzitip.us-east-1.rds.amazonaws.com"
PORT=5432
DB_NAME="mydb"
USERNAME="wchemz+demo@amazon.com"

# Generate authentication token
echo "Generating authentication token..."
TOKEN=$(aws rds generate-db-auth-token \
    --hostname $ENDPOINT \
    --port $PORT \
    --username $USERNAME)

echo "Token generated successfully!"
echo "Token: $TOKEN"
echo "Connecting to database..."

# Set debug mode for more information
export PGOPTIONS='-c client_min_messages=DEBUG1'

# Connect using psql with full path and verbose mode
PGPASSWORD=$TOKEN /opt/homebrew/opt/postgresql@15/bin/psql \
    "host=$ENDPOINT \
    port=$PORT \
    dbname=$DB_NAME \
    user=$USERNAME \
    sslmode=verify-full" \
    -v ON_ERROR_STOP=1
