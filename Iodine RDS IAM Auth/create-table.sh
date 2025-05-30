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

echo "Creating iodine table..."
PGPASSWORD=$TOKEN /opt/homebrew/opt/postgresql@15/bin/psql \
    "host=$ENDPOINT \
    port=$PORT \
    dbname=$DB_NAME \
    user=$USERNAME \
    sslmode=verify-full" \
    -c "CREATE TABLE IF NOT EXISTS iodine (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        atomic_number INTEGER NOT NULL,
        atomic_mass DECIMAL(10,4),
        discovery_year INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );" \
    -c "INSERT INTO iodine (name, atomic_number, atomic_mass, discovery_year) 
        VALUES ('Iodine', 53, 126.90447, 1811);"

echo "Table created and sample data inserted!"
