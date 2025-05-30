#!/bin/bash

# Set variables
CLUSTER_ID="aurora-postgres-cluster"
ENDPOINT="aurora-postgres-cluster.cluster-c5mxhlhzitip.us-east-1.rds.amazonaws.com"
PORT=5432
DB_NAME="postgres"
USERNAME="wchemz+demo@amazon.com"

# Enable IAM authentication
echo "Enabling IAM authentication on cluster $CLUSTER_ID..."
aws rds modify-db-cluster \
    --db-cluster-identifier $CLUSTER_ID \
    --enable-iam-database-authentication \
    --apply-immediately

echo "Waiting for cluster modification to complete..."
while true; do
    IAM_STATUS=$(aws rds describe-db-clusters \
        --db-cluster-identifier $CLUSTER_ID \
        --query 'DBClusters[0].IAMDatabaseAuthenticationEnabled' \
        --output text)
    
    if [ "$IAM_STATUS" = "true" ]; then
        echo "IAM authentication has been enabled successfully!"
        break
    else
        echo "Waiting for IAM authentication to be enabled..."
        sleep 10
    fi
done

aws rds wait db-cluster-available \
    --db-cluster-identifier $CLUSTER_ID

# Get resource ID
echo "Getting cluster resource ID..."
RESOURCE_ID=$(aws rds describe-db-clusters \
    --db-cluster-identifier $CLUSTER_ID \
    --query 'DBClusters[0].DbClusterResourceId' \
    --output text)

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
AWS_REGION=$(aws configure get region)

echo "IAM authentication enabled successfully!"
echo "----------------------------------------"
echo "Cluster Endpoint: $ENDPOINT"
echo "Port: $PORT"
echo "Database: $DB_NAME"
echo "Username: $USERNAME"
echo "Resource ID: $RESOURCE_ID"
echo ""
echo "Next steps:"
echo "1. Create an IAM policy for database access:"
echo "----------------------------------------"
cat << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "rds-db:connect"
            ],
            "Resource": [
                "arn:aws:rds-db:${AWS_REGION}:${AWS_ACCOUNT_ID}:dbuser:${RESOURCE_ID}/${USERNAME}"
            ]
        }
    ]
}
EOF
echo ""
echo "2. Generate authentication token when connecting:"
echo "----------------------------------------"
echo "TOKEN=\$(aws rds generate-db-auth-token --hostname $ENDPOINT --port $PORT --username $USERNAME)"
echo ""
echo "3. Connect using psql with SSL:"
echo "----------------------------------------"
echo "psql \"host=$ENDPOINT port=$PORT dbname=$DB_NAME user=$USERNAME password=\$TOKEN sslmode=verify-full\""
