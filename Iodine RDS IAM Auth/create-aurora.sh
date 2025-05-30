#!/bin/bash

# Set variables
VPC_ID="vpc-09f5473326ca06db1"
DB_NAME="mydb"
DB_USERNAME="dbadmin"
DB_PASSWORD="Password" # You should change this password
DB_PORT=5432
INSTANCE_CLASS="db.r5.large"

# Get public subnets from VPC
PUBLIC_SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
    --query 'Subnets[*].SubnetId' \
    --output text)

if [ -z "$PUBLIC_SUBNETS" ]; then
    echo "No public subnets found in VPC $VPC_ID"
    exit 1
fi

# Create or get DB subnet group
echo "Setting up DB subnet group..."
if ! aws rds describe-db-subnet-groups --db-subnet-group-name aurora-public-subnet-group >/dev/null 2>&1; then
    echo "Creating DB subnet group..."
    aws rds create-db-subnet-group \
        --db-subnet-group-name aurora-public-subnet-group \
        --db-subnet-group-description "Public subnet group for Aurora PostgreSQL" \
        --subnet-ids $PUBLIC_SUBNETS
else
    echo "Using existing DB subnet group"
fi

# Create or get security group
echo "Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=aurora-postgres-sg" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name aurora-postgres-sg \
        --description "Security group for Aurora PostgreSQL" \
        --vpc-id $VPC_ID \
        --output text)
else
    echo "Using existing security group: $SG_ID"
fi

# Get user's public IP
echo "Getting your public IP address..."
MY_IP=$(curl -s ifconfig.me)
MY_IP="$MY_IP/32"

# Update security group rules
echo "Updating security group rules for IP $MY_IP..."
aws ec2 revoke-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port $DB_PORT \
    --cidr 0.0.0.0/0 >/dev/null 2>&1 || true

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port $DB_PORT \
    --cidr $MY_IP >/dev/null 2>&1 || true

# Enable IAM authentication on existing cluster
echo "Enabling IAM authentication on existing cluster..."
aws rds modify-db-cluster \
    --db-cluster-identifier aurora-postgres-cluster \
    --enable-iam-database-authentication

echo "Waiting for cluster modification to complete..."
aws rds wait db-cluster-available \
    --db-cluster-identifier aurora-postgres-cluster

# Get cluster details
echo "Getting cluster endpoint..."
CLUSTER_ENDPOINT=$(aws rds describe-db-clusters \
    --db-cluster-identifier aurora-postgres-cluster \
    --query 'DBClusters[0].Endpoint' \
    --output text)

echo "IAM authentication enabled successfully!"
echo "Cluster Endpoint: $CLUSTER_ENDPOINT"
echo "Port: $DB_PORT"
echo "Database: $DB_NAME"
echo "Username: $DB_USERNAME"
echo "Password: $DB_PASSWORD"
