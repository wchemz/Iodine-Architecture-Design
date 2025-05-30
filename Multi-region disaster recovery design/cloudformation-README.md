# Multi-Region Disaster Recovery CloudFormation Template

This CloudFormation template creates a complete multi-region disaster recovery infrastructure for Iodine Software's AWS environment.

## Overview

The template provisions:
- Primary and DR region infrastructure
- Route53 DNS failover configuration
- Aurora PostgreSQL clusters with replication
- EC2 Auto Scaling Groups with warm standby
- AWS Backup configuration with cross-region copies

## Prerequisites

1. AWS CLI installed and configured
2. Appropriate IAM permissions
3. Two AWS regions selected (primary and DR)
4. Existing VPC infrastructure in both regions
5. SSL certificates in AWS Certificate Manager
6. Secrets in AWS Secrets Manager for database credentials

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Environment | Environment type | production |
| PrimaryRegion | Primary AWS region | us-east-1 |
| DRRegion | DR AWS region | us-east-2 |
| DBInstanceClass | Aurora instance class | db.r7g.xlarge |
| DRDBInstanceClass | DR Aurora instance class | db.r7g.large |
| EC2InstanceType | EC2 instance type | c6g.xlarge |
| DREC2InstanceType | DR EC2 instance type | t3.small |
| BackupRetentionPeriod | Backup retention days | 35 |

## Components

### Route53 Configuration
- Health checks for primary region
- DNS failover records
- Alias records for ALBs

### Primary Region
- Aurora PostgreSQL cluster
- EC2 Auto Scaling Group
- Application Load Balancer
- Security Groups and IAM roles

### DR Region
- Aurora PostgreSQL replica cluster
- Minimal EC2 Auto Scaling Group (warm standby)
- Application Load Balancer
- Security Groups and IAM roles

### AWS Backup
- Backup vault in primary region
- Cross-region copy to DR region
- Daily backup schedule
- 35-day retention period

## Deployment

1. Create required prerequisites (VPCs, subnets, etc.)
2. Update parameter values in parameters.json:
   - Review all parameters in parameters.json
   - Update values according to your environment:
     * HostedZoneId: Your Route53 hosted zone ID
     * Subnet IDs: Your VPC subnet IDs
     * Security Group IDs: Your security group IDs
     * DB Subnet Groups: Your RDS subnet group names
   - Validate the parameters match your infrastructure
3. Deploy the template:

```bash
aws cloudformation create-stack \
  --stack-name iodine-dr-infrastructure \
  --template-body file://dr-infrastructure.yaml \
  --parameters file://parameters.json \
  --capabilities CAPABILITY_NAMED_IAM
```

4. Monitor stack creation:

```bash
aws cloudformation describe-stacks \
  --stack-name iodine-dr-infrastructure
```

## Validation

After deployment, verify:
1. Route53 health checks are passing
2. Aurora replication is working
3. EC2 instances are running in both regions
4. Backups are being created and copied
5. DNS failover by testing the application URL

## Failover Testing

1. Test automated failover:
   - Stop primary Aurora cluster
   - Verify DNS failover to DR region
   - Confirm application accessibility

2. Test manual failover:
   - Follow runbook procedures
   - Execute failover commands
   - Verify application functionality

## Cost Optimization

The template implements cost optimization through:
- Smaller instance types in DR region
- Minimal EC2 capacity in DR region
- Spot Instances where applicable
- Automated backup lifecycle management

## Security

The template implements security best practices:
- HIPAA compliance configurations
- Encryption at rest and in transit
- IAM roles with least privilege
- Security group restrictions
- Private subnets for resources

## Monitoring

Monitor the infrastructure using:
- CloudWatch metrics and alarms
- Route53 health check status
- Aurora replication metrics
- EC2 instance health checks

## Support

For issues or questions, contact:
- Platform Team: Kelly Yao (Sr. Director Platform Engineering)
- DevOps Team: Brendan Laws (Director of DevOps)
- Security Team: Cheng Zhou (Director of SRE/InfoSec)
