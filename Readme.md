# Iodine Software Architecture Documentation

This repository contains comprehensive architecture documentation for various components of Iodine Software's AWS infrastructure.

## Components

### [Multi-Region Disaster Recovery](./Multi-region%20disaster%20recovery%20design/)
- Complete DR solution with 8-hour RTO and 15-minute RPO
- Active-passive configuration across regions
- Route53-based DNS failover
- Aurora PostgreSQL replication
- EC2 warm standby
- AWS Backup integration

### [RDS IAM Authentication](./Iodine%20RDS%20IAM%20Auth/)
- IAM-based authentication for RDS Aurora
- Secure access management
- Integration with AWS Identity Center

### [RDS Kerberos Authentication](./Iodine%20RDS%20Kerberos%20Auth/)
- Kerberos-based authentication flow
- Integration with Amazon Managed Active Directory
- Single sign-on experience for data team

### [SageMaker Pipeline](./Sagemaker%20pipeline/)
- Clinical AI model training pipeline with IAM RDS authentication
- Secure data extraction from Aurora PostgreSQL using IAM tokens
- 40% cost optimization through Spot instances
- 25% accuracy improvement in clinical AI models
- HIPAA-compliant infrastructure with end-to-end encryption
- Automated metrics logging to RDS with IAM authentication

## Key Features

### Security & Compliance
- HIPAA compliance across all components
- Encryption at rest and in transit
- IAM-based access control
- Audit logging and monitoring

### Cost Optimization
- Efficient resource utilization
- Spot Instance usage where applicable
- Auto-scaling configurations
- Multi-region cost management

### High Availability
- Multi-region deployment
- Automated failover capabilities
- Health monitoring and alerting
- Backup and recovery procedures

## Contact

For questions or support, please contact:
- AWS Team: Wei Chen (Sr. Solutions Architect, wchemz@amazon.com)
- AWS Team: McGraw, Milan (Senior AI/ML Specialist, mmcgraw@amazon.com)