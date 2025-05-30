# RDS Kerberos Authentication Architecture

## Kerberos Authentication Flow

```mermaid
sequenceDiagram
    actor Iodine Data team user
    participant JC as JumpCloud
    participant IAM IC as IAM Identity Center
    participant SM as Sagemaker
    participant BG as Background Process
    participant AD as Amazon Managed Active Directory
    participant Aurora PostgreSQL

    JC->>AD: Sync or establish trust relationship
    Iodine Data team user->>IAM IC: Log in via SAML from JumpCloud
    IAM IC->>SM: Authenticate user using STS
    Iodine Data team user->>SM: Launch notebook
    SM->>BG: Initiate background auth process using boto3/Kerberos
    BG->>IAM IC: Get user identity
    BG->>AD: Map IAM identity to AD user
    AD->>BG: Return AD user
    BG->>AD: Request Kerberos ticket
    AD->>BG: Provide Kerberos ticket
    BG->>Aurora PostgreSQL: Authenticate with Kerberos ticket
    Aurora PostgreSQL->>BG: Authentication successful
    BG->>SM: Provide authenticated database connection
    Iodine Data team user->>SM: Run queries in notebook
    SM->>Aurora PostgreSQL: Execute queries (with user identity)
    Aurora PostgreSQL->>SM: Return query results
    
    note over BG,Aurora PostgreSQL: Authentication persists for long-running queries
    note over SM,Aurora PostgreSQL: Access multiple clusters with same credentials
    note over Aurora PostgreSQL: Queries logged with specific user identity
```

## Architecture Components

### Training Environment
- GPU AutoScaling Group for model training
- S3 buckets for storing trained models

### Model Registry
- ClearML in US-East-1 for model registry and orchestration
- Integration with S3 for model storage

### Inference Environment
- EKS on EC2 for running inference workloads
- Kubernetes for container orchestration
- CPU-based inference optimization

### Authentication Flow
1. Initial trust relationship between JumpCloud and Amazon Managed AD
2. SAML-based authentication through IAM Identity Center
3. SageMaker notebook authentication using STS
4. Kerberos ticket generation for Aurora PostgreSQL access
5. Persistent authentication for long-running queries

## Key Benefits
- Single sign-on experience for data team users
- Centralized user management through JumpCloud
- Secure access to multiple Aurora PostgreSQL clusters
- Persistent authentication for improved user experience
- Detailed query logging with user identity

## Contact

For questions or support, please contact:
- AWS Team: Wei Chen (Sr. Solutions Architect, wchemz@amazon.com)
