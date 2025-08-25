## Phase 4: Enterprise-grade AWS architecture

### Objectives
- Distribute responsibilities across managed AWS services for scale, cost, and resilience.

### Target architecture
- Frontend: S3 + CloudFront (static), Route53
- API: ECS (Fargate/EC2 GPU) or AWS Lambda (lightweight U-Net path)
- Async processing: SQS + ECS workers or Step Functions
- Storage: S3 (artifacts), DynamoDB (job metadata), EFS (optional weights/cache)
- Secrets/Config: Secrets Manager + SSM Parameter Store
- Observability: CloudWatch, X-Ray, centralized dashboards and alarms
- Security: WAF, Shield, KMS, VPC private subnets, IAM least privilege

### Step-by-step
1) Split sync vs async
   - `/analyze` enqueues to SQS; returns job_id.
   - Worker consumes, runs `woundseg` pipeline, writes artifacts to S3, updates DynamoDB status.
   - `/status/{job_id}` and `/result/{job_id}` serve status and pre-signed URLs.

2) Model management
   - Store weights in S3; cache in EFS or container layer. Consider SageMaker Endpoints for MedSAM.
   - Add model versioning and blue/green rollout.

3) Data and compliance
   - S3 lifecycle rules; object tagging; encryption at rest and in transit.
   - Access logs, audit trails; PII handling policy.

4) Cost and performance
   - Right-size instances; autoscale on CPU/GPU/memory; spot for workers.
   - Feature flags to disable expensive explainability for high-throughput paths.

5) DevEx and CI/CD
   - IaC with Terraform/CDK; per-env stacks (dev/stage/prod).
   - Canary deployments and automated rollback on alarm.

6) Advanced features
   - Replace gTTS with Amazon Polly.
   - Add API keys/JWT/OIDC; throttling and quotas.
   - Multi-region DR (active/passive) if required.

### Acceptance criteria
- Horizontal scalability with queue-based workers; zero-downtime deploys; strong security posture.
- Clear job tracking and auditable artifacts; cost controls in place.


