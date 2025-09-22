## Phase 4: Enterprise-grade web platform on AWS

### Objectives
- Deliver a scalable, secure, multi-tenant web platform leveraging managed AWS services.

### Target architecture
- Frontend: S3 + CloudFront (SPA), Route53 custom domain, HTTPS
- API: ECS (Fargate/EC2 GPU) or AWS Lambda for lightweight U-Net paths
- Async processing: SQS + ECS workers or Step Functions for long-running jobs
- Storage: S3 (artifacts), DynamoDB (job metadata), EFS (optional weights/cache)
- Secrets/Config: Secrets Manager + SSM Parameter Store
- Observability: CloudWatch, X-Ray, centralized dashboards and alarms
- Security: WAF, Shield, KMS, VPC private subnets, IAM least privilege

### Step-by-step
1) Sync vs async API
   - `/analyze` (sync) for small images or U-Net-only; `/jobs` (async) for heavy pipelines.
   - Async: client requests upload URL → uploads to S3 → POST `/jobs` with S3 key → poll `/jobs/{id}` → download results via pre-signed URLs.
   - Done when:
     - Sync path returns result < p95 latency target; async path returns job_id.
     - `/jobs` lifecycle works end-to-end with worker processing and S3 artifacts.

2) Model management
   - Store weights in S3; cache in EFS or container layer.
   - Consider SageMaker Endpoints for MedSAM or GPU-heavy tasks.
   - Model versioning, blue/green rollout, and canary analysis.
   - Done when:
     - Models can be hot-swapped by version; rollback works.
     - Performance telemetry collected per model version.

3) Identity, auth, and tenancy
   - Cognito or OIDC (Auth0/Okta) for user auth; JWT on API.
   - Tenant isolation rules; quotas and rate limits per tenant.
   - Audit logging with user and tenant context.
   - Done when:
     - Auth tokens required; RBAC/tenant checks enforced on endpoints.
     - Quotas/rate limits measured and enforced; audit logs show user/tenant.

4) Data and compliance
   - S3 lifecycle rules; object tagging; encryption in transit and at rest (KMS).
   - PII handling, consent, and data retention controls.
   - Access logs and artifact provenance.
   - Done when:
     - Buckets have lifecycle, encryption, and access logging enabled.
     - Data retention policies documented and enforced; deletion flows verified.

5) Cost and performance
   - Right-size instances; autoscale on CPU/GPU/memory; spot workers.
   - Feature flags to disable expensive explainability for high-throughput paths.
   - Caching for repeated analyses; CDN for static SPA and common assets.
   - Done when:
     - Autoscaling policies hit targets; cost dashboards in place.
     - Feature flags dynamically toggle expensive paths without redeploy.

6) DevEx and CI/CD
   - IaC with Terraform/CDK; separate dev/stage/prod stacks.
   - CI for SPA (S3/CloudFront) and API (ECR/ECS/Lambda) with environment promotions.
   - Automated canaries and rollbacks on alarms.
   - Done when:
     - PR → preview → staged → prod promotion pipeline works with approvals.
     - Canary/rollback demonstrates safe deploy under fault injection.

7) Advanced features
   - Replace gTTS with Amazon Polly.
   - API keys/JWT/OIDC; throttling and quotas; signed download tokens.
   - Multi-region DR (active/passive) if required.
   - Done when:
     - Voice synthesis uses Polly; keys/tokens enforced where applicable.
     - DR runbook tested; RTO/RPO targets documented.

### Progress checklist
- [ ] Sync vs async API
  - [ ] Sync path meets p95 latency target; async returns `job_id`
  - [ ] `/jobs` lifecycle works E2E with worker and S3 artifacts
- [ ] Model management
  - [ ] Model versions deployed; blue/green and rollback validated
  - [ ] Perf telemetry captured per version
- [ ] Identity/Auth/Tenancy
  - [ ] Cognito/OIDC in place; JWT verified on API
  - [ ] Tenant isolation, quotas, rate limits enforced; audit logs contain user/tenant
- [ ] Data/Compliance
  - [ ] S3 lifecycle, encryption, access logging enabled
  - [ ] Data retention/deletion flows verified
- [ ] Cost/Performance
  - [ ] Autoscaling policies working; cost dashboards live
  - [ ] Feature flags toggle expensive paths without redeploy
- [ ] DevEx/CI/CD
  - [ ] Env stacks via Terraform/CDK; promotion pipeline running
  - [ ] Canary + automated rollback demonstrated
- [ ] Advanced features
  - [ ] Polly integrated; key/token protections enforced
  - [ ] DR runbook tested; RTO/RPO documented

### Acceptance criteria
- SPA + API operate at scale with queue-based workers and zero-downtime deploys.
- Strong security posture, tenant-aware auth, and auditable artifacts.
- Clear job tracking, predictable costs, and automated safeguards.


