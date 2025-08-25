## Phase 3: AWS hosted with same UX

### Objectives
- Host the API on AWS with minimal UX changes; local and cloud behave the same.
- Keep responses and routes identical to local API.

### Architecture (baseline)
- ECS (Fargate for CPU; EC2 GPU if MedSAM/GPU required) + ALB
- S3 for reports/artifacts; CloudFront for static UI (optional)
- SSM Parameter Store / Secrets Manager for config
- CloudWatch for logs/metrics; IAM roles for tasks

### Step-by-step
1) Containerization
   - Add `Dockerfile` (CPU) and `Dockerfile.gpu` (CUDA base, nvidia runtime).
   - Entrypoint: `uvicorn apps.local_api.main:app --host 0.0.0.0 --port 8080`.

2) Config binding
   - Map env vars from SSM to container (model paths become S3 URIs or EFS paths).
   - Switch `StorageService` to S3-backed implementation; return pre-signed URLs.

3) Infra
   - Provision ECS service + ALB target group + security groups.
   - Optionally EFS for model weights if not embedding in container.
   - Set health check `/healthz` and readiness `/readyz`.

4) CI/CD
   - Build/push image to ECR via GitHub Actions/CodeBuild.
   - Deploy via ECS rolling updates; add env promotion workflow.

5) Observability
   - Structured logs to CloudWatch; add metrics (latency, errors, queue depth if used).
   - Alarms on 5xx and p95 latency.

6) Security
   - WAF on ALB/CloudFront; limit upload size; MIME/type validation.
   - Short TTL pre-signed URLs; SSE-S3/KMS for buckets.

### Acceptance criteria
- The same client (local app/web) points to AWS URL and receives identical JSON and artifacts.
- Health checks pass; rolling deploys work; logs and alarms configured.


