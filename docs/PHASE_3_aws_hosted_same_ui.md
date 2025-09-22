## Phase 3: AWS hosted with same UX (SPA + API)

### Objectives
- Host the browser SPA and API on AWS with identical contracts as local.
- Keep routes/responses unchanged; switch via environment only.

### Architecture (baseline)
- SPA: S3 (static hosting) + CloudFront distribution with HTTPS.
- API: ECS (Fargate CPU; EC2 GPU if needed) + ALB.
- Storage: S3 for reports/artifacts; optional EFS for model cache.
- Config/Secrets: SSM Parameter Store / Secrets Manager.
- Observability: CloudWatch logs/metrics; IAM roles for tasks.

### Step-by-step
1) Containerization (API)
   - Add `Dockerfile` (CPU) and `Dockerfile.gpu` (CUDA base, nvidia runtime).
   - Entrypoint: `uvicorn apps.local_api.main:app --host 0.0.0.0 --port 8080`.
   - Done when:
     - Images build locally and run with `docker run` exposing `/healthz`.
     - GPU image passes `nvidia-smi` and loads models on a GPU host.

2) SPA hosting
   - Build SPA (`apps/web`) and deploy to S3; front with CloudFront.
   - Inject `VITE_API_BASE_URL` via env-specific build or runtime config JSON.
   - Done when:
     - SPA is accessible via CloudFront HTTPS URL/custom domain.
     - SPA calls the cloud API successfully using the configured base URL.

3) Config binding
   - Map env vars from SSM to container (model paths become S3 URIs or EFS paths).
   - Switch `StorageService` to S3-backed implementation; return pre-signed URLs.
   - Done when:
     - Task role can read SSM/Secrets; app logs confirm config loaded.
     - Artifacts saved to S3 and URLs returned are accessible (within TTL).

4) Browser-friendly uploads
   - Add endpoint to create pre-signed S3 URLs for direct browser uploads.
   - SPA uploads image to S3, then calls `/analyze` with the S3 key.
   - Done when:
     - Pre-signed PUT works from browser; uploaded object visible in bucket.
     - `/analyze` processes by S3 key and returns artifact URLs.

5) Infra
   - Provision ECS service + ALB target group + security groups.
   - Optional EFS for model weights/cache if not embedded in image.
   - Health `/healthz` and readiness `/readyz` checks.
   - Done when:
     - Service is healthy behind ALB; health checks green.
     - Blue/green or rolling update completes without downtime.

6) CI/CD
   - API: build/push image to ECR; deploy via ECS rolling updates.
   - SPA: build and sync to S3; invalidate CloudFront cache.
   - Done when:
     - Push to main triggers build/deploy for both API and SPA.
     - CloudFront invalidation completes and new version is live.

7) Observability
   - Structured logs to CloudWatch; metrics on latency, errors, and upload size.
   - Alarms on 5xx and p95 latency.
   - Done when:
     - Dashboards show API latency/error rates; alarms wire to notifications.

8) Security
   - WAF on ALB/CloudFront; limit upload size; MIME/type validation.
   - Short TTL pre-signed URLs; SSE-S3/KMS for buckets; CORS allow SPA origin.
   - Done when:
     - WAF rules active; CORS allows only the SPA origin.
     - Bucket encryption verified; pre-signed URLs expire as expected.

### Progress checklist
- [ ] Containerization (API)
  - [ ] API image builds and runs locally exposing `/healthz`
  - [ ] GPU image validated on GPU host (`nvidia-smi`, model loads)
- [ ] SPA hosting
  - [ ] SPA deployed to S3 and fronted by CloudFront HTTPS
  - [ ] SPA calls cloud API via configured base URL
- [ ] Config binding
  - [ ] Task role reads SSM/Secrets; app logs confirm config
  - [ ] Artifacts saved to S3; pre-signed URLs accessible within TTL
- [ ] Browser uploads
  - [ ] Pre-signed PUT works from browser; object visible in bucket
  - [ ] `/analyze` accepts S3 key and returns artifact URLs
- [ ] Infra
  - [ ] ECS service healthy behind ALB; health checks green
  - [ ] Rolling/blue-green deploy completes without downtime
- [ ] CI/CD
  - [ ] Push to main triggers API and SPA deployments
  - [ ] CloudFront cache invalidated; new SPA live
- [ ] Observability
  - [ ] Dashboards show latency/error metrics; alarms wired
- [ ] Security
  - [ ] WAF enabled; CORS restricted to SPA origin
  - [ ] S3 buckets encrypted; pre-signed URLs expire correctly

### Acceptance criteria
- SPA hosted on CloudFront talks to API behind ALB using same JSON schemas.
- Presigned upload flow works end-to-end; artifacts served via pre-signed URLs.
- Health checks pass; rolling deploys work; logs and alarms configured.


