# 07 — Deployment

Phase 4 notes. Two services, scaling on different signals, plus a shared object store.

---

## 1. The worker cannot run on plain Cloud Run

This is the deployment constraint most likely to be discovered late, so it is stated first.

**Cloud Run scales on inbound HTTP requests.** An `arq` worker polling Redis receives none, so
it never scales up from zero and, with `min-instances: 0`, never runs at all. Setting
`min-instances: 1` works but defeats NFR-06's scale-to-zero, which is the entire reason for
choosing a serverless target on a spiky workload.

| target | API | worker |
|---|---|---|
| **Azure Container Apps** | ✅ scales on HTTP | ✅ **scales on Redis queue length via KEDA, to zero** |
| Google Cloud Run | ✅ scales on HTTP | ❌ no inbound requests to scale on |
| Google Cloud Run **Jobs** | — | ✅ but needs a different dispatch design (Cloud Tasks or Pub/Sub push), not a config flag |

**Recommended: Azure Container Apps**, with a KEDA Redis-list scaler on the worker. The two
platforms are interchangeable for the API and are not for the worker, which the original spec
presented as a single choice.

## 2. Configuration

| variable | purpose | notes |
|---|---|---|
| `XLF_TOKEN_SECRET` | HMAC key for confirmation tokens | **No default.** An unset key must fail loudly rather than silently accept forged confirmations |
| `XLF_OBJECT_ROOT` | Object store root | Filesystem for self-hosted; S3/Azure Blob implement the same Protocol |
| `REDIS_URL` | Job state, queue, token replay | |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | Pinned to 1 | NFR-02 defines byte-identity relative to a *recorded* thread configuration; the manifest records what was used |

**Run more than one API instance only with `RedisReplayStore` configured.** The in-process
fallback is correct for a single instance and quietly wrong for several: each would accept the
same confirmation token once, so one confirmation could enqueue several jobs. A replay check
that does not work across instances is worse than none, because it reads as protection.

## 3. Sizing

Measured, not estimated (docs/05):

- A competition holds its cores for minutes. `WorkerSettings.max_jobs = 1`: each job already
  saturates its allotment through the engine's own `n_jobs`, and overcommitting slows every
  job rather than finishing any sooner.
- `job_timeout` is 3600s, which is generous on purpose. FR-801's resume is what makes a
  timeout survivable — a redelivered job restarts from its last completed fold.
- The image is 2.1 GB after removing 454 MB of CUDA libraries that `xgboost` pulls in for
  multi-GPU training we do not do. That is cold-start latency on a scale-to-zero target, and
  the next candidate is `llvmlite` + `numba` (208 MB), which exist only for `shap` in the
  `explain` extra — a worker image built without that extra would shed them, at the cost of
  maintaining two images.

## 4. Retention

`RetentionPolicy.sweep()` deletes uploaded panels and per-fold checkpoints past the window
(NFR-08, default 30 days). Results and manifests are **kept**: the v2 forecast-stability
feature compares against a previous cycle, and deleting manifests would make that impossible
without a migration. A leaderboard holds error metrics, not customer observations.

Run it as a scheduled job. Retention that only happens when someone asks is not retention.

## 5. What is not done

- **Auth is an `X-Owner` header stand-in.** The real scheme is an `HttpOnly; Secure;
  SameSite=Lax` session cookie (TS §6).
- **Encryption at rest** is delegated to the storage backend (S3 SSE / Azure SSE); the
  filesystem store does not encrypt, which is acceptable for self-hosted deployments where
  the disk is the customer's, and is not acceptable for our own hosting.
- **No Helm chart yet** (TS §12 promises one for enterprise pilots).
