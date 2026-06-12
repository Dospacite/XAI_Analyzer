# Traceguard

Traceguard is an explainable phishing-analysis application. It accepts a URL,
retrieves the page in an isolated scraper service, extracts the feature contract
used during training, sends that feature document to four independently
fine-tuned Hugging Face models, and lets the user ask follow-up questions about
the model findings.

The model repositories are public:

- `Dospacite/xai-phishing-qwen3-4b-merged`
- `Dospacite/xai-phishing-llama-3.1-8b-merged`
- `Dospacite/xai-phishing-deepseek-r1-qwen-7b-merged`
- `Dospacite/gemma4-e4b-unsloth-phishing-merged`

Public repositories let another runner create their own Hugging Face Inference
Endpoints from the same model artifacts. The application still requires an
`HF_TOKEN`, because it manages those endpoints by calling Hugging Face's
endpoint status, resume, pause, and inference APIs. The API refuses to start
without `HF_TOKEN`.

## Quickstart

Minimum steps to run the application locally:

1. Install Docker Engine with the Docker Compose plugin.
2. Create the runtime environment file:

   ```bash
   cd app
   cp .env.example .env
   ```

3. Edit `.env` and set the required values:

   ```dotenv
   HF_TOKEN=hf_your_token
   SCRAPER_TOKEN=replace-with-a-long-random-value
   SQLITE_PATH=/data/traceguard.sqlite3
   PUBLIC_APP_URL=http://localhost:3000

   HF_QWEN_ENDPOINT_URL=https://your-qwen-endpoint.endpoints.huggingface.cloud
   HF_LLAMA_ENDPOINT_URL=https://your-llama-endpoint.endpoints.huggingface.cloud
   HF_DEEPSEEK_ENDPOINT_URL=https://your-deepseek-endpoint.endpoints.huggingface.cloud
   HF_GEMMA_ENDPOINT_URL=https://your-gemma-endpoint.endpoints.huggingface.cloud

   HF_ENDPOINT_NAMESPACE=your-huggingface-namespace
   HF_QWEN_ENDPOINT_NAME=your-qwen-endpoint-name
   HF_LLAMA_ENDPOINT_NAME=your-llama-endpoint-name
   HF_DEEPSEEK_ENDPOINT_NAME=your-deepseek-endpoint-name
   HF_GEMMA_ENDPOINT_NAME=your-gemma-endpoint-name
   ```

   Follow-up chat also needs `QWEN_API_KEY`, `QWEN_BASE_URL`, and `QWEN_MODEL`.
   URL analysis can start without those values, but follow-up messages will not
   work until they are configured.

4. If the Hugging Face endpoints do not exist yet, create them from `.env`:

   ```bash
   python3 -m pip install "huggingface_hub>=0.19.0"
   python3 scripts/setup_hf_endpoints.py --wait --yes
   ```

   This creates billable dedicated endpoints and writes the generated
   `HF_*_ENDPOINT_URL` values back into `.env`.

5. Build and start the containers:

   ```bash
   docker compose up --build
   ```

6. Open the app:

   ```text
   http://localhost:3000
   ```

7. Optional health check:

   ```bash
   curl http://localhost:3000/api/health
   ```

   Expected response:

   ```json
   {"status":"ok"}
   ```

## Architecture

```text
Browser
  |
  v
Frontend: React + Nginx on http://localhost:3000
  |
  v
API: FastAPI on the internal Docker network
  |      |
  |      +--> SQLite thread store
  |      +--> Hugging Face Inference Endpoints
  |      +--> Qwen-compatible chat API for titles and follow-up questions
  |
  v
Scraper: Scrapling + Playwright on an isolated internal Docker network
```

The Docker Compose file starts three services:

- `frontend`: builds the React app and serves it with unprivileged Nginx.
- `api`: validates URLs, stores thread state in SQLite, extracts features, calls
  Hugging Face models, and handles follow-up chat.
- `scraper`: fetches public HTTP/HTTPS pages with a stricter network boundary.

Thread history is persisted in a SQLite database. The Docker setup stores that
database in the `traceguard_data` named volume.

## What You Need

Install or prepare:

- Docker Engine with the Docker Compose plugin.
- Network access to build Docker images and reach external APIs.
- A Hugging Face account and access token.
- Four Hugging Face Inference Endpoints, one per model repository.
- A Qwen-compatible chat API key for title generation and follow-up questions.

You do not need Node or Python on the host for normal Docker usage.

## Environment File

Create a local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env`.

### Required Application Values

```dotenv
HF_TOKEN=hf_your_token
SQLITE_PATH=/data/traceguard.sqlite3
SCRAPER_TOKEN=replace-with-a-long-random-value
PUBLIC_APP_URL=http://localhost:3000
```

`HF_TOKEN` is required. It must be able to call the four endpoint URLs and, when
`HF_MANAGE_ENDPOINT_LIFECYCLE=true`, read/resume/pause those endpoints.

`SQLITE_PATH` is where the API stores analysis threads and follow-up messages.
For Docker Compose, keep `/data/traceguard.sqlite3`; `/data` is backed by the
`traceguard_data` named volume.

`SCRAPER_TOKEN` is an internal shared secret between the API and scraper
containers. Use a long random value. It is not a user login token.

`PUBLIC_APP_URL` is used by FastAPI CORS. For local Docker usage, keep
`http://localhost:3000`.

Generate a scraper token with:

```bash
openssl rand -hex 32
```

or:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Hugging Face Model Values

```dotenv
HF_QWEN_REPO_ID=Dospacite/xai-phishing-qwen3-4b-merged
HF_LLAMA_REPO_ID=Dospacite/xai-phishing-llama-3.1-8b-merged
HF_DEEPSEEK_REPO_ID=Dospacite/xai-phishing-deepseek-r1-qwen-7b-merged
HF_GEMMA_REPO_ID=Dospacite/gemma4-e4b-unsloth-phishing-merged

HF_QWEN_ENDPOINT_URL=https://your-qwen-endpoint.endpoints.huggingface.cloud
HF_LLAMA_ENDPOINT_URL=https://your-llama-endpoint.endpoints.huggingface.cloud
HF_DEEPSEEK_ENDPOINT_URL=https://your-deepseek-endpoint.endpoints.huggingface.cloud
HF_GEMMA_ENDPOINT_URL=https://your-gemma-endpoint.endpoints.huggingface.cloud
```

The repository IDs identify the public model repositories. The endpoint URLs are
the actual services the application calls at runtime.

If an endpoint URL is blank, that model will show `Dedicated endpoint not
configured` in the UI.

### Endpoint Lifecycle Values

Lifecycle management is on by default:

```dotenv
HF_ENDPOINT_NAMESPACE=Dospacite
HF_MANAGE_ENDPOINT_LIFECYCLE=true
HF_ENDPOINT_START_TIMEOUT_SECONDS=900
HF_ENDPOINT_POLL_SECONDS=10

HF_QWEN_ENDPOINT_NAME=traceguard-qwen3-4b
HF_LLAMA_ENDPOINT_NAME=traceguard-llama-3-1-8b
HF_DEEPSEEK_ENDPOINT_NAME=traceguard-deepseek-r1-qwen-7b
HF_GEMMA_ENDPOINT_NAME=traceguard-gemma4-e4b
```

When lifecycle management is enabled, the API resumes an endpoint before
inference, polls until Hugging Face reports it as running, sends the inference
request, and pauses the endpoint after the last active in-process request
finishes.

Set `HF_ENDPOINT_NAMESPACE` to the Hugging Face namespace that owns the
endpoints. Set each `HF_*_ENDPOINT_NAME` to the endpoint name shown in
Hugging Face, not the full URL.

### Qwen Chat Values

The four phishing model calls use Hugging Face. Follow-up chat uses a
Qwen-compatible chat-completions API:

```dotenv
QWEN_API_KEY=replace_me
QWEN_BASE_URL=https://example.openai.azure.com/v1
QWEN_MODEL=qwen3.5-flash
```

If these values are blank or wrong:

- URL analysis can still run.
- Thread titles fall back to a hostname-based title.
- Follow-up messages fail because the assistant chat backend is unavailable.

`QWEN_BASE_URL` can be either the API root or a full `/chat/completions` URL.
The backend appends `/chat/completions` when needed.

### Optional Scraper Limits

```dotenv
SCRAPER_MAX_BYTES=5000000
SCRAPER_TIMEOUT_SECONDS=30
```

`SCRAPER_MAX_BYTES` limits the fetched response body size.

`SCRAPER_TIMEOUT_SECONDS` controls the browser/static fetch timeout.

## Create Hugging Face Endpoints

Each model needs its own dedicated Hugging Face Inference Endpoint. The app runs
all four models independently and combines their results.

Recommended starting hardware:

- Qwen3 4B merged: one 16 GB or larger GPU.
- Llama 3.1 8B merged BF16: one L40S or similar larger GPU.
- DeepSeek R1 Distill Qwen 7B BF16: one 24 GB or larger GPU.
- Gemma 4 E4B merged BF16: one 48 GB or larger GPU.

Endpoint setup:

1. Open the model repository on Hugging Face.
2. Choose `Deploy` and then `Inference Endpoints`.
3. Create a dedicated endpoint for the model.
4. Use the standard text-generation task for the merged Qwen and DeepSeek
   repositories.
5. Use the repository custom handler where required for Llama and Gemma.
6. Choose GPU capacity appropriate for the model.
7. Wait until the endpoint is created.
8. Copy the endpoint URL into `.env`.
9. Copy the endpoint name into the matching `HF_*_ENDPOINT_NAME` variable.

Endpoint hosting is billable. This repository does not create endpoints
automatically.

You can create the four endpoints from `.env` with the helper script:

```bash
python3 -m pip install "huggingface_hub>=0.19.0"
python3 scripts/setup_hf_endpoints.py --wait
```

The script reads `HF_TOKEN`, `HF_ENDPOINT_NAMESPACE`, model repository IDs, and
endpoint names from `.env`. It creates only missing endpoints, leaves existing
endpoints in place, and writes any available endpoint URLs back into `.env`.

Use `--dry-run` to preview the endpoint plan without creating anything:

```bash
python3 scripts/setup_hf_endpoints.py --dry-run
```

Endpoint creation settings default to AWS `us-east-1`, protected endpoints,
PyTorch, `text-generation`, one replica, and these GPUs:

- Qwen: `nvidia-l4` `x1`
- Llama: `nvidia-l40s` `x1`
- DeepSeek: `nvidia-l4` `x1`
- Gemma: `nvidia-l40s` `x1`

Override the shared settings with variables such as `HF_ENDPOINT_VENDOR`,
`HF_ENDPOINT_REGION`, `HF_ENDPOINT_INSTANCE_TYPE`, and
`HF_ENDPOINT_INSTANCE_SIZE`. Override a single model with variables such as
`HF_QWEN_ENDPOINT_INSTANCE_TYPE` or `HF_GEMMA_ENDPOINT_INSTANCE_SIZE`.

## Run With Docker Compose

Build and start everything:

```bash
docker compose up --build
```

Open:

```text
http://localhost:3000
```

The first build can take a while because the scraper image installs Playwright
Chromium and system browser dependencies.

Run in the background:

```bash
docker compose up --build -d
```

Stop the app:

```bash
docker compose down
```

This stops and removes the containers, but keeps the `traceguard_data` volume
and the SQLite database.

Stop the app and delete all persisted analyses:

```bash
docker compose down -v
```

Rebuild after code changes:

```bash
docker compose build
docker compose up
```

## Verify The Setup

Check running containers:

```bash
docker compose ps
```

The `api` and `scraper` services should become healthy. The frontend waits for
the API health check before starting.

Check the API through the frontend proxy:

```bash
curl http://localhost:3000/api/health
```

Expected response:

```json
{"status":"ok"}
```

Create an analysis:

1. Open `http://localhost:3000`.
2. Enter a complete URL such as `https://example.com/`.
3. Wait while the app retrieves the page, extracts features, resumes endpoints,
   and runs the model requests.
4. Open each model panel to inspect risk factors, mitigating factors, reasoning,
   and any endpoint errors.

The app creates the thread immediately and processes the URL in the background.
The frontend polls until the thread is ready.

## Frontend Development

Keep the backend running through Docker, then run Vite locally:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

FastAPI allows `http://localhost:5173` in CORS for local development.

## Backend Tests

Create or reuse a Python virtual environment, install backend requirements, and
run tests:

```bash
python -m venv .venv
.venv/bin/pip install -r services/api/requirements.txt
PYTHONPATH=services/api .venv/bin/pytest -q services/api/tests
```

The tests cover:

- Feature contract behavior.
- Model prompt formatting and parser behavior.
- Hugging Face endpoint lifecycle behavior.
- The required `HF_TOKEN` failure path.
- SQLite persistence for thread state.

## Frontend Build Check

```bash
cd frontend
npm install
npm run build
```

The Docker frontend image runs the same build during `docker compose build`.

## Troubleshooting

### API Fails With `HF_TOKEN is required`

Set `HF_TOKEN` in `.env`. The API intentionally refuses to start without a
Hugging Face token.

### API Container Is Unhealthy

Inspect logs:

```bash
docker compose logs api
```

Common causes are missing `HF_TOKEN`, an invalid endpoint namespace/name, a
token without endpoint permissions, or blocked network access to Hugging Face.

### Scraper Container Is Unhealthy

Inspect logs:

```bash
docker compose logs scraper
```

The first build may be slow because Playwright browser dependencies are large.
At runtime, the scraper rejects localhost, private IPs, reserved IPs, URLs with
embedded credentials, non-HTTP schemes, and ports other than 80 or 443.

### A Model Shows `Dedicated endpoint not configured`

The corresponding `HF_*_ENDPOINT_URL` value is blank. Add the endpoint URL to
`.env` and restart the API:

```bash
docker compose up -d --build api
```

### A Model Shows `401` Or `Unauthorized`

Check that `HF_TOKEN` belongs to the account or organization allowed to call the
endpoint. If lifecycle management is enabled, the token also needs permission to
read, resume, and pause the endpoint.

### A Model Shows Endpoint Lifecycle Errors

Check:

- `HF_ENDPOINT_NAMESPACE` matches the namespace that owns the endpoints.
- `HF_*_ENDPOINT_NAME` matches the endpoint name, not the model repository name.
- `HF_TOKEN` has permission to read/resume/pause the endpoint.

### A Model Times Out Or Returns `503`

The endpoint may still be starting, overloaded, or under-provisioned. Increase
`HF_ENDPOINT_START_TIMEOUT_SECONDS`, choose a larger GPU, or inspect the
endpoint logs in Hugging Face.

### Follow-Up Chat Fails

Check:

```dotenv
QWEN_API_KEY=
QWEN_BASE_URL=
QWEN_MODEL=
```

The initial URL analysis and follow-up assistant use different providers. A
working Hugging Face setup does not automatically make follow-up chat work.

### Thread History Is Missing After Restart

Check that `SQLITE_PATH=/data/traceguard.sqlite3` and that the Compose volume is
mounted:

```bash
docker compose config
docker volume ls
```

Thread history is deleted if you run `docker compose down -v`, remove the
`traceguard_data` volume, or point `SQLITE_PATH` at a non-persistent location.

### Frontend Loads But API Calls Fail

Use the Docker URL `http://localhost:3000`, not the internal API container URL.
Nginx proxies `/api/*` requests to the API service inside the Docker network.

## Security Notes

- Treat the verdicts as decision support, not an automatic blocking decision.
- The scraper intentionally refuses local and private-network targets to reduce
  SSRF risk.
- The scraper does not receive Hugging Face or Qwen credentials.
- Raw HTML is stored in SQLite as part of the thread state and is not returned
  directly by the frontend as executable content.
- Rotate any credential that was previously committed, copied into notebooks, or
  shared outside the intended runtime environment.

## Operational Notes

- Keep `HF_MANAGE_ENDPOINT_LIFECYCLE=true` unless you intentionally want to
  manage endpoint startup and shutdown yourself.
- SQLite persistence is suitable for a single API replica. Do not run multiple
  API containers against the same SQLite file.
- For a multi-replica API deployment, replace FastAPI background tasks and the
  SQLite store with a durable job queue and a server database.
- For production, put the frontend behind HTTPS and set `PUBLIC_APP_URL` to the
  real public origin.

## Project Layout

```text
.
|-- docker-compose.yml
|-- .env.example
|-- frontend/
|   |-- src/
|   |-- Dockerfile
|   `-- nginx.conf
|-- services/
|   |-- api/
|   |   |-- app/
|   |   |-- tests/
|   |   `-- Dockerfile
|   `-- scraper/
|       |-- app/
|       `-- Dockerfile
|-- hf_handlers/
|-- scripts/
`-- adapter/
```

The `adapter`, `final_adapter`, `deepseek-phishing-v3-final`, and
`gemma4-e4b-unsloth-phishing` directories contain model artifacts, handlers,
training outputs, or supporting files. The runtime Docker app uses the Hugging
Face endpoint URLs configured in `.env`.
