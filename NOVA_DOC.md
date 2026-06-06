# NOVA — Documentation complète

> **Pour Karl.**
> Ce doc couvre tout ce que t'as besoin pour comprendre, faire tourner, et continuer le projet.
> Écrit en québécois informel parce que c'est toi l'audience.

---

## Table des matières

1. [Vision du projet](#1-vision-du-projet)
2. [Architecture](#2-architecture)
3. [Structure des fichiers](#3-structure-des-fichiers)
4. [Base de données](#4-base-de-données)
5. [API — tous les endpoints](#5-api--tous-les-endpoints)
6. [Variables d'environnement](#6-variables-denvironnement)
7. [Installation & run local (Mac)](#7-installation--run-local-mac)
8. [Déploiement sur le serveur](#8-déploiement-sur-le-serveur-proxmox--ubuntu-vm)
9. [Ajouter un agent](#9-ajouter-un-agent)
10. [Roadmap](#10-roadmap)

---

## 1. Vision du projet

NOVA c'est ton Jarvis personnel. Self-hosted, accessible partout via Tailscale, bilingue.

**Principe de base :**
- **Cerveau local** → Ollama + llama3.2 pour les conversations de tous les jours (gratuit, privé, offline)
- **Cerveau cloud** → Claude API pour les tâches complexes (analyse de code, recherche approfondie)
- **Mémoire persistante** → PostgreSQL pour l'historique + les faits sur toi
- **Mémoire sémantique** → ChromaDB (à venir) pour chercher dans tout l'historique par sens, pas juste par mots-clés
- **Voix** → Whisper (STT) + Piper (TTS), voix française québécoise + anglaise
- **Multi-agents** → agents spécialisés pour les études, projets, job, recherche

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NOVA — Vue d'ensemble                           │
└─────────────────────────────────────────────────────────────────────────┘

  iPhone / MacBook / n'importe quel appareil
        │  HTTPS via Tailscale
        ▼
  ┌─────────────────────────────────────────────────────┐
  │              FastAPI  (main.py :8000)               │
  │                                                     │
  │   GET  /           → Interface web                  │
  │   POST /chat/stream → SSE streaming                 │
  │   POST /speak      → TTS (Piper)                    │
  │   POST /transcribe → STT (Whisper)                  │
  │   GET  /memory/*   → Historique + faits             │
  │   GET  /health     → Status système                 │
  └──────────┬──────────────┬───────────────────────────┘
             │              │
     ┌───────▼──────┐  ┌───▼────────────────────────┐
     │   LLM Router │  │       Memory Manager        │
     │  (core/llm)  │  │      (core/memory)          │
     │              │  │                             │
     │  ┌─────────┐ │  │  ┌──────────────────────┐  │
     │  │  Ollama │ │  │  │    PostgreSQL         │  │
     │  │llama3.2 │ │  │  │  conversations        │  │
     │  │:11434   │ │  │  │  messages             │  │
     │  └─────────┘ │  │  │  facts                │  │
     │              │  │  └──────────────────────┘  │
     │  ┌─────────┐ │  │                             │
     │  │  Claude │ │  │  ┌──────────────────────┐  │
     │  │   API   │ │  │  │   ChromaDB (futur)   │  │
     │  │ (futur) │ │  │  │   mémoire vectorielle │  │
     │  └─────────┘ │  │  └──────────────────────┘  │
     └──────────────┘  └────────────────────────────┘
             │
     ┌───────▼──────────────────────────────────────┐
     │              Agent Router (futur)             │
     │                                               │
     │  StudyAgent   WorkAgent   ProjectAgent        │
     │  (ÉTS)        (Tricentis) (BudgetApp/homelab) │
     └───────────────────────────────────────────────┘
             │
     ┌───────▼───────────────────┐
     │  Voice Pipeline           │
     │  Whisper STT  Piper TTS   │
     │  (core/stt)   (core/tts)  │
     └───────────────────────────┘

Infrastructure (serveur Proxmox):
  Ubuntu VM (docker-host) @ 100.124.106.27 (Tailscale)
    ├── Ollama  :11434
    ├── PostgreSQL  :5432 (db: jarvis)
    ├── Open WebUI  :3000
    └── NOVA Docker container  :8000
```

### Flux d'une conversation (streaming)

```
Karl tape un message
      │
      ▼
POST /chat/stream (JSON: message, session_id)
      │
      ├── 1. get_or_create_session()   → PostgreSQL
      ├── 2. get_conversation_history() → contexte des 12 derniers tours
      ├── 3. get_facts()               → faits sur Karl (mémoire long terme)
      ├── 4. detect_language()         → fr / en
      ├── 5. save_message(user)        → PostgreSQL
      │
      ▼
Ollama /api/chat  (stream=true)
      │
      ▼  tokens arrivent un par un
SSE  data: {"type":"token","content":"Salut"}
SSE  data: {"type":"token","content":" Karl"}
SSE  data: {"type":"done","language":"fr"}
      │
      ├── 6. save_message(assistant)   → PostgreSQL
      │
      ▼
Interface web reçoit les tokens, les affiche en temps réel
```

---

## 3. Structure des fichiers

```
nova-mac/
│
├── main.py                 # Point d'entrée FastAPI — tous les endpoints
│
├── core/
│   ├── __init__.py         # Package marker
│   ├── config.py           # Settings Pydantic (lit .env) — tout config ici
│   ├── memory.py           # PostgreSQL: sessions, messages, facts
│   ├── llm.py              # Client Ollama + détection de langue + system prompt NOVA
│   ├── tts.py              # Piper TTS — synthèse vocale fr/en
│   └── stt.py              # Whisper STT — transcription audio→texte
│
├── agents/
│   └── __init__.py         # Classe de base BaseAgent pour les futurs agents
│
├── static/
│   └── index.html          # Interface web complète (HTML + CSS + JS vanilla)
│                           # Pas de build step, pas de npm — juste un fichier
│
├── scripts/
│   └── download_voices.py  # Télécharge les modèles Piper (fr + en)
│
├── voices/                 # Modèles TTS (créé par download_voices.py)
│   ├── fr_FR-siwis-medium.onnx
│   ├── fr_FR-siwis-medium.onnx.json
│   ├── en_US-lessac-high.onnx
│   └── en_US-lessac-high.onnx.json
│
├── requirements.txt        # Dépendances Python
├── .env.example            # Template de config → copier en .env
├── .env                    # Ta config locale (ne jamais committer!)
├── Dockerfile              # Image Docker pour déploiement serveur
├── docker-compose.yml      # Orchestration Docker (NOVA + volumes)
└── NOVA_DOC.md             # Ce fichier
```

### Détails des fichiers clés

**`main.py`**
C'est le cœur du serveur FastAPI. Il monte les routes, gère le lifespan (connexion/déconnexion PostgreSQL), et orchestre les appels entre les modules. Si tu veux ajouter un endpoint, c'est ici.

**`core/config.py`**
Toute la config passe par là via `pydantic-settings`. Lis le `.env`, valide les types, expose un objet `settings` importé partout. Pour ajouter une variable, tu l'ajoutes ici **et** dans `.env.example`.

**`core/memory.py`**
Gère toute l'interaction avec PostgreSQL. Crée les tables au démarrage si elles existent pas (`_init_schema`). Seed les faits initiaux sur toi. Expose des méthodes async pour sauvegarder/lire les messages et les faits. **Point d'extension ChromaDB** : dans `save_message()` et `upsert_fact()`, tu pourras brancher ChromaDB pour écrire les embeddings en parallèle.

**`core/llm.py`**
Client Ollama avec support streaming. Contient le system prompt NOVA (ton identité, ton contexte, les règles). La fonction `detect_language()` utilise une heuristique de mots-indicateurs français — simple mais efficace pour le québécois. Pour un futur Claude API, tu ajouteras une méthode `chat_claude()` ici.

**`core/tts.py`**
Appelle le binaire Piper via subprocess async. Prend du texte + une langue, retourne des bytes WAV. Les modèles `.onnx` doivent être dans `./voices/`. Si Piper est pas installé, l'endpoint `/speak` retourne 503 proprement.

**`core/stt.py`**
Charge Whisper lazily (au premier appel, pas au démarrage). Transcrit un fichier audio en texte + détecte la langue automatiquement. Roule dans un thread pool pour pas bloquer l'event loop asyncio.

**`agents/__init__.py`**
Classe abstraite `BaseAgent`. Pour créer un nouvel agent: subclasse ça, implémente `can_handle()` et `handle()`. Voir section [Ajouter un agent](#9-ajouter-un-agent).

**`static/index.html`**
L'UI complète en un seul fichier (HTML + CSS + JS vanilla). Esthétique Iron Man: fond très sombre, bleu cyan, rouge. Pas de framework, pas de build step. Le JS gère: streaming SSE, enregistrement audio (MediaRecorder API), lecture TTS, chargement des sessions, auto-resize du textarea.

---

## 4. Base de données

### Schéma PostgreSQL

```sql
-- Une conversation = une session de chat (identifiée par UUID)
CREATE TABLE conversations (
    id          SERIAL PRIMARY KEY,
    session_id  UUID NOT NULL UNIQUE,
    title       TEXT,                        -- auto-généré du 1er message
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Tous les messages (user + assistant) liés à une conversation
CREATE TABLE messages (
    id               SERIAL PRIMARY KEY,
    conversation_id  INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role             VARCHAR(20) NOT NULL,   -- 'user' | 'assistant' | 'system'
    content          TEXT NOT NULL,
    language         VARCHAR(10) DEFAULT 'en',  -- 'fr' | 'en'
    agent_used       VARCHAR(50),               -- futur: quel agent a répondu
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Faits persistants sur toi — mémoire long terme
CREATE TABLE facts (
    id          SERIAL PRIMARY KEY,
    category    VARCHAR(50)  NOT NULL,   -- 'profile' | 'tech' | 'project' | 'preference'
    key         VARCHAR(100) NOT NULL,
    value       TEXT NOT NULL,
    source      VARCHAR(50) DEFAULT 'user',  -- 'user' | 'system' | 'agent'
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(category, key)
);
```

### Faits seedés automatiquement

Au démarrage, si les faits n'existent pas encore:

| category  | key       | value |
|-----------|-----------|-------|
| profile   | name      | Karl Augustin |
| profile   | role      | IT Admin Tricentis + étudiant ÉTS |
| profile   | location  | Quebec, Canada |
| tech      | stack     | Python, PostgreSQL, Next.js, Kotlin, Docker, M365 |
| tech      | server    | Proxmox VE → Ubuntu VM @ 100.124.106.27 (Tailscale) |
| project   | nova      | Assistant IA personnel self-hosted |

### Ajouter des faits manuellement

```bash
curl -X POST http://localhost:8000/memory/facts \
  -H "Content-Type: application/json" \
  -d '{"category":"preference","key":"morning_routine","value":"café avant tout, pas de meetings avant 9h"}'
```

---

## 5. API — tous les endpoints

### `GET /health`
Status du système.

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "nova": "NOVA",
  "version": "1.0.0",
  "ollama": { "available": true, "model": "llama3.2", "url": "http://localhost:11434" },
  "tts": { "available": true, "voices": { "fr": true, "en": true } },
  "stt": { "available": true, "model": "base" },
  "claude_api": false
}
```

---

### `POST /chat`
Chat sans streaming (réponse complète d'un coup).

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Salut NOVA, comment ça va?", "session_id": "550e8400-e29b-41d4-a716-446655440000"}'
```
```json
{
  "response": "Salut Karl! Tout roule de mon bord. Qu'est-ce que je peux faire pour toi?",
  "language": "fr",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "conversation_id": 1
}
```

Si `session_id` est omis, un nouveau UUID est généré automatiquement.

---

### `POST /chat/stream`
Chat avec streaming SSE (tokens en temps réel). **C'est cet endpoint qu'utilise l'UI.**

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explique-moi Docker en 3 lignes", "session_id": null}' \
  --no-buffer
```

Les événements SSE arrivent dans cet ordre:
```
data: {"type":"meta","session_id":"abc-123","conversation_id":5}

data: {"type":"token","content":"Docker"}
data: {"type":"token","content":" c'est"}
data: {"type":"token","content":" une plateforme"}
...

data: {"type":"done","language":"fr"}
```

En cas d'erreur:
```
data: {"type":"error","message":"connection refused"}
```

---

### `POST /speak`
Synthèse vocale — retourne un fichier WAV.

```bash
curl -X POST http://localhost:8000/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour Karl, je suis NOVA.", "language": "fr"}' \
  --output nova.wav

# Jouer l'audio sur Mac
afplay nova.wav
```

Retourne `503` si Piper n'est pas installé.

---

### `POST /transcribe`
Transcription audio → texte (Whisper).

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@recording.wav"
```
```json
{
  "text": "Salut NOVA, c'est quoi le statut du serveur?",
  "language": "fr"
}
```

Formats supportés: wav, mp3, m4a, webm, ogg (tout ce que ffmpeg gère).

---

### `GET /memory/sessions`
Liste des 15 dernières conversations.

```bash
curl http://localhost:8000/memory/sessions
```
```json
{
  "sessions": [
    {
      "session_id": "abc-123",
      "title": "Explique-moi Docker en 3 lignes",
      "updated_at": "2026-06-05T14:23:00Z",
      "message_count": 6
    }
  ]
}
```

---

### `GET /memory/history/{session_id}`
Historique complet d'une session.

```bash
curl http://localhost:8000/memory/history/abc-123
```
```json
{
  "session_id": "abc-123",
  "conversation_id": 5,
  "messages": [
    { "role": "user", "content": "Salut!", "language": "fr", "created_at": "..." },
    { "role": "assistant", "content": "Salut Karl!", "language": "fr", "created_at": "..." }
  ]
}
```

---

### `GET /memory/facts`
Tous les faits en mémoire (optionnel: filtrer par catégorie).

```bash
curl http://localhost:8000/memory/facts
curl http://localhost:8000/memory/facts?category=project
```

---

### `POST /memory/facts`
Ajouter ou mettre à jour un fait.

```bash
curl -X POST http://localhost:8000/memory/facts \
  -H "Content-Type: application/json" \
  -d '{"category":"preference","key":"editor","value":"VS Code avec Vim mode"}'
```

---

### `DELETE /memory/facts/{category}/{key}`
Supprimer un fait.

```bash
curl -X DELETE http://localhost:8000/memory/facts/preference/editor
```

---

## 6. Variables d'environnement

Toutes les vars sont dans `.env`. Copie `.env.example` pour commencer.

| Variable | Défaut | Description |
|----------|--------|-------------|
| `APP_NAME` | `NOVA` | Nom de l'app |
| `DEBUG` | `false` | Mode debug FastAPI |
| `HOST` | `0.0.0.0` | Interface d'écoute |
| `PORT` | `8000` | Port HTTP |
| `DB_HOST` | `localhost` | Hôte PostgreSQL |
| `DB_PORT` | `5432` | Port PostgreSQL |
| `DB_NAME` | `jarvis` | Nom de la base |
| `DB_USER` | `jarvis` | Utilisateur PostgreSQL |
| `DB_PASSWORD` | — | **À changer!** |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL Ollama |
| `OLLAMA_MODEL` | `llama3.2` | Modèle LLM |
| `ANTHROPIC_API_KEY` | *(vide)* | Clé Claude API (futur) |
| `PIPER_BINARY` | `/usr/local/bin/piper` | Chemin binaire Piper |
| `PIPER_VOICES_DIR` | `./voices` | Dossier modèles vocaux |
| `PIPER_VOICE_FR` | `fr_FR-siwis-medium` | Voix française |
| `PIPER_VOICE_EN` | `en_US-lessac-high` | Voix anglaise |
| `WHISPER_MODEL` | `base` | Taille modèle Whisper |
| `CHROMA_HOST` | `localhost` | Hôte ChromaDB (futur) |

---

## 7. Installation & run local (Mac)

### Prérequis

```bash
# Python 3.11+
python3 --version

# PostgreSQL (si pas déjà installé)
brew install postgresql@16
brew services start postgresql@16

# ffmpeg (requis par Whisper)
brew install ffmpeg

# Ollama (si pas déjà installé)
brew install ollama
ollama serve &
ollama pull llama3.2
```

### Setup du projet

```bash
cd ~/nova-mac

# Environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Dépendances
pip install -r requirements.txt

# Config
cp .env.example .env
# Édite .env et mets ton mot de passe PostgreSQL
```

### Base de données

```bash
# Créer la base si elle existe pas encore
createdb -U $(whoami) jarvis
psql jarvis -c "CREATE USER jarvis WITH PASSWORD 'ton_mdp';"
psql jarvis -c "GRANT ALL PRIVILEGES ON DATABASE jarvis TO jarvis;"

# Les tables sont créées automatiquement au démarrage de NOVA
```

### Piper TTS (optionnel — pour la voix)

```bash
# Option 1: Homebrew (si disponible)
brew install piper-tts

# Option 2: Téléchargement manuel
# Va sur https://github.com/rhasspy/piper/releases
# Télécharge piper_macos_aarch64.tar.gz (Apple Silicon) ou piper_macos_x86_64.tar.gz
# Extrait et mets le binaire dans /usr/local/bin/piper

# Vérifier
piper --version

# Télécharger les modèles vocaux
python scripts/download_voices.py
```

### Lancer NOVA

```bash
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Ouvre http://localhost:8000 dans ton browser.

### Développement avec hot-reload

`--reload` recharge automatiquement le serveur quand tu changes un fichier `.py`.
L'`index.html` est servi statiquement — rafraîchis le browser pour voir les changements UI.

---

## 8. Déploiement sur le serveur (Proxmox → Ubuntu VM via Tailscale)

### Contexte

- Serveur: Ubuntu VM sur Proxmox VE 9.2
- IP Tailscale: `100.124.106.27`
- Docker déjà installé
- Ollama déjà en train de tourner sur le host (`:11434`)
- PostgreSQL déjà en train de tourner sur le host (`db: jarvis, user: jarvis`)

### Étapes

#### 1. Transférer les fichiers

```bash
# Depuis ton Mac
rsync -av --exclude='.env' --exclude='venv' --exclude='__pycache__' \
  ~/nova-mac/ karl@100.124.106.27:/opt/nova/

# OU avec scp
scp -r ~/nova-mac/ karl@100.124.106.27:/opt/nova/
```

#### 2. Configurer l'environnement sur le serveur

```bash
ssh karl@100.124.106.27
cd /opt/nova

cp .env.example .env
nano .env
```

Points importants dans le `.env` du serveur:
```bash
DB_HOST=host.docker.internal    # PostgreSQL sur le host Docker
DB_PASSWORD=ton_vrai_mdp
OLLAMA_BASE_URL=http://host.docker.internal:11434
PIPER_BINARY=/app/piper/piper   # Piper est installé dans le conteneur
```

#### 3. Télécharger les modèles vocaux sur le serveur

```bash
# Installer Python temporairement pour télécharger
sudo apt install python3 -y
python3 scripts/download_voices.py
# Les modèles sont dans ./voices/ — Docker les monte en volume
```

#### 4. Build et démarrer

```bash
docker compose build
docker compose up -d

# Vérifier les logs
docker compose logs -f nova

# Vérifier que NOVA répond
curl http://localhost:8000/health
```

#### 5. Accès via Tailscale

Depuis ton Mac ou iPhone:
```
http://100.124.106.27:8000
```

Tant que Tailscale est connecté, ça marche de n'importe où dans le monde.

### Mise à jour (après des changements)

```bash
# Depuis le serveur
cd /opt/nova
git pull  # si tu utilises git, sinon rsync depuis le Mac

docker compose build nova
docker compose up -d nova
docker compose logs -f nova
```

### Reverse proxy Nginx (optionnel — pour HTTPS)

```nginx
server {
    listen 80;
    server_name nova.local;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        # Requis pour le streaming SSE
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
```

---

## 9. Ajouter un agent

Les agents permettent de spécialiser NOVA pour des tâches précises. Exemple: un agent étude qui connaît tes cours à l'ÉTS, un agent travail qui connaît l'infrastructure Tricentis.

### Structure d'un agent

```python
# agents/study_agent.py
from agents import BaseAgent
from typing import Dict


class StudyAgent(BaseAgent):
    name = "study"
    description = "Aide pour les cours à l'ÉTS Montréal"
    icon = "📚"

    TRIGGER_KEYWORDS = [
        "cours", "devoir", "exam", "examen", "prof",
        "ÉTS", "génie", "LOG", "MEC", "laboratoire",
    ]

    async def can_handle(self, message: str, context: Dict) -> bool:
        lower = message.lower()
        return any(kw in lower for kw in self.TRIGGER_KEYWORDS)

    async def handle(self, message: str, context: Dict) -> str:
        # Tu peux override le system prompt ici
        return await self.llm.chat(
            message,
            context.get("history", []),
            context.get("facts", []),
            agent_system_prompt=self.get_system_prompt(),
        )

    def get_system_prompt(self) -> str:
        return """Tu es NOVA en mode Étude.
        Karl est étudiant en génie logiciel à l'ÉTS Montréal.
        Aide-le avec ses cours, exercices, et projets académiques.
        Sois pédagogique mais pas condescendant — Karl est capable."""
```

### Enregistrer l'agent dans main.py

```python
# Dans main.py, ajoute:
from agents.study_agent import StudyAgent

# Crée la liste des agents (après les imports)
AGENTS: list[BaseAgent] = [
    StudyAgent(llm_client, memory_manager),
    # WorkAgent(llm_client, memory_manager),
    # ProjectAgent(llm_client, memory_manager),
]

# Dans l'endpoint /chat/stream, avant d'appeler llm_client:
active_agent = None
for agent in AGENTS:
    if await agent.can_handle(req.message, {"history": history, "facts": facts}):
        active_agent = agent
        break

if active_agent:
    response = await active_agent.handle(req.message, {"history": history, "facts": facts})
else:
    # fallback: LLM généraliste
    response = await llm_client.chat(req.message, history, facts)
```

### Agents prévus

| Agent | Triggers | Description |
|-------|----------|-------------|
| `StudyAgent` | cours, devoir, exam, ÉTS | Aide académique |
| `WorkAgent` | Tricentis, M365, ticket, PowerShell | Support IT Tricentis |
| `ProjectAgent` | BudgetApp, homelab, NOVA, Docker | Aide projets perso |
| `ResearchAgent` | recherche, trouve, article, compare | Recherche web (via Claude API) |

---

## 10. Roadmap

### Phase 2 — Agents (prochain sprint)

- [ ] `StudyAgent` — contexte cours ÉTS, syllabus, dates d'examen
- [ ] `WorkAgent` — runbooks IT Tricentis, gestion M365
- [ ] `ProjectAgent` — statut BudgetApp, commits récents, TODO
- [ ] Router d'agents dans main.py

### Phase 3 — Mémoire vectorielle

- [ ] Installer ChromaDB dans Docker
- [ ] Embeddings des messages via Ollama (`nomic-embed-text`)
- [ ] Recherche sémantique dans l'historique via `/memory/search?q=...`
- [ ] Brancher ChromaDB dans `core/memory.py`

### Phase 4 — Claude API (tâches complexes)

- [ ] Ajouter `anthropic` à `requirements.txt`
- [ ] Méthode `chat_claude()` dans `core/llm.py`
- [ ] Détection automatique: tâche simple → Ollama, tâche complexe → Claude
- [ ] `ResearchAgent` avec web search (Claude + tool use)

### Phase 5 — Voice wake word

- [ ] Intégration `pvporcupine` (Picovoice) ou `openWakeWord`
- [ ] Daemon Python qui écoute en background sur le serveur
- [ ] "Hey NOVA" → démarre l'enregistrement automatiquement
- [ ] Mode conversation continue (pas besoin de cliquer)

### Phase 6 — Mobile

- [ ] PWA: ajouter un manifest.json + service worker pour installer sur iPhone
- [ ] Push notifications (pour rappels, alertes)
- [ ] Widget iOS via Raccourcis Apple (appelle l'API NOVA)

### Phase 7 — Intégrations

- [ ] Google Calendar (voir agenda, créer des événements)
- [ ] Microsoft 365 (emails, Teams) via Microsoft Graph API
- [ ] GitHub (statut PR, issues, commits récents)
- [ ] Home Assistant (contrôle domotique)

---

## Notes de déploiement

**PostgreSQL existant vs nouveau**

Le `docker-compose.yml` est configuré pour utiliser ton PostgreSQL existant sur le host via `host.docker.internal`. Les tables sont créées automatiquement au premier démarrage de NOVA.

**Ollama sur le host**

Ollama tourne déjà sur le host. NOVA dans Docker s'y connecte via `http://host.docker.internal:11434`. Si tu veux que Ollama soit accessible depuis l'extérieur, assure-toi qu'il écoute sur `0.0.0.0:11434` (pas juste localhost).

**Logs**

```bash
# Voir les logs NOVA en temps réel
docker compose logs -f nova

# Logs PostgreSQL (si service Docker)
docker compose logs -f db
```

**Backup de la base**

```bash
# Sur le serveur
pg_dump -U jarvis jarvis > nova_backup_$(date +%Y%m%d).sql

# Restore
psql -U jarvis jarvis < nova_backup_20260605.sql
```

---

*NOVA v1.0.0 — Construit par Karl avec Claude Code*
*"C'est pas de la magie, c'est juste de la bonne ingénierie."*
