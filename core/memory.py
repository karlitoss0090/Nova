import uuid
from typing import List, Dict, Optional
import asyncpg

from core.config import settings


class MemoryManager:
    """
    PostgreSQL-backed memory for NOVA.

    Tables:
      conversations — one row per chat session (UUID key)
      messages      — all user/assistant turns linked to a conversation
      facts         — persistent key-value facts about Karl and his world

    Future extension point: plug ChromaDB in upsert_fact() and save_message()
    to also write embeddings for semantic search.
    """

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            min_size=2,
            max_size=10,
        )
        await self._init_schema()

    async def disconnect(self):
        if self.pool:
            await self.pool.close()

    # ── Schema ────────────────────────────────────────────────────────────

    async def _init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          SERIAL PRIMARY KEY,
                    session_id  UUID NOT NULL UNIQUE,
                    title       TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id               SERIAL PRIMARY KEY,
                    conversation_id  INTEGER NOT NULL
                                         REFERENCES conversations(id)
                                         ON DELETE CASCADE,
                    role             VARCHAR(20) NOT NULL,   -- 'user' | 'assistant' | 'system'
                    content          TEXT NOT NULL,
                    language         VARCHAR(10) DEFAULT 'en',
                    agent_used       VARCHAR(50),            -- future: which agent answered
                    created_at       TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id          SERIAL PRIMARY KEY,
                    category    VARCHAR(50)  NOT NULL,   -- 'profile' | 'preference' | 'project' | ...
                    key         VARCHAR(100) NOT NULL,
                    value       TEXT NOT NULL,
                    source      VARCHAR(50) DEFAULT 'user',  -- 'user' | 'system' | 'agent'
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(category, key)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv  ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_messages_time  ON messages(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
            """)
            await self._seed_facts(conn)

    async def _seed_facts(self, conn):
        """Seed initial facts about Karl if not already present."""
        seeds = [
            ("profile", "name",       "Karl Augustin"),
            ("profile", "role",       "IT Administrator at Tricentis + Software Engineering student at ÉTS Montréal"),
            ("profile", "location",   "Quebec, Canada"),
            ("profile", "languages",  "Quebec French (informal) and English"),
            ("tech",    "stack",      "Python, PostgreSQL, Next.js, Android/Kotlin, Docker, Microsoft 365"),
            ("tech",    "server",     "Proxmox VE 9.2 → Ubuntu VM (docker-host) @ 100.124.106.27 via Tailscale"),
            ("tech",    "running",    "Ollama + llama3.2, PostgreSQL (db: jarvis), Open WebUI"),
            ("project", "nova",       "Personal AI assistant — self-hosted Jarvis on home server"),
            ("project", "budgetapp",  "Personal budget tracking application"),
        ]
        for category, key, value in seeds:
            await conn.execute(
                """
                INSERT INTO facts (category, key, value, source)
                VALUES ($1, $2, $3, 'system')
                ON CONFLICT (category, key) DO NOTHING
                """,
                category, key, value,
            )

    # ── Sessions ──────────────────────────────────────────────────────────

    async def get_or_create_session(self, session_id: str) -> int:
        sid = uuid.UUID(session_id)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM conversations WHERE session_id = $1", sid
            )
            if row:
                await conn.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = $1", row["id"]
                )
                return row["id"]
            row = await conn.fetchrow(
                "INSERT INTO conversations (session_id) VALUES ($1) RETURNING id", sid
            )
            return row["id"]

    async def update_session_title(self, conversation_id: int, title: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET title = $1 WHERE id = $2", title, conversation_id
            )

    async def get_recent_sessions(self, limit: int = 15) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.session_id,
                    COALESCE(c.title,
                        LEFT((SELECT content FROM messages
                              WHERE conversation_id = c.id AND role = 'user'
                              ORDER BY created_at LIMIT 1), 60),
                        'New conversation') AS title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [dict(r) for r in rows]

    # ── Messages ──────────────────────────────────────────────────────────

    async def save_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        language: str = "en",
        agent_used: Optional[str] = None,
    ):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, language, agent_used)
                VALUES ($1, $2, $3, $4, $5)
                """,
                conversation_id, role, content, language, agent_used,
            )

    async def get_conversation_history(
        self, conversation_id: int, limit: int = 20
    ) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, language, agent_used, created_at
                FROM messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                conversation_id, limit,
            )
        return [dict(r) for r in rows]

    # ── Facts ─────────────────────────────────────────────────────────────

    async def get_facts(self, category: Optional[str] = None) -> List[Dict]:
        async with self.pool.acquire() as conn:
            if category:
                rows = await conn.fetch(
                    "SELECT * FROM facts WHERE category = $1 ORDER BY key", category
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM facts ORDER BY category, key"
                )
        return [dict(r) for r in rows]

    async def upsert_fact(
        self,
        category: str,
        key: str,
        value: str,
        source: str = "user",
    ):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO facts (category, key, value, source)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (category, key)
                DO UPDATE SET value = $3, source = $4, updated_at = NOW()
                """,
                category, key, value, source,
            )

    async def delete_fact(self, category: str, key: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM facts WHERE category = $1 AND key = $2", category, key
            )


memory_manager = MemoryManager()
