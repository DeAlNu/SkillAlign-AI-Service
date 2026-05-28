-- ============================================================
-- SkillAlign AI Service — Supabase Tables untuk Learning Path
-- Jalankan di: Supabase Dashboard → SQL Editor
-- ============================================================

-- ── Tabel 1: skill_courses (cache hasil fetch Gemini + YouTube) ──────────
CREATE TABLE IF NOT EXISTS skill_courses (
    id                BIGSERIAL PRIMARY KEY,
    skill_normalized  TEXT        NOT NULL,   -- e.g. "typescript", "docker"
    platform          TEXT        NOT NULL,   -- "coursera" | "free_resources" | "youtube"
    title             TEXT        NOT NULL,
    url               TEXT        NOT NULL,
    thumbnail_url     TEXT,
    level             TEXT,                   -- "Beginner" | "Intermediate" | dll
    is_valid          BOOLEAN     NOT NULL DEFAULT TRUE,  -- FALSE = placeholder "no results"
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at        TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days')
);

-- Index untuk query cepat per skill + platform
CREATE INDEX IF NOT EXISTS idx_skill_courses_lookup
    ON skill_courses (skill_normalized, platform, is_valid, expires_at);

-- ── Tabel 2: learning_path_sessions (log sesi user) ──────────────────────
CREATE TABLE IF NOT EXISTS learning_path_sessions (
    id                BIGSERIAL PRIMARY KEY,
    user_id           TEXT,                   -- nullable, opsional
    target_job_title  TEXT        NOT NULL,
    missing_skills    JSONB       NOT NULL DEFAULT '[]',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index untuk query per user
CREATE INDEX IF NOT EXISTS idx_lp_sessions_user
    ON learning_path_sessions (user_id, created_at DESC);

-- ── Row Level Security (opsional, pakai service_role key → bypass RLS) ───
-- ALTER TABLE skill_courses ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE learning_path_sessions ENABLE ROW LEVEL SECURITY;

-- ── Verifikasi ────────────────────────────────────────────────────────────
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('skill_courses', 'learning_path_sessions');
