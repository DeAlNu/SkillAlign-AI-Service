-- ============================================================
-- SkillAlign Learning Path — Supabase Schema
-- Jalankan di: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- Tabel 1: Cache kursus per skill per platform
CREATE TABLE IF NOT EXISTS skill_courses (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_normalized TEXT NOT NULL,           -- "typescript", "docker" (lowercase, snake_case)
    platform        TEXT NOT NULL             -- 'coursera' | 'free_resources' | 'youtube'
                    CHECK (platform IN ('coursera', 'free_resources', 'youtube')),
    title           TEXT NOT NULL,
    url             TEXT NOT NULL,
    thumbnail_url   TEXT,
    level           TEXT,                     -- 'Beginner', 'Intermediate', 'Pemula', dll
    is_valid        BOOLEAN DEFAULT true,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (skill_normalized, platform, url)
);

-- Index untuk lookup cepat by skill + platform
CREATE INDEX IF NOT EXISTS idx_skill_courses_lookup
    ON skill_courses (skill_normalized, platform, created_at DESC);

-- Tabel 2: Session learning path per user (opsional, untuk riwayat)
CREATE TABLE IF NOT EXISTS learning_path_sessions (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               TEXT,
    target_job_title      TEXT NOT NULL,
    missing_skills        JSONB NOT NULL,     -- array of {skill, tier, frequency}
    created_at            TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lp_sessions_user
    ON learning_path_sessions (user_id, created_at DESC);

-- Enable Row Level Security (RLS) — service_role key bypass otomatis
ALTER TABLE skill_courses          ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_path_sessions ENABLE ROW LEVEL SECURITY;
