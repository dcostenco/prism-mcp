-- Migration 038: Dark Factory Pipelines

BEGIN;

CREATE TABLE IF NOT EXISTS public.dark_factory_pipelines (
  id TEXT PRIMARY KEY,
  project TEXT NOT NULL,
  user_id TEXT NOT NULL DEFAULT 'default',
  status TEXT NOT NULL,
  current_step TEXT NOT NULL,
  iteration INTEGER NOT NULL,
  started_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  spec TEXT NOT NULL,
  error TEXT,
  last_heartbeat TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pipelines_status ON public.dark_factory_pipelines(user_id, project, status);

ALTER TABLE public.dark_factory_pipelines ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all operations for authenticated and service roles"
  ON public.dark_factory_pipelines
  AS PERMISSIVE
  FOR ALL
  USING (true);

COMMIT;
