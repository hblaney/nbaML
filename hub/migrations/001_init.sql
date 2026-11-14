-- Additive Postgres schema for Cursor Session Hub (tmux + Cloud Agents + permanent memory)

create table if not exists fleet_sessions (
  id text primary key,
  owner_key text not null,
  slug text not null,
  title text not null,
  role text not null default 'worker',
  kind text not null default 'tmux',
  status text not null default 'idle',
  tmux_target text not null default '',
  workdir text not null default '/workspace',
  branch text not null default '',
  cloud_agent_id text not null default '',
  repo_url text not null default '',
  pane_snapshot text not null default '',
  is_sample boolean not null default false,
  last_heartbeat_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_fleet_sessions_owner_updated
  on fleet_sessions (owner_key, updated_at desc);

create unique index if not exists idx_fleet_sessions_owner_slug
  on fleet_sessions (owner_key, slug);

create table if not exists fleet_messages (
  id text primary key,
  owner_key text not null,
  sender_kind text not null,
  from_session_id text,
  from_label text not null,
  to_session_id text,
  to_label text not null,
  body text not null,
  delivery_mode text not null default 'logged',
  delivery_status text not null default 'delivered',
  metadata_json text not null default '{}',
  is_sample boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists idx_fleet_messages_owner_created
  on fleet_messages (owner_key, created_at asc);

create index if not exists idx_fleet_messages_owner_to_session
  on fleet_messages (owner_key, to_session_id, created_at asc);

create index if not exists idx_fleet_messages_owner_from_session
  on fleet_messages (owner_key, from_session_id, created_at asc);

create index if not exists idx_fleet_messages_fts
  on fleet_messages using gin (to_tsvector('english', coalesce(from_label, '') || ' ' || coalesce(to_label, '') || ' ' || coalesce(body, '')));

create table if not exists fleet_tmux_outbox (
  id text primary key,
  owner_key text not null,
  session_id text not null,
  tmux_target text not null,
  action text not null,
  payload_text text not null default '',
  workdir text not null default '/workspace',
  status text not null default 'pending',
  created_at timestamptz not null default now(),
  delivered_at timestamptz
);

create index if not exists idx_fleet_outbox_owner_status
  on fleet_tmux_outbox (owner_key, status, created_at asc);
