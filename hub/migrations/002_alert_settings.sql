-- Per-owner ntfy topic settings stored in Postgres so no env var setup is required

create table if not exists fleet_alert_settings (
  owner_key text primary key,
  ntfy_topic text not null,
  enabled boolean not null default true,
  updated_at timestamptz not null default now()
);
