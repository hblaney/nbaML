"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRightLeft,
  Cloud,
  Copy,
  Check,
  Database,
  History,
  MessageSquare,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Terminal,
  Zap,
} from "lucide-react";

type SessionItem = {
  id: string;
  slug: string;
  title: string;
  role: string;
  kind: "tmux" | "cloud";
  status: string;
  tmux_target: string;
  workdir: string;
  branch: string;
  cloud_agent_id: string;
  repo_url: string;
  pane_snapshot: string;
  is_sample: boolean;
  last_heartbeat_at: string | null;
  updated_at: string;
};

type MessageItem = {
  id: string;
  sender_kind: "user" | "session" | "system";
  from_session_id: string | null;
  from_label: string;
  to_session_id: string | null;
  to_label: string;
  body: string;
  delivery_mode: string;
  delivery_status: string;
  is_sample: boolean;
  created_at: string;
};

type BootstrapPayload = {
  viewer: {
    name: string;
    email: string;
    simulated: boolean;
    stage: "production" | "preview";
  };
  runtime: {
    cloudApiConfigured: boolean;
    managedPostgres: boolean;
    pendingOutbox: number;
  };
  sessions: SessionItem[];
  messages: MessageItem[];
};

function formatClock(iso: string | null): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

export default function FleetHubClient() {
  const [data, setData] = useState<BootstrapPayload | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [selectedId, setSelectedId] = useState<string>("bus");
  const [mobileTab, setMobileTab] = useState<
    "sessions" | "chat" | "prizepicks" | "terminal" | "search"
  >("chat");
  const [rightTab, setRightTab] = useState<"prizepicks" | "terminal" | "search" | "setup">(
    "prizepicks",
  );

  const [draft, setDraft] = useState("");
  const [sendMode, setSendMode] = useState<"user" | "session">("user");
  const [fromSessionSlug, setFromSessionSlug] = useState<string>("coordinator");
  const [sending, setSending] = useState(false);

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchScope, setSearchScope] = useState<"all" | "inter_session" | "user">("all");
  const [searchResults, setSearchResults] = useState<MessageItem[]>([]);
  const [searching, setSearching] = useState(false);

  // Spawn modal state
  const [showSpawn, setShowSpawn] = useState(false);
  const [spawnTitle, setSpawnTitle] = useState("");
  const [spawnKind, setSpawnKind] = useState<"tmux" | "cloud">("tmux");
  const [spawnWorkdir, setSpawnWorkdir] = useState("/workspace");
  const [spawnBranch, setSpawnBranch] = useState("");
  const [spawnRepo, setSpawnRepo] = useState("https://github.com/hblaney/nbaML");
  const [spawnPrompt, setSpawnPrompt] = useState("");
  const [spawning, setSpawning] = useState(false);
  const [repoList, setRepoList] = useState<Array<{ name: string; url: string }>>([]);
  const [autopsyData, setAutopsyData] = useState<{
    configured: boolean;
    ntfyTopic?: string;
    channels: string[];
    recentAutopsies: Array<{
      date: string;
      slip: string;
      legs: Array<{ player: string; pick: string; actual: string; tag: string; note: string }>;
    }>;
  } | null>(null);

  // 24/7 PrizePicks Autopilot live state
  const [ppLoading, setPpLoading] = useState(false);
  const [ppData, setPpData] = useState<{
    scannedAt?: string;
    upcomingGames?: number;
    bestSlipType?: string;
    bestSlipLegs?: Array<{
      gameId: string;
      matchup: string;
      player: string;
      team: string;
      stat: string;
      side: "MORE" | "LESS";
      ppLine: number;
      dkLine: number;
      dkOdds: string;
      pCalibrated: number;
    }>;
    allTopEdges?: Array<{
      gameId: string;
      matchup: string;
      player: string;
      team: string;
      stat: string;
      side: "MORE" | "LESS";
      ppLine: number;
      dkLine: number;
      dkOdds: string;
      pCalibrated: number;
    }>;
    rdUpgrades?: Array<{ id: string; status: string; title: string; summary: string }>;
  } | null>(null);

  const loadPrizePicksLive = useCallback(async () => {
    setPpLoading(true);
    try {
      const resp = await fetch("/api/prizepicks", { cache: "no-store" });
      if (resp.ok) {
        setPpData(await resp.json());
      }
    } finally {
      setPpLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPrizePicksLive();
    fetch("/api/cloud-repos", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (j?.repositories) setRepoList(j.repositories);
      })
      .catch(() => {});
    fetch("/api/autopsies", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (j) setAutopsyData(j);
      })
      .catch(() => {});
  }, [loadPrizePicksLive]);

  const [copiedCmd, setCopiedCmd] = useState(false);
  const [pingNote, setPingNote] = useState("");
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const refresh = useCallback(async () => {
    try {
      const resp = await fetch("/api/bootstrap", { cache: "no-store" });
      if (resp.status === 401 || resp.status === 403) {
        setAuthError("Sign in through Cursor to access your private Fleet Hub.");
        setLoading(false);
        return;
      }
      if (!resp.ok) {
        setLoading(false);
        return;
      }
      const json = (await resp.json()) as BootstrapPayload;
      setAuthError(null);
      setData(json);
      setSelectedId((prev) => {
        if (prev === "bus") {
          return json.sessions[0]?.id ?? "bus";
        }
        const exists = json.sessions.some((s) => s.id === prev);
        return exists ? prev : json.sessions[0]?.id ?? "bus";
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => {
      void refresh();
    }, 2500);
    return () => clearInterval(timer);
  }, [refresh]);

  const activeSession = useMemo(() => {
    if (!data || selectedId === "bus") return null;
    return data.sessions.find((s) => s.id === selectedId) ?? null;
  }, [data, selectedId]);

  const visibleMessages = useMemo(() => {
    if (!data) return [];
    if (selectedId === "bus") {
      return data.messages.filter(
        (m) =>
          m.sender_kind === "session" &&
          m.from_session_id !== null &&
          m.to_session_id !== null,
      );
    }
    return data.messages.filter(
      (m) =>
        m.to_session_id === selectedId ||
        m.from_session_id === selectedId ||
        (activeSession &&
          (m.to_label === activeSession.slug || m.from_label === activeSession.slug)),
    );
  }, [data, selectedId, activeSession]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visibleMessages.length, selectedId]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!draft.trim() || sending) return;
    const targetSession =
      activeSession ?? data?.sessions.find((s) => s.role === "coordinator") ?? data?.sessions[0];
    if (!targetSession) return;

    setSending(true);
    try {
      const resp = await fetch("/api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          senderKind: sendMode,
          fromSessionId: sendMode === "session" ? fromSessionSlug : undefined,
          fromLabel: sendMode === "session" ? fromSessionSlug : undefined,
          toSessionId: targetSession.id,
          toLabel: targetSession.slug,
          body: draft.trim(),
        }),
      });
      if (resp.ok) {
        setDraft("");
        await refresh();
      }
    } finally {
      setSending(false);
    }
  };

  const runSearch = useCallback(
    async (q: string, scope: "all" | "inter_session" | "user") => {
      if (!q.trim()) {
        setSearchResults([]);
        return;
      }
      setSearching(true);
      try {
        const params = new URLSearchParams({ q: q.trim(), scope });
        const resp = await fetch(`/api/search?${params.toString()}`, {
          cache: "no-store",
        });
        if (resp.ok) {
          const json = (await resp.json()) as { results: MessageItem[] };
          setSearchResults(json.results ?? []);
        }
      } finally {
        setSearching(false);
      }
    },
    [],
  );

  useEffect(() => {
    const t = setTimeout(() => {
      void runSearch(searchQuery, searchScope);
    }, 220);
    return () => clearTimeout(t);
  }, [searchQuery, searchScope, runSearch]);

  const handleSpawn = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!spawnTitle.trim() || spawning) return;
    setSpawning(true);
    try {
      const resp = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: spawnTitle.trim(),
          kind: spawnKind,
          workdir: spawnWorkdir.trim() || "/workspace",
          branch: spawnBranch.trim(),
          repoUrl: spawnRepo.trim(),
          initialPrompt: spawnPrompt.trim(),
        }),
      });
      if (resp.ok) {
        const created = (await resp.json()) as { session: SessionItem };
        setShowSpawn(false);
        setSpawnTitle("");
        setSpawnPrompt("");
        await refresh();
        if (created.session?.id) {
          setSelectedId(created.session.id);
          setMobileTab("chat");
        }
      }
    } finally {
      setSpawning(false);
    }
  };

  const injectMemoryIntoDraft = (m: MessageItem) => {
    const snippet = `[Recalled from Postgres history (${m.from_label} -> ${m.to_label})]: "${m.body}" — `;
    setDraft((prev) => (prev ? `${prev}\n${snippet}` : snippet));
    setMobileTab("chat");
  };

  const bridgeCommand =
    "python3 -m tools.cursor_fleet.cli daemon --hub-url http://127.0.0.1:3000";

  if (loading && !data) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-[#090b10] text-zinc-400">
        <div className="flex items-center gap-3 text-sm">
          <RefreshCw className="h-4 w-4 animate-spin text-indigo-400" />
          Loading Cursor Fleet Hub...
        </div>
      </div>
    );
  }

  if (authError) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-[#090b10] p-6">
        <div className="max-w-md rounded-2xl border border-zinc-800 bg-zinc-900/80 p-6 text-center">
          <ShieldCheck className="mx-auto mb-3 h-8 w-8 text-indigo-400" />
          <h1 className="text-lg font-semibold text-zinc-100">Cursor Sign-In Required</h1>
          <p className="mt-2 text-sm text-zinc-400">{authError}</p>
        </div>
      </div>
    );
  }

  const sessions = data?.sessions ?? [];
  const interSessionCount =
    data?.messages.filter(
      (m) =>
        m.sender_kind === "session" &&
        m.from_session_id !== null &&
        m.to_session_id !== null,
    ).length ?? 0;

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-[#090b10]">
      {/* Top Header Bar */}
      <header className="flex shrink-0 items-center justify-between border-b border-zinc-800/80 bg-[#0d1017] px-3 py-2.5 sm:px-5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-500/15 ring-1 ring-indigo-500/30">
            <Zap className="h-4 w-4 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold tracking-tight text-zinc-100">
                Cursor Fleet Hub
              </span>
              <span className="hidden rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-400 ring-1 ring-emerald-500/25 sm:inline-flex items-center gap-1">
                <Database className="h-3 w-3" />
                {data?.runtime.managedPostgres ? "Postgres Live" : "PGlite Local SQL"}
              </span>
              {data?.viewer.simulated && (
                <span
                  title="Local dev server uses simulated identity from cursor/identity.dev.ts"
                  className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-300 ring-1 ring-amber-500/25"
                >
                  Simulated Dev Viewer
                </span>
              )}
            </div>
            <p className="hidden text-[11px] text-zinc-400 sm:block">
              Zero-polling tmux push + Cloud Agent steering + permanent searchable Postgres bus
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              setRightTab("search");
              setMobileTab("search");
            }}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900/90 px-2.5 py-1.5 text-xs text-zinc-300 hover:border-zinc-700 hover:text-white"
          >
            <Search className="h-3.5 w-3.5 text-indigo-400" />
            <span className="hidden sm:inline">Search Memory</span>
          </button>
          <button
            type="button"
            onClick={() => setShowSpawn(true)}
            className="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white shadow-sm hover:bg-indigo-500"
          >
            <Plus className="h-3.5 w-3.5" />
            <span>New Session</span>
          </button>
        </div>
      </header>

      {/* Main 3-Column Layout (Desktop) / Tabbed View (Phone) */}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* LEFT PANE: Session Fleet + Inter-Session Bus */}
        <aside
          className={`${
            mobileTab === "sessions" ? "flex" : "hidden"
          } w-full shrink-0 flex-col border-r border-zinc-800/80 bg-[#0b0e14] md:flex md:w-72`}
        >
          <div className="flex items-center justify-between px-3.5 pt-3 pb-2">
            <span className="text-[11px] font-semibold tracking-wider text-zinc-400 uppercase">
              Active Sessions ({sessions.length})
            </span>
            {data?.runtime.pendingOutbox ? (
              <span className="rounded-full bg-indigo-500/15 px-2 py-0.5 text-[10px] text-indigo-300">
                {data.runtime.pendingOutbox} queued
              </span>
            ) : null}
          </div>

          <div className="flex-1 space-y-1 overflow-y-auto px-2.5 pb-3">
            {sessions.map((s) => {
              const isSelected = selectedId === s.id;
              const isLive = s.status === "active" || s.status === "running";
              return (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => {
                    setSelectedId(s.id);
                    setMobileTab("chat");
                  }}
                  className={`w-full rounded-xl p-2.5 text-left transition ${
                    isSelected
                      ? "bg-indigo-500/15 ring-1 ring-indigo-500/40"
                      : "hover:bg-zinc-900/80"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span
                        className={`h-2 w-2 shrink-0 rounded-full ${
                          isLive ? "bg-emerald-400 shadow-[0_0_8px_#34d399]" : "bg-zinc-500"
                        }`}
                      />
                      <span className="truncate text-xs font-semibold text-zinc-100">
                        {s.title}
                      </span>
                    </div>
                    <span className="shrink-0 rounded bg-zinc-800/90 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                      {s.kind === "cloud" ? "cloud" : "tmux"}
                    </span>
                  </div>

                  <div className="mt-1.5 flex items-center justify-between text-[11px] text-zinc-400">
                    <span className="truncate font-mono text-[10px] text-zinc-500">
                      {s.kind === "tmux" ? s.tmux_target || `${s.slug}:0.0` : s.cloud_agent_id || "cloud-agent"}
                    </span>
                    <div className="flex items-center gap-1">
                      {s.role === "coordinator" && (
                        <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] font-medium text-indigo-300">
                          lead
                        </span>
                      )}
                      {s.is_sample && (
                        <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
                          sample
                        </span>
                      )}
                    </div>
                  </div>
                </button>
              );
            })}

            <div className="pt-2">
              <div className="px-1 pb-1.5 text-[11px] font-semibold tracking-wider text-zinc-400 uppercase">
                Agent Mesh
              </div>
              <button
                type="button"
                onClick={() => {
                  setSelectedId("bus");
                  setMobileTab("chat");
                }}
                className={`w-full rounded-xl p-2.5 text-left transition ${
                  selectedId === "bus"
                    ? "bg-amber-500/15 ring-1 ring-amber-500/40"
                    : "hover:bg-zinc-900/80"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-xs font-semibold text-amber-200">
                    <ArrowRightLeft className="h-3.5 w-3.5 text-amber-400" />
                    Inter-Session Bus
                  </span>
                  <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-medium text-amber-300">
                    {interSessionCount}
                  </span>
                </div>
                <p className="mt-1 text-[11px] text-zinc-400">
                  Every message sessions send to each other
                </p>
              </button>
            </div>
          </div>
        </aside>

        {/* CENTER PANE: Real-Time Push Chat & Inter-Session Stream */}
        <main
          className={`${
            mobileTab === "chat" ? "flex" : "hidden"
          } min-w-0 flex-1 flex-col bg-[#090b10] md:flex`}
        >
          {/* Active Session Subheader */}
          <div className="flex items-center justify-between border-b border-zinc-800/80 bg-[#0c0f16] px-4 py-2.5">
            <div className="min-w-0">
              {selectedId === "bus" ? (
                <div className="flex items-center gap-2">
                  <ArrowRightLeft className="h-4 w-4 text-amber-400" />
                  <h2 className="truncate text-sm font-semibold text-zinc-100">
                    Inter-Session Message Bus
                  </h2>
                </div>
              ) : activeSession ? (
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="truncate text-sm font-semibold text-zinc-100">
                      {activeSession.title}
                    </h2>
                    <span className="rounded-md bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300">
                      @{activeSession.slug}
                    </span>
                  </div>
                  <p className="truncate text-[11px] text-zinc-400">
                    {activeSession.kind === "tmux"
                      ? `Pushes straight into tmux (${activeSession.tmux_target || `${activeSession.slug}:0.0`}) — zero tool polling`
                      : `Steers Cursor Cloud Agent (${activeSession.cloud_agent_id || "API ready"})`}
                  </p>
                </div>
              ) : null}
            </div>

            {activeSession && (
              <button
                type="button"
                onClick={() => {
                  setRightTab("terminal");
                  setMobileTab("terminal");
                }}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-xs text-zinc-300 hover:border-zinc-700 lg:hidden"
              >
                <Terminal className="h-3.5 w-3.5 text-emerald-400" />
                <span>Pane</span>
              </button>
            )}
          </div>

          {/* Messages Feed */}
          <div className="flex-1 space-y-3 overflow-y-auto px-3.5 py-4 sm:px-5">
            {visibleMessages.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-zinc-800 p-8 text-center text-sm text-zinc-500">
                No messages in this view yet. Send a message below — it logs permanently to Postgres and pushes straight into the session.
              </div>
            ) : (
              visibleMessages.map((m) => {
                const isUser = m.sender_kind === "user";
                const isSessionToSession =
                  m.sender_kind === "session" &&
                  m.from_session_id !== null &&
                  m.to_session_id !== null;

                return (
                  <div
                    key={m.id}
                    className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}
                  >
                    <div
                      className={`max-w-[90%] sm:max-w-[78%] rounded-2xl px-3.5 py-2.5 text-sm shadow-sm ${
                        isUser
                          ? "bg-indigo-600 text-white"
                          : isSessionToSession
                            ? "border border-amber-500/30 bg-amber-500/10 text-zinc-100"
                            : "border border-zinc-800 bg-zinc-900/90 text-zinc-100"
                      }`}
                    >
                      <div className="mb-1 flex flex-wrap items-center gap-1.5 text-[11px] opacity-85">
                        <span className="font-semibold">{m.from_label}</span>
                        <span>→</span>
                        <span className="font-mono">{m.to_label}</span>
                        {isSessionToSession && (
                          <span className="rounded bg-amber-500/20 px-1.5 py-0.2 text-[10px] text-amber-300">
                            session↔session
                          </span>
                        )}
                        <span className="ml-auto text-[10px] opacity-70">
                          {formatClock(m.created_at)}
                        </span>
                      </div>
                      <p className="whitespace-pre-wrap leading-relaxed">{m.body}</p>
                      <div className="mt-1.5 flex items-center justify-end gap-2 text-[10px] opacity-75">
                        <span>{m.delivery_mode}</span>
                        <span>•</span>
                        <span>{m.delivery_status}</span>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Sticky Mobile & Desktop Composer */}
          <form
            onSubmit={handleSend}
            className="border-t border-zinc-800/80 bg-[#0c0f16] p-3 sm:px-4"
          >
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setSendMode("user")}
                  className={`rounded-md px-2 py-0.5 text-[11px] font-medium ${
                    sendMode === "user"
                      ? "bg-indigo-500/20 text-indigo-300 ring-1 ring-indigo-500/40"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  Send as You
                </button>
                <button
                  type="button"
                  onClick={() => setSendMode("session")}
                  className={`rounded-md px-2 py-0.5 text-[11px] font-medium ${
                    sendMode === "session"
                      ? "bg-amber-500/20 text-amber-300 ring-1 ring-amber-500/40"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  Session → Session Handoff
                </button>
                {sendMode === "session" && (
                  <select
                    value={fromSessionSlug}
                    onChange={(e) => setFromSessionSlug(e.target.value)}
                    className="rounded border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-200"
                  >
                    {sessions.map((s) => (
                      <option key={s.id} value={s.slug}>
                        from @{s.slug}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <button
                type="button"
                onClick={() =>
                  setDraft(
                    "Search our Postgres message history for what we decided earlier and summarize it.",
                  )
                }
                className="hidden items-center gap-1 text-[11px] text-zinc-400 hover:text-indigo-300 sm:inline-flex"
              >
                <Sparkles className="h-3 w-3" />
                Ask session to search Postgres memory
              </button>
            </div>

            <div className="flex items-end gap-2">
              <textarea
                rows={2}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleSend(e);
                  }
                }}
                placeholder={
                  activeSession
                    ? `Push message into @${activeSession.slug} (Enter to send, Shift+Enter newline)...`
                    : "Push message into @coordinator..."
                }
                className="min-h-[44px] max-h-36 flex-1 resize-none rounded-xl border border-zinc-800 bg-zinc-900/90 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
              />
              <button
                type="submit"
                disabled={sending || !draft.trim()}
                className="flex h-[44px] shrink-0 items-center justify-center gap-1.5 rounded-xl bg-indigo-600 px-4 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:opacity-40"
              >
                <Send className="h-4 w-4" />
                <span className="hidden sm:inline">Push</span>
              </button>
            </div>
          </form>
        </main>

        {/* RIGHT PANE: 24/7 PrizePicks Autopilot + Live tmux Pane + Permanent Postgres Search */}
        <section
          className={`${
            mobileTab === "prizepicks" || mobileTab === "terminal" || mobileTab === "search"
              ? "flex"
              : "hidden"
          } w-full shrink-0 flex-col border-l border-zinc-800/80 bg-[#0b0e14] lg:flex lg:w-[420px]`}
        >
          <div className="flex overflow-x-auto border-b border-zinc-800/80 px-2 pt-2">
            <button
              type="button"
              onClick={() => {
                setRightTab("prizepicks");
                setMobileTab("prizepicks");
              }}
              className={`flex shrink-0 items-center gap-1.5 border-b-2 px-2.5 py-2 text-xs font-medium ${
                rightTab === "prizepicks" && mobileTab !== "search" && mobileTab !== "terminal"
                  ? "border-fuchsia-400 text-fuchsia-300"
                  : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Sparkles className="h-3.5 w-3.5" />
              Props 24/7
            </button>
            <button
              type="button"
              onClick={() => {
                setRightTab("terminal");
                setMobileTab("terminal");
              }}
              className={`flex shrink-0 items-center gap-1.5 border-b-2 px-2.5 py-2 text-xs font-medium ${
                rightTab === "terminal" || mobileTab === "terminal"
                  ? "border-emerald-400 text-emerald-300"
                  : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Terminal className="h-3.5 w-3.5" />
              tmux
            </button>
            <button
              type="button"
              onClick={() => {
                setRightTab("search");
                setMobileTab("search");
              }}
              className={`flex shrink-0 items-center gap-1.5 border-b-2 px-2.5 py-2 text-xs font-medium ${
                rightTab === "search" || mobileTab === "search"
                  ? "border-indigo-400 text-indigo-300"
                  : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <History className="h-3.5 w-3.5" />
              Memory
            </button>
            <button
              type="button"
              onClick={() => setRightTab("setup")}
              className={`flex shrink-0 items-center gap-1.5 border-b-2 px-2.5 py-2 text-xs font-medium ${
                rightTab === "setup" &&
                mobileTab !== "search" &&
                mobileTab !== "terminal" &&
                mobileTab !== "prizepicks"
                  ? "border-amber-400 text-amber-300"
                  : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Cloud className="h-3.5 w-3.5" />
              Bridge
            </button>
          </div>

          {rightTab === "prizepicks" &&
          mobileTab !== "search" &&
          mobileTab !== "terminal" ? (
            <div className="flex min-h-0 flex-1 flex-col overflow-y-auto p-3.5 text-xs">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[11px] font-semibold tracking-wider text-fuchsia-300 uppercase">
                    24/7 PrizePicks Autopilot
                  </span>
                  <p className="text-[11px] text-zinc-400">
                    {ppData?.bestSlipType ?? "Scanning live board vs DraftKings..."}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void loadPrizePicksLive()}
                  disabled={ppLoading}
                  className="flex items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-[11px] text-zinc-300 hover:border-zinc-700"
                >
                  <RefreshCw className={`h-3 w-3 ${ppLoading ? "animate-spin" : ""}`} />
                  Scan
                </button>
              </div>

              {/* Optimal 1-a-Day Card */}
              <div className="mt-3 rounded-xl border border-fuchsia-500/30 bg-fuchsia-500/5 p-3">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-semibold text-zinc-100">
                    Best 1-a-Day Slip (1 Player / Game)
                  </span>
                  <span className="text-zinc-400">
                    {ppData?.scannedAt ? formatClock(ppData.scannedAt) : ""}
                  </span>
                </div>

                {ppData?.bestSlipLegs && ppData.bestSlipLegs.length > 0 ? (
                  <div className="mt-2.5 space-y-2">
                    {ppData.bestSlipLegs.map((leg, idx) => (
                      <div
                        key={`${leg.gameId}-${leg.player}-${leg.stat}`}
                        className="rounded-lg border border-zinc-800/90 bg-zinc-900/90 p-2"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-zinc-100">
                            {idx + 1}. {leg.player} ({leg.team})
                          </span>
                          <span
                            className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold ${
                              leg.side === "LESS"
                                ? "bg-rose-500/20 text-rose-300"
                                : "bg-emerald-500/20 text-emerald-300"
                            }`}
                          >
                            {leg.side} {leg.ppLine} {leg.stat}
                          </span>
                        </div>
                        <div className="mt-1 flex items-center justify-between text-[10px] text-zinc-400">
                          <span>
                            {leg.matchup} • DK {leg.dkLine} ({leg.dkOdds})
                          </span>
                          <span className="font-mono text-fuchsia-300">
                            {(leg.pCalibrated * 100).toFixed(1)}% win
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-2 text-[11px] text-zinc-400">
                    {ppLoading
                      ? "Pulling live PrizePicks & DraftKings lines..."
                      : "Between slates — waiting for pre-game DraftKings two-way prop lines to post for tomorrow's games."}
                  </p>
                )}
              </div>

              {/* 24/7 R&D Lab Promotions */}
              <div className="mt-3 space-y-2">
                <div className="text-[11px] font-semibold tracking-wider text-zinc-400 uppercase">
                  Autonomous R&D Hillclimber Log
                </div>
                {(ppData?.rdUpgrades ?? []).map((rd) => (
                  <div
                    key={rd.id}
                    className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-2.5"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-zinc-200">{rd.title}</span>
                      <span
                        className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                          rd.status === "PROMOTED"
                            ? "bg-emerald-500/20 text-emerald-300"
                            : "bg-zinc-800 text-zinc-400"
                        }`}
                      >
                        {rd.status}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] leading-relaxed text-zinc-400">
                      {rd.summary}
                    </p>
                  </div>
                ))}
              </div>

              {/* Post-Close Slip Autopsies + SMS Alert Feed */}
              <div className="mt-3 rounded-xl border border-indigo-500/30 bg-indigo-500/5 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-semibold text-indigo-200">
                    iPhone Push Topic (ntfy app)
                  </span>
                  <button
                    type="button"
                    onClick={async () => {
                      setPingNote("Sending...");
                      const r = await fetch("/api/autopsies", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({}),
                      });
                      const j = (await r.json()) as { ok?: boolean; status?: string };
                      setPingNote(j.ok ? "Sent to iPhone!" : j.status ?? "Error");
                      setTimeout(() => setPingNote(""), 3500);
                    }}
                    className="rounded-lg bg-indigo-600 px-2.5 py-1 text-[10px] font-medium text-white hover:bg-indigo-500"
                  >
                    {pingNote || "Send Test Push"}
                  </button>
                </div>
                <p className="mt-1 text-[10px] text-zinc-400">
                  In the iPhone <b>ntfy</b> app, tap <b>+</b> and type this exact Topic name:
                </p>
                <div className="mt-1.5 flex items-center justify-between rounded-lg bg-black/60 px-2.5 py-1.5 font-mono text-xs text-emerald-300">
                  <span>{autopsyData?.ntfyTopic ?? "loading..."}</span>
                  <button
                    type="button"
                    onClick={() => {
                      if (autopsyData?.ntfyTopic) {
                        void navigator.clipboard?.writeText(autopsyData.ntfyTopic);
                      }
                    }}
                    className="text-[10px] text-zinc-400 hover:text-white"
                  >
                    Copy
                  </button>
                </div>
              </div>

              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-semibold tracking-wider text-zinc-400 uppercase">
                    Post-Close Slip Autopsies
                  </span>
                </div>

                {(autopsyData?.recentAutopsies ?? []).map((au) => (
                  <div
                    key={au.date}
                    className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-2.5"
                  >
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold text-zinc-100">{au.date}</span>
                      <span className="font-mono text-[10px] text-indigo-300">{au.slip}</span>
                    </div>
                    <div className="mt-2 space-y-1.5">
                      {au.legs.map((lg) => (
                        <div
                          key={`${au.date}-${lg.player}`}
                          className="rounded-lg bg-black/40 p-2 text-[11px]"
                        >
                          <div className="flex items-center justify-between gap-1">
                            <span className="font-medium text-zinc-200">
                              {lg.player} • {lg.pick}
                            </span>
                            <span className="font-mono text-[10px] text-zinc-400">
                              {lg.actual}
                            </span>
                          </div>
                          <div className="mt-0.5 font-mono text-[10px] text-fuchsia-300">
                            {lg.tag}
                          </div>
                          <p className="mt-0.5 text-[10px] leading-relaxed text-zinc-400">
                            {lg.note}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : rightTab === "search" || mobileTab === "search" ? (
            <div className="flex min-h-0 flex-1 flex-col p-3.5">
              <div className="relative">
                <Search className="pointer-events-none absolute top-2.5 left-3 h-4 w-4 text-zinc-500" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search all permanent session history..."
                  className="w-full rounded-xl border border-zinc-800 bg-zinc-900 py-2 pr-3 pl-9 text-xs text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
                />
              </div>

              <div className="mt-2 flex items-center gap-1.5">
                {(["all", "inter_session", "user"] as const).map((sc) => (
                  <button
                    key={sc}
                    type="button"
                    onClick={() => setSearchScope(sc)}
                    className={`rounded-md px-2 py-1 text-[11px] ${
                      searchScope === sc
                        ? "bg-indigo-500/20 text-indigo-300 ring-1 ring-indigo-500/30"
                        : "bg-zinc-900 text-zinc-400"
                    }`}
                  >
                    {sc === "all"
                      ? "All History"
                      : sc === "inter_session"
                        ? "Session↔Session"
                        : "My Messages"}
                  </button>
                ))}
              </div>

              <div className="mt-3 flex-1 space-y-2.5 overflow-y-auto">
                {searching ? (
                  <p className="text-xs text-zinc-500">Searching Postgres index...</p>
                ) : !searchQuery.trim() ? (
                  <p className="text-xs leading-relaxed text-zinc-500">
                    Every prompt, reply, and inter-session handoff is stored in Postgres with GIN full-text indexing. Search any past decision and inject it into an active session.
                  </p>
                ) : searchResults.length === 0 ? (
                  <p className="text-xs text-zinc-500">No matching messages found.</p>
                ) : (
                  searchResults.map((r) => (
                    <div
                      key={r.id}
                      className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-2.5 text-xs"
                    >
                      <div className="flex items-center justify-between text-[11px] text-zinc-400">
                        <span className="font-mono text-indigo-300">
                          {r.from_label} → {r.to_label}
                        </span>
                        <span>{formatClock(r.created_at)}</span>
                      </div>
                      <p className="mt-1 line-clamp-4 text-zinc-200">{r.body}</p>
                      <button
                        type="button"
                        onClick={() => injectMemoryIntoDraft(r)}
                        className="mt-2 text-[11px] font-medium text-indigo-400 hover:text-indigo-300"
                      >
                        Recall into active chat →
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>
          ) : rightTab === "setup" ? (
            <div className="flex-1 space-y-3 overflow-y-auto p-3.5 text-xs text-zinc-300">
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-3">
                <h3 className="font-semibold text-zinc-100">1. Zero-Polling Local tmux Bridge</h3>
                <p className="mt-1 text-[11px] leading-relaxed text-zinc-400">
                  Run the bridge daemon on your Mac/Linux box to push web & phone messages straight into live Cursor <code className="text-indigo-300">agent</code> tmux panes via <code className="text-indigo-300">send-keys</code> (mid-turn steering, zero polling):
                </p>
                <div className="mt-2 flex items-center justify-between gap-2 rounded-lg bg-black/60 p-2 font-mono text-[11px] text-emerald-300">
                  <span className="truncate">{bridgeCommand}</span>
                  <button
                    type="button"
                    onClick={() => {
                      void navigator.clipboard?.writeText(bridgeCommand);
                      setCopiedCmd(true);
                      setTimeout(() => setCopiedCmd(false), 1500);
                    }}
                    className="shrink-0 text-zinc-400 hover:text-white"
                  >
                    {copiedCmd ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-3">
                <h3 className="font-semibold text-zinc-100">2. Coordinator MCP Tools</h3>
                <p className="mt-1 text-[11px] leading-relaxed text-zinc-400">
                  Every Cursor CLI session loads <code className="text-indigo-300">.cursor/mcp.json</code> so your Coordinator session can start workers, message sibling sessions, and search Postgres memory autonomously:
                </p>
                <ul className="mt-2 space-y-1 font-mono text-[11px] text-zinc-300">
                  <li>• fleet_spawn_tmux_session</li>
                  <li>• fleet_message_session</li>
                  <li>• fleet_read_tmux_pane</li>
                  <li>• fleet_search_history</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col p-3">
              <div className="mb-2 flex items-center justify-between text-[11px] text-zinc-400">
                <span className="font-mono text-emerald-400">
                  {activeSession
                    ? activeSession.tmux_target || activeSession.slug
                    : "select a session"}
                </span>
                <span>
                  {activeSession?.last_heartbeat_at
                    ? `Synced ${formatClock(activeSession.last_heartbeat_at)}`
                    : "Live snapshot"}
                </span>
              </div>
              <pre className="flex-1 overflow-auto rounded-xl border border-zinc-800/90 bg-[#05070a] p-3 font-mono text-[11px] leading-relaxed text-emerald-300/90">
                {activeSession?.pane_snapshot ||
                  "$ Waiting for tmux capture-pane heartbeat..."}
              </pre>
            </div>
          )}
        </section>
      </div>

      {/* Phone Bottom Navigation Bar */}
      <nav className="grid shrink-0 grid-cols-5 border-t border-zinc-800 bg-[#0d1017] py-1.5 md:hidden">
        <button
          type="button"
          onClick={() => setMobileTab("sessions")}
          className={`flex flex-col items-center gap-0.5 py-1 text-[10px] ${
            mobileTab === "sessions" ? "text-indigo-400" : "text-zinc-400"
          }`}
        >
          <Zap className="h-4 w-4" />
          Sessions
        </button>
        <button
          type="button"
          onClick={() => setMobileTab("chat")}
          className={`flex flex-col items-center gap-0.5 py-1 text-[10px] ${
            mobileTab === "chat" ? "text-indigo-400" : "text-zinc-400"
          }`}
        >
          <MessageSquare className="h-4 w-4" />
          Chat
        </button>
        <button
          type="button"
          onClick={() => {
            setRightTab("prizepicks");
            setMobileTab("prizepicks");
          }}
          className={`flex flex-col items-center gap-0.5 py-1 text-[10px] ${
            mobileTab === "prizepicks" ? "text-fuchsia-400" : "text-zinc-400"
          }`}
        >
          <Sparkles className="h-4 w-4" />
          Props 24/7
        </button>
        <button
          type="button"
          onClick={() => {
            setRightTab("terminal");
            setMobileTab("terminal");
          }}
          className={`flex flex-col items-center gap-0.5 py-1 text-[10px] ${
            mobileTab === "terminal" ? "text-emerald-400" : "text-zinc-400"
          }`}
        >
          <Terminal className="h-4 w-4" />
          tmux
        </button>
        <button
          type="button"
          onClick={() => {
            setRightTab("search");
            setMobileTab("search");
          }}
          className={`flex flex-col items-center gap-0.5 py-1 text-[10px] ${
            mobileTab === "search" ? "text-indigo-400" : "text-zinc-400"
          }`}
        >
          <Search className="h-4 w-4" />
          Memory
        </button>
      </nav>

      {/* Spawn New Session Modal */}
      {showSpawn && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <form
            onSubmit={handleSpawn}
            className="w-full max-w-md rounded-2xl border border-zinc-800 bg-[#0f131c] p-5 shadow-xl"
          >
            <h3 className="text-base font-semibold text-zinc-100">
              Spawn New Cursor Session
            </h3>
            <p className="mt-1 text-xs text-zinc-400">
              Start a local tmux Cursor CLI worker or launch a Cursor Cloud Agent.
            </p>

            <div className="mt-4 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setSpawnKind("tmux")}
                  className={`rounded-xl border p-2.5 text-left text-xs ${
                    spawnKind === "tmux"
                      ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-200"
                      : "border-zinc-800 bg-zinc-900 text-zinc-400"
                  }`}
                >
                  <div className="font-semibold">Local tmux CLI</div>
                  <div className="mt-0.5 text-[10px] opacity-80">
                    Spawns `agent` in tmux
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => setSpawnKind("cloud")}
                  className={`rounded-xl border p-2.5 text-left text-xs ${
                    spawnKind === "cloud"
                      ? "border-indigo-500/50 bg-indigo-500/10 text-indigo-200"
                      : "border-zinc-800 bg-zinc-900 text-zinc-400"
                  }`}
                >
                  <div className="font-semibold">Cloud Agent</div>
                  <div className="mt-0.5 text-[10px] opacity-80">
                    Cursor /v1/agents VM
                  </div>
                </button>
              </div>

              <div>
                <label className="block text-xs text-zinc-400">Session Name</label>
                <input
                  type="text"
                  required
                  value={spawnTitle}
                  onChange={(e) => setSpawnTitle(e.target.value)}
                  placeholder="e.g. friday-props-worker"
                  className="mt-1 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus:border-indigo-500 focus:outline-none"
                />
              </div>

              {spawnKind === "tmux" ? (
                <div>
                  <label className="block text-xs text-zinc-400">Working Directory</label>
                  <input
                    type="text"
                    value={spawnWorkdir}
                    onChange={(e) => setSpawnWorkdir(e.target.value)}
                    className="mt-1 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-200 focus:border-indigo-500 focus:outline-none"
                  />
                </div>
              ) : (
                <div className="space-y-2">
                  {repoList.length > 0 && (
                    <div>
                      <label className="block text-xs text-zinc-400">Quick Pick Repo</label>
                      <select
                        value={spawnRepo}
                        onChange={(e) => setSpawnRepo(e.target.value)}
                        className="mt-1 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-2.5 py-2 text-xs text-zinc-200"
                      >
                        {repoList.map((r) => (
                          <option key={r.url} value={r.url}>
                            {r.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-xs text-zinc-400">GitHub Repo URL</label>
                      <input
                        type="url"
                        value={spawnRepo}
                        onChange={(e) => setSpawnRepo(e.target.value)}
                        className="mt-1 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-2.5 py-2 font-mono text-xs text-zinc-200"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-zinc-400">Base Branch</label>
                      <input
                        type="text"
                        value={spawnBranch}
                        onChange={(e) => setSpawnBranch(e.target.value)}
                        placeholder="main"
                        className="mt-1 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-2.5 py-2 font-mono text-xs text-zinc-200"
                      />
                    </div>
                  </div>
                </div>
              )}

              <div>
                <label className="block text-xs text-zinc-400">Opening Task (Pushed Immediately)</label>
                <textarea
                  rows={3}
                  value={spawnPrompt}
                  onChange={(e) => setSpawnPrompt(e.target.value)}
                  placeholder="Check tonight's PrizePicks lines against DraftKings and message @coordinator..."
                  className="mt-1 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-100 focus:border-indigo-500 focus:outline-none"
                />
              </div>
            </div>

            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowSpawn(false)}
                className="rounded-xl border border-zinc-800 px-4 py-2 text-xs text-zinc-300 hover:bg-zinc-900"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={spawning || !spawnTitle.trim()}
                className="rounded-xl bg-indigo-600 px-4 py-2 text-xs font-medium text-white hover:bg-indigo-500 disabled:opacity-40"
              >
                {spawning ? "Spawning..." : "Launch Session"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
