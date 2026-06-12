import {
  AlertTriangle,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleDot,
  Globe2,
  Menu,
  Plus,
  ScanSearch,
  Send,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Verdict = "phishing" | "benign" | "unknown";
type ThreadStatus = "processing" | "ready" | "error";

type Analysis = {
  key: string;
  name: string;
  status: "pending" | "running" | "complete" | "error";
  verdict: Verdict;
  phishing_factors: string[];
  benign_factors: string[];
  reasoning: string;
  raw_output: string;
  error?: string | null;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

type ThreadSummary = {
  id: string;
  title: string;
  url: string;
  status: ThreadStatus;
  overall_verdict: Verdict;
  updated_at: string;
};

type Thread = ThreadSummary & {
  progress: string;
  error?: string | null;
  analyses: Analysis[];
  messages: Message[];
  website?: {
    title?: string;
    final_url?: string;
    status_code?: number;
    fetched_at?: string;
    screenshot?: string;
  } | null;
  document?: Record<string, unknown> | null;
};

const api = {
  async list(): Promise<ThreadSummary[]> {
    const response = await fetch("/api/threads");
    if (!response.ok) throw new Error("Could not load threads.");
    return (await response.json()).items;
  },
  async get(id: string): Promise<Thread> {
    const response = await fetch(`/api/threads/${id}`);
    if (!response.ok) throw new Error("Could not load this analysis.");
    return response.json();
  },
  async create(url: string): Promise<Thread> {
    const response = await fetch("/api/threads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not start analysis.");
    return data;
  },
  async remove(id: string): Promise<void> {
    const response = await fetch(`/api/threads/${id}`, { method: "DELETE" });
    if (!response.ok) throw new Error("Could not delete thread.");
  },
  async message(id: string, content: string): Promise<Thread> {
    const response = await fetch(`/api/threads/${id}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not send message.");
    return data;
  },
};

function VerdictMark({ verdict, compact = false }: { verdict: Verdict; compact?: boolean }) {
  const phishing = verdict === "phishing";
  const benign = verdict === "benign";
  return (
    <span className={`verdict verdict--${verdict} ${compact ? "verdict--compact" : ""}`}>
      {phishing ? <ShieldAlert size={15} /> : benign ? <ShieldCheck size={15} /> : <CircleDot size={13} />}
      {verdict === "unknown" ? "Pending" : verdict}
    </span>
  );
}

function StartForm({
  onSubmit,
  busy,
  compact = false,
}: {
  onSubmit: (url: string) => Promise<void>;
  busy: boolean;
  compact?: boolean;
}) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    let parsed: URL;
    try {
      parsed = new URL(url.trim());
      if (!["http:", "https:"].includes(parsed.protocol)) throw new Error();
    } catch {
      setError("Enter a complete http:// or https:// URL.");
      return;
    }
    try {
      await onSubmit(parsed.toString());
    } catch {
      return;
    }
    setUrl("");
  }

  return (
    <form className={`url-form ${compact ? "url-form--compact" : ""}`} onSubmit={submit}>
      <label className="url-label" htmlFor={compact ? "compact-url" : "start-url"}>
        Enter a URL
      </label>
      <div className="url-field">
        <Globe2 size={18} />
        <input
          aria-label="Website URL"
          autoComplete="url"
          inputMode="url"
          id={compact ? "compact-url" : "start-url"}
          onChange={(event) => setUrl(event.target.value)}
          placeholder="https://example.com/login"
          required
          value={url}
        />
        <button disabled={busy} type="submit">
          {busy ? "Starting..." : compact ? <ArrowUpRight size={18} /> : "Analyze URL"}
        </button>
      </div>
      {error && <p className="field-error">{error}</p>}
    </form>
  );
}

function AnalysisPanel({ analysis, index }: { analysis: Analysis; index: number }) {
  const [open, setOpen] = useState(index === 0);
  const pending = analysis.status === "pending" || analysis.status === "running";

  return (
    <article className={`analysis analysis--${analysis.status}`} style={{ animationDelay: `${index * 90}ms` }}>
      <button className="analysis__header" onClick={() => setOpen((value) => !value)} type="button">
        <span className="model-index">0{index + 1}</span>
        <span className="model-title">
          <strong>{analysis.name}</strong>
          {analysis.status !== "complete" && (
            <small>
              {analysis.status === "running" ? "Inference running" : analysis.status === "pending" ? "Waiting" : "Unavailable"}
            </small>
          )}
        </span>
        <VerdictMark verdict={analysis.verdict} />
        <ChevronDown className={open ? "rotate" : ""} size={18} />
      </button>

      {open && (
        <div className="analysis__body">
          {pending && (
            <div className="model-loading">
              <span />
              <p>The feature document is being evaluated by this model.</p>
            </div>
          )}
          {analysis.status === "error" && (
            <div className="model-error">
              <AlertTriangle size={18} />
              <p>{analysis.error || "This Hugging Face endpoint did not return a result."}</p>
            </div>
          )}
          {analysis.status === "complete" && (
            <>
              <div className="evidence-columns">
                <section>
                  <p className="eyebrow eyebrow--risk">Risk factors</p>
                  <ul>
                    {(analysis.phishing_factors.length ? analysis.phishing_factors : ["No strong risk factor cited."]).map(
                      (factor) => (
                        <li key={factor}>
                          <X size={14} />
                          <span>{factor}</span>
                        </li>
                      ),
                    )}
                  </ul>
                </section>
                <section>
                  <p className="eyebrow eyebrow--safe">Mitigating factors</p>
                  <ul>
                    {(analysis.benign_factors.length ? analysis.benign_factors : ["No strong mitigating factor cited."]).map(
                      (factor) => (
                        <li key={factor}>
                          <Check size={14} />
                          <span>{factor}</span>
                        </li>
                      ),
                    )}
                  </ul>
                </section>
              </div>
              <div className="reasoning">
                <p className="eyebrow">Reasoning</p>
                <p>{analysis.reasoning || analysis.raw_output}</p>
              </div>
            </>
          )}
        </div>
      )}
    </article>
  );
}

function Sidebar({
  threads,
  activeId,
  onSelect,
  onNew,
  onDelete,
  mobileOpen,
  onClose,
}: {
  threads: ThreadSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  mobileOpen: boolean;
  onClose: () => void;
}) {
  return (
    <>
      {mobileOpen && <button aria-label="Close menu" className="sidebar-scrim" onClick={onClose} />}
      <aside className={`sidebar ${mobileOpen ? "sidebar--open" : ""}`}>
        <div className="brand">
          <span className="brand-mark">
            <ScanSearch size={20} />
          </span>
          <span>
            <strong>Traceguard</strong>
          </span>
        </div>
        <button className="new-thread" onClick={onNew} type="button">
          <Plus size={17} />
          New analysis
        </button>
        <div className="thread-heading">
          <span>Threads</span>
          <span>{threads.length}</span>
        </div>
        <nav className="thread-list">
          {threads.map((thread) => (
            <div className={`thread-item ${thread.id === activeId ? "thread-item--active" : ""}`} key={thread.id}>
              <button
                onClick={() => {
                  onSelect(thread.id);
                  onClose();
                }}
                type="button"
              >
                <span className={`thread-dot thread-dot--${thread.overall_verdict}`} />
                <span>
                  <strong>{thread.title}</strong>
                  <small>{new URL(thread.url).hostname}</small>
                </span>
              </button>
              <button aria-label={`Delete ${thread.title}`} className="delete-thread" onClick={() => onDelete(thread.id)}>
                <Trash2 size={14} />
              </button>
            </div>
          ))}
          {!threads.length && <p className="sidebar-empty">No saved analyses yet.</p>}
        </nav>
      </aside>
    </>
  );
}

export default function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [thread, setThread] = useState<Thread | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [sending, setSending] = useState(false);
  const [message, setMessage] = useState("");
  const [notice, setNotice] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [documentOpen, setDocumentOpen] = useState(false);
  const conversationEnd = useRef<HTMLDivElement>(null);

  const refreshList = useCallback(async () => {
    const items = await api.list();
    setThreads(items);
    return items;
  }, []);

  const loadThread = useCallback(async (id: string) => {
    const selected = await api.get(id);
    setThread(selected);
    setActiveId(id);
    return selected;
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const items = await refreshList();
        if (items[0]) await loadThread(items[0].id);
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "The application could not start.");
      } finally {
        setLoading(false);
      }
    })();
  }, [loadThread, refreshList]);

  useEffect(() => {
    if (!thread || thread.status !== "processing") return;
    const timer = window.setInterval(async () => {
      try {
        const updated = await loadThread(thread.id);
        await refreshList();
        if (updated.status !== "processing") window.clearInterval(timer);
      } catch {
        window.clearInterval(timer);
      }
    }, 2200);
    return () => window.clearInterval(timer);
  }, [loadThread, refreshList, thread]);

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [thread?.messages.length]);

  async function createThread(url: string) {
    setCreating(true);
    setNotice("");
    try {
      const created = await api.create(url);
      setThread(created);
      setActiveId(created.id);
      await refreshList();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not start analysis.");
      throw error;
    } finally {
      setCreating(false);
    }
  }

  async function deleteThread(id: string) {
    if (!window.confirm("Delete this analysis and its conversation?")) return;
    try {
      await api.remove(id);
      const items = await refreshList();
      if (id === activeId) {
        if (items[0]) await loadThread(items[0].id);
        else {
          setActiveId(null);
          setThread(null);
        }
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not delete analysis.");
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    const content = message.trim();
    if (!thread || !content || sending) return;
    setSending(true);
    setMessage("");
    setNotice("");
    setThread({
      ...thread,
      messages: [...thread.messages, { role: "user", content, created_at: new Date().toISOString() }],
    });
    try {
      setThread(await api.message(thread.id, content));
      await refreshList();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not send message.");
      await loadThread(thread.id);
    } finally {
      setSending(false);
    }
  }

  const completeAnalyses = useMemo(
    () => thread?.analyses.filter((analysis) => analysis.status === "complete") ?? [],
    [thread],
  );

  if (loading) {
    return (
      <main className="boot">
        <span className="brand-mark">
          <ScanSearch size={22} />
        </span>
        <p>Loading forensic workspace</p>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        activeId={activeId}
        mobileOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
        onDelete={deleteThread}
        onNew={() => {
          setActiveId(null);
          setThread(null);
          setMobileOpen(false);
        }}
        onSelect={(id) => void loadThread(id)}
        threads={threads}
      />

      <main className="workspace">
        <header className="mobile-header">
          <button aria-label="Open threads" onClick={() => setMobileOpen(true)}>
            <Menu size={20} />
          </button>
          <strong>Traceguard</strong>
          <span />
        </header>

        {notice && (
          <div className="notice">
            <AlertTriangle size={16} />
            <span>{notice}</span>
            <button aria-label="Dismiss" onClick={() => setNotice("")}>
              <X size={15} />
            </button>
          </div>
        )}

        {!thread ? (
          <section className="empty-state">
            <StartForm busy={creating} onSubmit={createThread} />
          </section>
        ) : (
          <div className="thread-view">
            <header className="thread-top">
              <div>
                <h1>{thread.title}</h1>
                <a href={thread.url} rel="noreferrer" target="_blank">
                  <span>{thread.url}</span>
                  <ArrowUpRight size={14} />
                </a>
              </div>
              <VerdictMark verdict={thread.overall_verdict} />
            </header>

            <section className="page-preview">
              {thread.website?.screenshot ? (
                <img alt={`Screenshot of ${thread.title}`} src={thread.website.screenshot} />
              ) : (
                <div className="screenshot-empty">Screenshot will appear after retrieval.</div>
              )}
              <div className="document-actions">
                <button onClick={() => setDocumentOpen((value) => !value)} type="button">
                  {documentOpen ? "Hide JSON document" : "View JSON document"}
                </button>
              </div>
              {documentOpen && (
                <pre className="document-json">
                  <code>{JSON.stringify(thread.document ?? {}, null, 2)}</code>
                </pre>
              )}
            </section>

            {thread.status === "processing" && (
              <section className="progress-line">
                <span className="progress-pulse" />
                <div>
                  <strong>{thread.progress || "Preparing analysis"}</strong>
                  <p>Results appear as each stage completes. You can leave this thread and return later.</p>
                </div>
              </section>
            )}

            {thread.status === "error" && (
              <section className="fatal-error">
                <AlertTriangle size={20} />
                <div>
                  <strong>Analysis stopped</strong>
                  <p>{thread.error || "The website could not be analyzed."}</p>
                </div>
              </section>
            )}

            {thread.website && (
              <div className="capture-strip">
                <span>HTTP {thread.website.status_code ?? "—"}</span>
              </div>
            )}

            <section className="model-section">
              <div className="section-heading">
                <div>
                  <h2>Model evidence</h2>
                </div>
                <p>
                  {completeAnalyses.length} of {thread.analyses.length} complete
                </p>
              </div>
              <div className="analysis-list">
                {thread.analyses.map((analysis, index) => (
                  <AnalysisPanel analysis={analysis} index={index} key={analysis.key} />
                ))}
              </div>
            </section>

            {thread.status === "ready" && (
              <section className="conversation">
                <div className="messages">
                  {thread.messages.map((item, index) => (
                    <div className={`message message--${item.role}`} key={`${item.created_at}-${index}`}>
                      <span>{item.role === "user" ? "You" : "Qwen"}</span>
                      <p>{item.content}</p>
                    </div>
                  ))}
                  {sending && (
                    <div className="message message--assistant message--typing">
                      <span>Qwen</span>
                      <p>
                        <i />
                        <i />
                        <i />
                      </p>
                    </div>
                  )}
                  <div ref={conversationEnd} />
                </div>

                <form className="composer" onSubmit={sendMessage}>
                  <textarea
                    aria-label="Follow-up question"
                    disabled={sending}
                    onChange={(event) => setMessage(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }}
                    placeholder="Ask a follow-up about this website..."
                    rows={1}
                    value={message}
                  />
                  <button aria-label="Send message" disabled={sending || !message.trim()} type="submit">
                    <Send size={17} />
                  </button>
                </form>
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
