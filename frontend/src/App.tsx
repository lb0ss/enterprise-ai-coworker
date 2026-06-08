import { useState, useRef, useEffect, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API = 'http://localhost:8001'

interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    streaming?: boolean
    queryId?: number  // set when [QUERY_ID:N] arrives; enables feedback buttons
}

interface UploadResult {
    filename: string
    chunks_indexed: number
}

interface DocumentsResponse {
    documents: UploadResult[]
    total_chunks: number
}

interface AgentStep {
    id: string
    type: 'tool_call' | 'tool_result' | 'token'
    tool?: string
    input?: string
    content: string
}

interface AgentRun {
    id: string
    task: string
    steps: AgentStep[]
    answer: string
    running: boolean
}

interface AnalyticsData {
    total_queries: number
    cache_hits: number
    cache_miss_rate: number
    avg_latency_ms: number
    total_tokens: number
    thumbs_up: number
    thumbs_down: number
    recent_queries: {
        question: string
        latency_ms: number
        token_count: number
        cache_hit: number
        created_at: string
    }[]
}

export default function App() {
    const [tab, setTab] = useState<'chat' | 'agent' | 'analytics'>('chat')

    // shared
    const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
    const [indexedDocuments, setIndexedDocuments] = useState<UploadResult[]>([])
    const [uploading, setUploading] = useState(false)
    const [uploadError, setUploadError] = useState<string | null>(null)
    const [backendUp, setBackendUp] = useState(false)

    // chat tab
    const [messages, setMessages] = useState<Message[]>([])
    const [question, setQuestion] = useState('')
    const [streaming, setStreaming] = useState(false)
    const [lastQueryId, setLastQueryId] = useState<number | null>(null)
    // tracks which query_ids have been rated so we don't show buttons after voting
    const [feedbackGiven, setFeedbackGiven] = useState<Set<number>>(new Set())

    // analytics tab
    const [analytics, setAnalytics] = useState<AnalyticsData | null>(null)
    const [analyticsLoading, setAnalyticsLoading] = useState(false)

    const messagesEndRef = useRef<HTMLDivElement>(null)

    // agent tab
    const [agentRuns, setAgentRuns] = useState<AgentRun[]>([])
    const [agentTask, setAgentTask] = useState('')
    const [agentRunning, setAgentRunning] = useState(false)
    const agentEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    useEffect(() => {
        agentEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [agentRuns])

    useEffect(() => {
        const check = async () => {
            try {
                const res = await fetch(`${API}/health`)
                setBackendUp(res.ok)
            } catch {
                setBackendUp(false)
            }
        }
        check()
        const interval = setInterval(check, 30000)
        return () => clearInterval(interval)
    }, [])

    useEffect(() => {
        const loadDocuments = async () => {
            try {
                const { data } = await axios.get<DocumentsResponse>(`${API}/documents`)
                if (data.documents.length > 0) {
                    setIndexedDocuments(data.documents)
                    // set uploadResult so chat input is enabled
                    setUploadResult(data.documents[data.documents.length - 1])
                }
            } catch {
                // silently fail — no documents indexed yet
            }
        }
        loadDocuments()
    }, [])

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        const file = acceptedFiles[0]
        if (!file) return
        setUploading(true)
        setUploadError(null)
        const formData = new FormData()
        formData.append('file', file)
        try {
            const { data } = await axios.post<UploadResult>(`${API}/upload`, formData)
            setUploadResult(data)
            setIndexedDocuments((prev) => {
                const exists = prev.find((d) => d.filename === data.filename)
                return exists ? prev : [...prev, data]
            })
        } catch (err: any) {
            setUploadError(err.response?.data?.detail ?? 'Upload failed')
        } finally {
            setUploading(false)
        }
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'text/plain': ['.txt'], 'text/markdown': ['.md'], 'text/x-python': ['.py'] },
        maxFiles: 1,
        disabled: uploading,
    })

    const sendMessage = async () => {
        if (!question.trim() || streaming) return

        const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: question.trim() }
        const assistantMsg: Message = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: '',
            streaming: true,
        }

        setMessages((prev) => [...prev, userMsg, assistantMsg])
        setQuestion('')
        setStreaming(true)

        try {
            const history = messages
                .filter((m) => !m.streaming)
                .map((m) => ({ role: m.role, content: m.content }))

            const response = await fetch(`${API}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: userMsg.content, history }),
            })

            const reader = response.body!.getReader() // gets the readable stream from the fetch response
            const decoder = new TextDecoder()

            while (true) {
                const { done, value } = await reader.read() // waits for the next chunk of bytes to arrive from the backend
                if (done) break
                const lines = decoder.decode(value).split('\n') // converts those raw bytes into a string
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue // 'data: ' is SSE protocol format
                    const data = line.slice(6)
                    if (data === '[DONE]') break
                    if (data.startsWith('[QUERY_ID:')) {
                        const id = parseInt(data.slice(10, -1))
                        setLastQueryId(id)
                        // stamp queryId onto the assistant message so feedback buttons appear
                        setMessages((prev) =>
                            prev.map((m) =>
                                m.id === assistantMsg.id ? { ...m, queryId: id } : m
                            )
                        )
                        continue
                    }
                    setMessages((prev) =>
                        prev.map((m) =>
                            m.id === assistantMsg.id ? { ...m, content: m.content + data } : m
                        )
                    )
                }
            }
        } catch {
            setMessages((prev) =>
                prev.map((m) =>
                    m.id === assistantMsg.id
                        ? { ...m, content: 'Error: failed to get a response.' }
                        : m
                )
            )
        } finally {
            setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsg.id ? { ...m, streaming: false } : m))
            )
            setStreaming(false)
        }
    }

    const runAgent = async () => {
        if (!agentTask.trim() || agentRunning) return

        const run: AgentRun = {
            id: crypto.randomUUID(),
            task: agentTask.trim(),
            steps: [],
            answer: '',
            running: true,
        }

        setAgentRuns((prev) => [...prev, run])
        setAgentTask('')
        setAgentRunning(true)

        try {
            const response = await fetch(`${API}/agent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task: run.task }),
            })

            const reader = response.body!.getReader()
            const decoder = new TextDecoder()

            while (true) {
                const { done, value } = await reader.read()
                if (done) break
                const lines = decoder.decode(value).split('\n')
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue
                    const data = line.slice(6)
                    if (data === '[DONE]') break

                    const event = JSON.parse(data)

                    if (event.type === 'tool_call') {
                        const step: AgentStep = {
                            id: crypto.randomUUID(),
                            type: 'tool_call',
                            tool: event.tool,
                            input: event.input,
                            content: '',
                        }
                        setAgentRuns((prev) =>
                            prev.map((r) =>
                                r.id === run.id ? { ...r, steps: [...r.steps, step] } : r
                            )
                        )
                    } else if (event.type === 'tool_result') {
                        const step: AgentStep = {
                            id: crypto.randomUUID(),
                            type: 'tool_result',
                            tool: event.tool,
                            content: event.content,
                        }
                        setAgentRuns((prev) =>
                            prev.map((r) =>
                                r.id === run.id ? { ...r, steps: [...r.steps, step] } : r
                            )
                        )
                    } else if (event.type === 'token') {
                        setAgentRuns((prev) =>
                            prev.map((r) =>
                                r.id === run.id
                                    ? { ...r, answer: r.answer + event.content }
                                    : r
                            )
                        )
                    }
                }
            }
        } catch {
            setAgentRuns((prev) =>
                prev.map((r) =>
                    r.id === run.id ? { ...r, answer: 'Error: agent failed to complete the task.' } : r
                )
            )
        } finally {
            setAgentRuns((prev) =>
                prev.map((r) => (r.id === run.id ? { ...r, running: false } : r))
            )
            setAgentRunning(false)
        }
    }

    const handleChatKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const handleAgentKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            runAgent()
        }
    }

    const fetchAnalytics = async () => {
        setAnalyticsLoading(true)
        try {
            const res = await fetch(`${API}/analytics`)
            const data: AnalyticsData = await res.json()
            setAnalytics(data)
        } catch {
            // silently fail — backend may be warming up
        } finally {
            setAnalyticsLoading(false)
        }
    }

    const sendFeedback = async (queryId: number, rating: 1 | -1) => {
        setFeedbackGiven((prev) => new Set(prev).add(queryId))
        try {
            await fetch(`${API}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query_id: queryId, rating }),
            })
        } catch {
            // feedback is best-effort; don't surface errors to the user
        }
    }

    return (
        <div className="flex flex-col h-screen bg-background text-foreground">
            {/* Header */}
            <header className="shrink-0 border-b border-border/50 px-6 h-12 flex items-center gap-3">
                <div
                    className={`w-1.5 h-1.5 rounded-full ${backendUp ? 'bg-green-500' : 'bg-red-500'}`}
                    title={backendUp ? 'Backend online' : 'Backend offline'}
                />
                <span className="text-sm font-medium tracking-wide">DocMind</span>

                {/* Tab switcher */}
                <div className="ml-6 flex items-center gap-1 bg-muted/40 rounded-lg p-1">
                    <button
                        onClick={() => setTab('chat')}
                        className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                            tab === 'chat'
                                ? 'bg-background text-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground'
                        }`}
                    >
                        Chat
                    </button>
                    <button
                        onClick={() => setTab('agent')}
                        className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                            tab === 'agent'
                                ? 'bg-background text-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground'
                        }`}
                    >
                        Agent
                    </button>
                    <button
                        onClick={() => { setTab('analytics'); fetchAnalytics() }}
                        className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                            tab === 'analytics'
                                ? 'bg-background text-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground'
                        }`}
                    >
                        Analytics
                    </button>
                </div>
            </header>

            {/* Body */}
            <div className="flex flex-1 overflow-hidden">
                {/* LEFT: Upload panel */}
                <aside className="w-64 shrink-0 border-r border-border/50 bg-muted/25 flex flex-col p-4 gap-4">
                    <div>
                        <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-3">
                            Document
                        </p>

                        <div
                            {...getRootProps()}
                            className={[
                                'border border-dashed rounded-xl p-5 cursor-pointer transition-colors',
                                'flex flex-col items-center justify-center gap-2 text-center min-h-[120px]',
                                isDragActive
                                    ? 'border-[var(--accent-border)] bg-[var(--accent-bg)]'
                                    : 'border-border/70 hover:border-border hover:bg-muted/30',
                                uploading ? 'opacity-50 pointer-events-none' : '',
                            ].join(' ')}
                        >
                            <input {...getInputProps()} />
                            {uploading ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-muted-foreground border-t-foreground rounded-full animate-spin" />
                                    <span className="text-xs text-muted-foreground">Indexing...</span>
                                </>
                            ) : isDragActive ? (
                                <span className="text-sm text-foreground">Drop to upload</span>
                            ) : (
                                <>
                                    <svg
                                        className="w-6 h-6 text-muted-foreground"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                        strokeWidth={1.5}
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                                        />
                                    </svg>
                                    <span className="text-xs text-muted-foreground">
                                        Drop a file or click to browse
                                    </span>
                                    <span className="text-[10px] text-muted-foreground/50">
                                        .txt · .md · .py
                                    </span>
                                </>
                            )}
                        </div>

                        {uploadError && (
                            <p className="text-xs text-destructive mt-2">{uploadError}</p>
                        )}
                    </div>

                    {indexedDocuments.length > 0 && (
                        <div className="space-y-1.5">
                            {indexedDocuments.map((doc) => (
                                <div key={doc.filename} className="rounded-lg bg-muted/50 px-3 py-2.5 space-y-0.5">
                                    <div className="flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-green-500 shrink-0" />
                                        <span className="text-xs font-medium truncate">{doc.filename}</span>
                                    </div>
                                    <p className="text-xs text-muted-foreground pl-3.5">
                                        {doc.chunks_indexed} chunks indexed
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}

                    <p className="mt-auto text-[10px] text-muted-foreground/40 leading-relaxed">
                        Files are chunked, embedded via OpenAI, and stored locally in ChromaDB.
                    </p>
                </aside>

                {/* RIGHT: main panel */}
                <main className="flex flex-1 flex-col overflow-hidden">
                    {tab === 'chat' ? (
                        <>
                            {/* Chat messages */}
                            <div className="flex-1 overflow-y-auto px-8 py-8 space-y-6">
                                {messages.length === 0 && (
                                    <div className="h-full flex items-center justify-center">
                                        <div className="text-center space-y-3">
                                            <p className="text-2xl font-medium tracking-tight text-foreground/80">
                                                What would you like to know?
                                            </p>
                                            <p className="text-sm text-muted-foreground">
                                                Upload a document, then ask anything about it.
                                            </p>
                                        </div>
                                    </div>
                                )}
                                {messages.map((msg) => (
                                    <div
                                        key={msg.id}
                                        className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
                                    >
                                        <div
                                            className={[
                                                'text-sm leading-7',
                                                msg.role === 'user'
                                                    ? 'max-w-[70%] bg-muted rounded-3xl px-5 py-3 text-foreground'
                                                    : 'max-w-[85%] text-foreground prose prose-sm',
                                            ].join(' ')}
                                        >
                                            {msg.role === 'assistant'
                                                ? <ReactMarkdown>{msg.content}</ReactMarkdown>
                                                : msg.content
                                            }
                                            {msg.streaming && msg.content === '' && (
                                                <span className="inline-flex gap-1 items-center">
                                                    <span className="w-1 h-1 bg-muted-foreground rounded-full animate-bounce [animation-delay:0ms]" />
                                                    <span className="w-1 h-1 bg-muted-foreground rounded-full animate-bounce [animation-delay:150ms]" />
                                                    <span className="w-1 h-1 bg-muted-foreground rounded-full animate-bounce [animation-delay:300ms]" />
                                                </span>
                                            )}
                                            {msg.streaming && msg.content !== '' && (
                                                <span className="inline-block w-0.5 h-3.5 bg-foreground/70 ml-0.5 align-middle animate-pulse" />
                                            )}
                                        </div>
                                        {/* feedback buttons — only on completed assistant messages with a queryId */}
                                        {msg.role === 'assistant' && msg.queryId && !msg.streaming && (
                                            <div className="flex gap-1 mt-1 ml-1">
                                                {feedbackGiven.has(msg.queryId) ? (
                                                    <span className="text-[10px] text-muted-foreground/50">Thanks for the feedback</span>
                                                ) : (
                                                    <>
                                                        <button
                                                            onClick={() => sendFeedback(msg.queryId!, 1)}
                                                            className="text-muted-foreground/40 hover:text-green-500 transition-colors p-1 rounded"
                                                            title="Helpful"
                                                        >
                                                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                                                <path strokeLinecap="round" strokeLinejoin="round" d="M6.633 10.5c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 012.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 00.322-1.672V3a.75.75 0 01.75-.75A2.25 2.25 0 0116.5 4.5c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 01-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 00-1.423-.23H5.904M14.25 9h2.25M5.904 18.75c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 01-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 10.203 4.167 9.75 5 9.75h1.053c.472 0 .745.556.5.96a8.958 8.958 0 00-1.302 4.665c0 1.194.232 2.333.654 3.375z" />
                                                            </svg>
                                                        </button>
                                                        <button
                                                            onClick={() => sendFeedback(msg.queryId!, -1)}
                                                            className="text-muted-foreground/40 hover:text-red-500 transition-colors p-1 rounded"
                                                            title="Not helpful"
                                                        >
                                                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                                                <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 15h2.25m8.024-9.75c.011.05.028.1.052.148.591 1.2.924 2.55.924 3.977a8.96 8.96 0 01-.999 4.125m.023-8.25c-.076-.365.183-.75.575-.75h.908c.889 0 1.713.518 1.972 1.368.339 1.11.521 2.287.521 3.507 0 1.553-.295 3.036-.831 4.398C20.613 14.547 19.833 15 19 15h-1.053c-.472 0-.745-.556-.5-.96a8.95 8.95 0 00.303-.54m.023-8.25H16.48a4.5 4.5 0 01-1.423-.23l-3.114-1.04a4.5 4.5 0 00-1.423-.23H6.504c-.618 0-1.217.247-1.605.729A11.95 11.95 0 002.25 12c0 .434.023.863.068 1.285C2.427 14.306 3.346 15 4.372 15h3.126c.618 0 .991.724.725 1.282A7.471 7.471 0 007.5 19.5a2.25 2.25 0 002.25 2.25.75.75 0 00.75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 002.86-2.4c.498-.634 1.226-1.08 2.032-1.08h.384" />
                                                            </svg>
                                                        </button>
                                                    </>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                ))}
                                <div ref={messagesEndRef} />
                            </div>

                            {/* Chat input — pill-shaped floating bar */}
                            <div className="shrink-0 px-6 pb-5 pt-2">
                                <div className="flex items-end gap-3 rounded-full border border-border/70 bg-background shadow-[0_2px_16px_rgba(0,0,0,0.06)] px-5 py-3">
                                    <textarea
                                        value={question}
                                        onChange={(e) => setQuestion(e.target.value)}
                                        onKeyDown={handleChatKeyDown}
                                        placeholder={
                                            uploadResult
                                                ? 'Ask a question about your document...'
                                                : 'Upload a document first'
                                        }
                                        disabled={!uploadResult || streaming}
                                        rows={1}
                                        className="flex-1 resize-none bg-transparent text-sm placeholder:text-muted-foreground/60
                                            focus:outline-none disabled:opacity-40 disabled:cursor-not-allowed"
                                        style={{ lineHeight: '1.5rem', maxHeight: '8rem', overflowY: 'auto' }}
                                    />
                                    <button
                                        onClick={sendMessage}
                                        disabled={!question.trim() || streaming || !uploadResult}
                                        className="size-8 rounded-full bg-foreground text-background flex items-center justify-center
                                            shrink-0 disabled:opacity-20 disabled:cursor-not-allowed hover:opacity-80 transition-opacity"
                                    >
                                        {streaming ? (
                                            <div className="w-3 h-3 border-2 border-background/30 border-t-background rounded-full animate-spin" />
                                        ) : (
                                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5 12 3m0 0 7.5 7.5M12 3v18" />
                                            </svg>
                                        )}
                                    </button>
                                </div>
                                <p className="text-[10px] text-center text-muted-foreground/30 mt-2">
                                    Enter to send · Shift+Enter for new line
                                </p>
                            </div>
                        </>
                    ) : tab === 'analytics' ? (
                        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
                            {/* refresh button */}
                            <div className="flex items-center justify-between">
                                <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                                    LLMOps Dashboard
                                </p>
                                <button
                                    onClick={fetchAnalytics}
                                    disabled={analyticsLoading}
                                    className="text-[10px] text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
                                >
                                    {analyticsLoading ? 'Loading...' : 'Refresh'}
                                </button>
                            </div>

                            {analytics ? (
                                <>
                                    {/* Stat cards */}
                                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                                        {[
                                            { label: 'Total Queries', value: analytics.total_queries },
                                            { label: 'Cache Hits', value: analytics.cache_hits },
                                            { label: 'Cache Miss Rate', value: `${(analytics.cache_miss_rate * 100).toFixed(0)}%` },
                                            { label: 'Avg Latency', value: `${analytics.avg_latency_ms} ms` },
                                            { label: 'Total Tokens', value: analytics.total_tokens.toLocaleString() },
                                            { label: '👍 Helpful', value: analytics.thumbs_up },
                                            { label: '👎 Not Helpful', value: analytics.thumbs_down },
                                        ].map((stat) => (
                                            <div
                                                key={stat.label}
                                                className="rounded-lg border border-border bg-muted/20 px-4 py-3 space-y-1"
                                            >
                                                <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
                                                    {stat.label}
                                                </p>
                                                <p className="text-xl font-semibold tabular-nums">{stat.value}</p>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Recent queries table */}
                                    <div>
                                        <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-3">
                                            Recent Queries
                                        </p>
                                        {analytics.recent_queries.length === 0 ? (
                                            <p className="text-xs text-muted-foreground">No queries yet.</p>
                                        ) : (
                                            <div className="rounded-lg border border-border overflow-hidden">
                                                <table className="w-full text-xs">
                                                    <thead>
                                                        <tr className="border-b border-border bg-muted/30">
                                                            <th className="text-left px-4 py-2 text-muted-foreground font-medium">Question</th>
                                                            <th className="text-right px-4 py-2 text-muted-foreground font-medium whitespace-nowrap">Latency</th>
                                                            <th className="text-right px-4 py-2 text-muted-foreground font-medium">Cache</th>
                                                            <th className="text-right px-4 py-2 text-muted-foreground font-medium">Time</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {analytics.recent_queries.map((q, i) => (
                                                            <tr
                                                                key={i}
                                                                className="border-b border-border last:border-0 hover:bg-muted/10 transition-colors"
                                                            >
                                                                <td className="px-4 py-2.5 max-w-[300px] truncate">{q.question}</td>
                                                                <td className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">
                                                                    {q.cache_hit ? '—' : `${q.latency_ms} ms`}
                                                                </td>
                                                                <td className="px-4 py-2.5 text-right">
                                                                    {q.cache_hit ? (
                                                                        <span className="text-green-500">HIT</span>
                                                                    ) : (
                                                                        <span className="text-muted-foreground/50">MISS</span>
                                                                    )}
                                                                </td>
                                                                <td className="px-4 py-2.5 text-right text-muted-foreground whitespace-nowrap">
                                                                    {new Date(q.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                                </td>
                                                            </tr>
                                                        ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        )}
                                    </div>
                                </>
                            ) : (
                                <div className="flex items-center justify-center h-40">
                                    <p className="text-sm text-muted-foreground">
                                        {analyticsLoading ? 'Loading...' : 'No data yet.'}
                                    </p>
                                </div>
                            )}
                        </div>
                    ) : (
                        <>
                            {/* Agent execution tracker */}
                            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
                                {agentRuns.length === 0 && (
                                    <div className="h-full flex items-center justify-center">
                                        <div className="text-center space-y-1.5">
                                            <p className="text-sm text-muted-foreground">
                                                Give the agent a task to complete.
                                            </p>
                                            <p className="text-xs text-muted-foreground/50">
                                                It will reason, use tools, and report back.
                                            </p>
                                        </div>
                                    </div>
                                )}

                                {agentRuns.map((run) => (
                                    <div key={run.id} className="space-y-2">
                                        {/* Task */}
                                        <div className="flex justify-end">
                                            <div className="max-w-[75%] rounded-xl px-4 py-2.5 text-sm bg-primary text-primary-foreground">
                                                {run.task}
                                            </div>
                                        </div>

                                        {/* Execution steps */}
                                        <div className="space-y-1.5 pl-2">
                                            {run.steps.map((step) => (
                                                <div key={step.id}>
                                                    {step.type === 'tool_call' && (
                                                        <div className="flex items-start gap-2 text-xs text-muted-foreground">
                                                            <div className="w-4 h-4 mt-0.5 shrink-0 rounded border border-border flex items-center justify-center">
                                                                <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />
                                                            </div>
                                                            <span>
                                                                Calling{' '}
                                                                <span className="font-mono text-foreground">
                                                                    {step.tool}
                                                                </span>
                                                                {step.input && (
                                                                    <span className="text-muted-foreground/70">
                                                                        {' '}with{' '}
                                                                        <span className="font-mono">
                                                                            {step.input.slice(0, 80)}
                                                                        </span>
                                                                    </span>
                                                                )}
                                                            </span>
                                                        </div>
                                                    )}
                                                    {step.type === 'tool_result' && (
                                                        <div className="flex items-start gap-2 text-xs text-muted-foreground">
                                                            <div className="w-4 h-4 mt-0.5 shrink-0 rounded border border-border flex items-center justify-center">
                                                                <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
                                                            </div>
                                                            <span className="font-mono text-[10px] bg-muted/40 rounded px-2 py-1 leading-relaxed line-clamp-2">
                                                                {step.content}
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            ))}

                                            {/* Running indicator */}
                                            {run.running && run.answer === '' && (
                                                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                                    <div className="w-4 h-4 shrink-0 rounded border border-border flex items-center justify-center">
                                                        <div className="w-2.5 h-2.5 border border-muted-foreground border-t-foreground rounded-full animate-spin" />
                                                    </div>
                                                    <span>Thinking...</span>
                                                </div>
                                            )}
                                        </div>

                                        {/* Final answer */}
                                        {run.answer && (
                                            <div className="bg-muted text-foreground border border-border rounded-xl px-4 py-2.5 text-sm leading-relaxed prose prose-sm max-w-none">
                                                <ReactMarkdown>{run.answer}</ReactMarkdown>
                                                {run.running && (
                                                    <span className="inline-block w-0.5 h-3.5 bg-foreground/70 ml-0.5 align-middle animate-pulse" />
                                                )}
                                            </div>
                                        )}
                                    </div>
                                ))}
                                <div ref={agentEndRef} />
                            </div>

                            {/* Agent input */}
                            <div className="shrink-0 border-t border-border p-4">
                                <div className="flex gap-2 items-end">
                                    <textarea
                                        value={agentTask}
                                        onChange={(e) => setAgentTask(e.target.value)}
                                        onKeyDown={handleAgentKeyDown}
                                        placeholder="Give the agent a task..."
                                        disabled={agentRunning}
                                        rows={1}
                                        className="flex-1 resize-none rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm
                                            placeholder:text-muted-foreground focus:outline-none focus:border-muted-foreground
                                            disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                                        style={{ lineHeight: '1.5rem', maxHeight: '8rem', overflowY: 'auto' }}
                                    />
                                    <button
                                        onClick={runAgent}
                                        disabled={!agentTask.trim() || agentRunning}
                                        className="size-9 rounded-lg bg-primary text-primary-foreground flex items-center justify-center
                                            shrink-0 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-primary/80 transition-colors"
                                    >
                                        {agentRunning ? (
                                            <div className="w-3.5 h-3.5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                                        ) : (
                                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                                            </svg>
                                        )}
                                    </button>
                                </div>
                                <p className="text-[10px] text-muted-foreground/40 mt-2 text-right">
                                    Enter to send · Shift+Enter for new line
                                </p>
                            </div>
                        </>
                    )}
                </main>
            </div>
        </div>
    )
}
