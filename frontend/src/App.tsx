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

export default function App() {
    const [tab, setTab] = useState<'chat' | 'agent'>('chat')

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

    return (
        <div className="flex flex-col h-screen bg-background text-foreground">
            {/* Header */}
            <header className="shrink-0 border-b border-border px-6 h-12 flex items-center gap-3">
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
                </div>
            </header>

            {/* Body */}
            <div className="flex flex-1 overflow-hidden">
                {/* LEFT: Upload panel */}
                <aside className="w-72 shrink-0 border-r border-border flex flex-col p-5 gap-5">
                    <div>
                        <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-3">
                            Document
                        </p>

                        <div
                            {...getRootProps()}
                            className={[
                                'border-2 border-dashed rounded-lg p-6 cursor-pointer transition-colors',
                                'flex flex-col items-center justify-center gap-2 text-center min-h-[140px]',
                                isDragActive
                                    ? 'border-[var(--accent-border)] bg-[var(--accent-bg)]'
                                    : 'border-border hover:border-muted-foreground hover:bg-muted/20',
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
                                <div key={doc.filename} className="rounded-lg border border-border bg-muted/20 p-3 space-y-1">
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
                            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
                                {messages.length === 0 && (
                                    <div className="h-full flex items-center justify-center">
                                        <div className="text-center space-y-1.5">
                                            <p className="text-sm text-muted-foreground">
                                                Upload a document, then ask a question.
                                            </p>
                                            <p className="text-xs text-muted-foreground/50">
                                                Answers are grounded in your document.
                                            </p>
                                        </div>
                                    </div>
                                )}
                                {messages.map((msg) => (
                                    <div
                                        key={msg.id}
                                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                    >
                                        <div
                                            className={[
                                                'max-w-[75%] rounded-xl px-4 py-2.5 text-sm leading-relaxed',
                                                msg.role === 'user'
                                                    ? 'bg-primary text-primary-foreground'
                                                    : 'bg-muted text-foreground border border-border prose prose-sm max-w-none',
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
                                    </div>
                                ))}
                                <div ref={messagesEndRef} />
                            </div>

                            {/* Chat input */}
                            <div className="shrink-0 border-t border-border p-4">
                                <div className="flex gap-2 items-end">
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
                                        className="flex-1 resize-none rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm
                                            placeholder:text-muted-foreground focus:outline-none focus:border-muted-foreground
                                            disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                                        style={{ lineHeight: '1.5rem', maxHeight: '8rem', overflowY: 'auto' }}
                                    />
                                    <button
                                        onClick={sendMessage}
                                        disabled={!question.trim() || streaming || !uploadResult}
                                        className="size-9 rounded-lg bg-primary text-primary-foreground flex items-center justify-center
                                            shrink-0 disabled:opacity-30 disabled:cursor-not-allowed hover:bg-primary/80 transition-colors"
                                    >
                                        {streaming ? (
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
