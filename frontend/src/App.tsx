import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type ThemeMode = 'light' | 'dark'

type LanguageMeta = {
  code: string
  label: string
  nllb_code: string
  script: string
}

type HealthResponse = {
  status: string
  model_status: string
  mode: string
  safetensors: boolean
  hf_transfer: boolean
}

type CandidateBreakdown = {
  punctuation: number
  entities: number
  length: number
  target_script: number
  confidence: number
  total: number
}

type TranslationCandidate = {
  candidate_id: string
  strategy: string
  text: string
  confidence: number
  score: number
  breakdown: CandidateBreakdown
  notes: string[]
}

type TranslationResponse = {
  source_language: string
  target_language: string
  pair_label: string
  input_text: string
  selected_candidate: TranslationCandidate
  candidates: TranslationCandidate[]
  model_status: string
  retry_used: boolean
  diagnostics: {
    candidate_count: number
    selected_strategy: string
    selected_total_score: number
    selected_confidence: number
  }
}

const sampleText = 'Please translate this sentence into a natural Indian-language form.'

const configuredApiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '')
const apiBaseUrl = configuredApiBase ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')

function apiUrl(path: string): string {
  return apiBaseUrl ? `${apiBaseUrl}${path}` : path
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="4.5" />
      <path d="M12 2.5v2.3M12 19.2v2.3M4.8 4.8l1.6 1.6M17.6 17.6l1.6 1.6M2.5 12h2.3M19.2 12h2.3M4.8 19.2l1.6-1.6M17.6 6.4l1.6-1.6" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M16.5 14.8A7.2 7.2 0 0 1 9.2 7.5c0-1.2.3-2.3.8-3.3A8.4 8.4 0 1 0 19.7 17c-1.1-.4-2.2-.9-3.2-2.2Z" />
    </svg>
  )
}

function SwapIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 7h10l-2.5-2.5M17 17H7l2.5 2.5" />
    </svg>
  )
}

function SpinnerIcon() {
  return <span className="spinner-icon" aria-hidden="true" />
}

function CopyIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M9 9.5V7.2A2.2 2.2 0 0 1 11.2 5h5.6A2.2 2.2 0 0 1 19 7.2v5.6a2.2 2.2 0 0 1-2.2 2.2h-2.3" />
      <path d="M13 9H7.2A2.2 2.2 0 0 0 5 11.2v5.6A2.2 2.2 0 0 0 7.2 19h5.6a2.2 2.2 0 0 0 2.2-2.2V11.4" />
    </svg>
  )
}

function SpeakIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M11 6 7.5 9H5v6h2.5L11 18V6Z" />
      <path d="M14.5 9.5a4 4 0 0 1 0 5" />
      <path d="M17 7a7 7 0 0 1 0 10" />
    </svg>
  )
}

function ShareIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8.5 13.5 15.5 18" />
      <path d="M15.5 6 8.5 10.5" />
      <circle cx="6.5" cy="12" r="2" />
      <circle cx="17.5" cy="6" r="2" />
      <circle cx="17.5" cy="18" r="2" />
    </svg>
  )
}

function App() {
  const [languages, setLanguages] = useState<LanguageMeta[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [sourceLanguage, setSourceLanguage] = useState<string>('')
  const [targetLanguage, setTargetLanguage] = useState<string>('')
  const [text, setText] = useState(sampleText)
  const [response, setResponse] = useState<TranslationResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [themeMode, setThemeMode] = useState<ThemeMode>('light')
  const [maxCandidates, setMaxCandidates] = useState(3)
  const [showAllCandidates, setShowAllCandidates] = useState(false)
  const [languageMenuOpen, setLanguageMenuOpen] = useState<'source' | 'target' | null>(null)
  const [infoOpen, setInfoOpen] = useState(false)
  const [copied, setCopied] = useState(false)

  const languageControlsRef = useRef<HTMLDivElement | null>(null)
  const infoRef = useRef<HTMLDivElement | null>(null)

  const languageByCode = useMemo(() => {
    return languages.reduce<Record<string, LanguageMeta>>((accumulator, language) => {
      accumulator[language.code] = language
      return accumulator
    }, {})
  }, [languages])

  const sourceLanguageMeta = sourceLanguage ? languageByCode[sourceLanguage] : undefined
  const targetLanguageMeta = targetLanguage ? languageByCode[targetLanguage] : undefined
  const outputText = response?.selected_candidate.text ?? ''
  const wordCount = useMemo(() => text.trim().split(/\s+/).filter(Boolean).length, [text])
  const characterCount = text.length

  useEffect(() => {
    const savedTheme = window.localStorage.getItem('theme-mode')
    if (savedTheme === 'light' || savedTheme === 'dark') {
      setThemeMode(savedTheme)
      return
    }

    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    setThemeMode(prefersDark ? 'dark' : 'light')
  }, [])

  useEffect(() => {
    window.localStorage.setItem('theme-mode', themeMode)
  }, [themeMode])

  useEffect(() => {
    const loadData = async () => {
      try {
        const [languagesResponse, healthResponse] = await Promise.all([
          fetch(apiUrl('/api/languages')),
          fetch(apiUrl('/api/health')),
        ])

        if (languagesResponse.ok) {
          const payload = (await languagesResponse.json()) as { languages: LanguageMeta[] }
          setLanguages(payload.languages ?? [])
        }

        if (healthResponse.ok) {
          const payload = (await healthResponse.json()) as HealthResponse
          setHealth(payload)
        } else {
          setHealth({ status: 'offline', model_status: 'backend unavailable', mode: 'unknown', safetensors: false, hf_transfer: false })
        }
      } catch {
        setHealth({ status: 'offline', model_status: 'backend unavailable', mode: 'unknown', safetensors: false, hf_transfer: false })
      }
    }

    void loadData()
  }, [])

  useEffect(() => {
    if (languages.length === 0) {
      return
    }

    setSourceLanguage((current) => current && languages.some((language) => language.code === current) ? current : (languages.find((language) => language.code === 'en')?.code ?? languages[0]?.code ?? ''))
    setTargetLanguage((current) => {
      if (current && languages.some((language) => language.code === current)) {
        return current
      }

      const preferredTarget = languages.find((language) => language.code === 'hi')?.code
      const fallbackTarget = languages.find((language) => language.code !== sourceLanguage)?.code ?? languages[1]?.code ?? languages[0]?.code ?? ''
      return preferredTarget ?? fallbackTarget
    })
  }, [languages, sourceLanguage])

  useEffect(() => {
    if (!languageMenuOpen && !infoOpen) {
      return undefined
    }

    const handlePointerDown = (event: MouseEvent) => {
      const targetNode = event.target as Node
      if (languageMenuOpen && languageControlsRef.current && !languageControlsRef.current.contains(targetNode)) {
        setLanguageMenuOpen(null)
      }
      if (infoOpen && infoRef.current && !infoRef.current.contains(targetNode)) {
        setInfoOpen(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    return () => document.removeEventListener('mousedown', handlePointerDown)
  }, [languageMenuOpen, infoOpen])

  useEffect(() => {
    if (!copied) {
      return undefined
    }

    const timer = window.setTimeout(() => setCopied(false), 1600)
    return () => window.clearTimeout(timer)
  }, [copied])

  const openLanguageMenu = (mode: 'source' | 'target') => {
    setLanguageMenuOpen((current) => (current === mode ? null : mode))
  }

  const selectLanguage = (code: string) => {
    if (languageMenuOpen === 'source') {
      setSourceLanguage(code)
    } else if (languageMenuOpen === 'target') {
      setTargetLanguage(code)
    }
    setLanguageMenuOpen(null)
    setResponse(null)
  }

  const swapLanguages = () => {
    setSourceLanguage(targetLanguage)
    setTargetLanguage(sourceLanguage)
    setResponse(null)
  }

  const translateText = async () => {
    setLoading(true)
    setError(null)

    try {
      const apiResponse = await fetch(apiUrl('/api/translate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          source_language: sourceLanguage,
          target_language: targetLanguage,
          max_candidates: maxCandidates,
        }),
      })

      if (!apiResponse.ok) {
        const payload = (await apiResponse.json().catch(() => null)) as { detail?: string } | null
        throw new Error(payload?.detail ?? 'Translation failed')
      }

      const payload = (await apiResponse.json()) as TranslationResponse
      setResponse(payload)
      setShowAllCandidates(false)
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : 'Translation failed')
    } finally {
      setLoading(false)
    }
  }

  const copyTranslation = async () => {
    if (!outputText) {
      return
    }

    await navigator.clipboard.writeText(outputText)
    setCopied(true)
  }

  const speakTranslation = () => {
    if (!outputText || !('speechSynthesis' in window)) {
      return
    }

    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(outputText)
    utterance.lang = targetLanguage === 'hi' ? 'hi-IN' : 'en-IN'
    window.speechSynthesis.speak(utterance)
  }

  const shareTranslation = async () => {
    if (!outputText) {
      return
    }

    if (navigator.share) {
      await navigator.share({
        title: 'Vakya',
        text: outputText,
        url: window.location.href,
      })
      return
    }

    await navigator.clipboard.writeText(outputText)
    setCopied(true)
  }

  const apiOnline = health?.status === 'ok'

  return (
    <main className={`shell theme-${themeMode}`}>
      <header className="topbar">
        <div className="brand-block">
          <span className="site-name">Vakya</span>
        </div>

        <div className="topbar-right" ref={infoRef}>
          <div className={apiOnline ? 'status-chip status-online' : 'status-chip status-offline'}>
            <span className="status-dot" />
            <span>{apiOnline ? 'API online' : 'API offline'}</span>
          </div>
          <button className="icon-button theme-button" type="button" onClick={() => setThemeMode((current) => (current === 'dark' ? 'light' : 'dark'))} aria-label="Toggle theme">
            {themeMode === 'dark' ? <SunIcon /> : <MoonIcon />}
          </button>
          <button className="icon-button info-button" type="button" onClick={() => setInfoOpen((current) => !current)} aria-label="Show API information">
            ⓘ
          </button>
          {infoOpen ? (
            <div className="header-info" role="note">
              <span>
                mode <code>{health?.mode ?? 'unknown'}</code>
              </span>
              <span>
                safetensors <code>{health ? String(health.safetensors) : 'false'}</code>
              </span>
              <span>
                hf_transfer <code>{health ? String(health.hf_transfer) : 'false'}</code>
              </span>
            </div>
          ) : null}
        </div>
      </header>

      <section className="workspace">
        <article className="panel panel-input">
          <div className="panel-head">
            <span className="panel-title">Input</span>
            <button className="load-sample" type="button" onClick={() => setText(sampleText)}>
              Load sample
            </button>
          </div>

          <div className="language-controls" ref={languageControlsRef}>
            <div className="language-picker">
              <button className="language-pill" type="button" onClick={() => openLanguageMenu('source')}>
                <span className="pill-label">{sourceLanguageMeta?.label ?? 'Select language'}</span>
                <span className="pill-script">{sourceLanguageMeta?.script ?? 'script'}</span>
              </button>
              {languageMenuOpen === 'source' ? (
                <div className="language-menu">
                  {languages.map((language) => (
                    <button key={language.code} className={language.code === sourceLanguage ? 'language-menu-item active' : 'language-menu-item'} type="button" onClick={() => selectLanguage(language.code)}>
                      <span>{language.label}</span>
                      <small>{language.script}</small>
                    </button>
                  ))}
                </div>
              ) : null}
            </div>

            <button className="swap-control" type="button" onClick={swapLanguages} aria-label="Swap languages">
              <SwapIcon />
            </button>

            <div className="language-picker">
              <button className="language-pill" type="button" onClick={() => openLanguageMenu('target')}>
                <span className="pill-label">{targetLanguageMeta?.label ?? 'Select language'}</span>
                <span className="pill-script">{targetLanguageMeta?.script ?? 'script'}</span>
              </button>
              {languageMenuOpen === 'target' ? (
                <div className="language-menu">
                  {languages.map((language) => (
                    <button key={language.code} className={language.code === targetLanguage ? 'language-menu-item active' : 'language-menu-item'} type="button" onClick={() => selectLanguage(language.code)}>
                      <span>{language.label}</span>
                      <small>{language.script}</small>
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </div>

          <div className="candidates-stepper-row">
            <span>Candidates</span>
            <div className="stepper" aria-label="Maximum candidates">
              <button type="button" onClick={() => setMaxCandidates((current) => Math.max(1, current - 1))} aria-label="Decrease candidates">
                −
              </button>
              <span>{maxCandidates}</span>
              <button type="button" onClick={() => setMaxCandidates((current) => Math.min(5, current + 1))} aria-label="Increase candidates">
                +
              </button>
            </div>
          </div>

          <label className="field-label">
            <span>Text</span>
            <textarea
              className="input-field"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Type or paste text here"
              rows={10}
            />
          </label>

          <div className="input-footer">
            <div className="word-count">
              <span>{wordCount} words</span>
              {characterCount > 500 ? <span>{characterCount} chars</span> : null}
            </div>
            <button className="translate-button" type="button" onClick={translateText} disabled={loading || text.trim().length === 0 || !sourceLanguage || !targetLanguage}>
              {loading ? (
                <>
                  <SpinnerIcon />
                  <span>Translating…</span>
                </>
              ) : (
                <span>Translate</span>
              )}
            </button>
          </div>

          {error ? <p className="error-line">{error}</p> : null}
        </article>

        <article className="panel panel-output">
          <div className="panel-head">
            <span className="panel-title">Output</span>
            {response ? <span className="retry-note">{response.retry_used ? 'retry used' : 'direct select'}</span> : null}
          </div>

          <div className={response ? 'output-body output-filled' : 'output-body output-empty'}>
            {response ? (
              <>
                <div className="pair-label">{response.pair_label}</div>
                <p className="translated-text">{response.selected_candidate.text}</p>

                <div className="score-row">
                  <span className="score-item">
                    <span>Punctuation</span>
                    <code>{response.selected_candidate.breakdown.punctuation.toFixed(2)}</code>
                  </span>
                  <span className="score-item">
                    <span>Entities</span>
                    <code>{response.selected_candidate.breakdown.entities.toFixed(2)}</code>
                  </span>
                  <span className="score-item">
                    <span>Length</span>
                    <code>{response.selected_candidate.breakdown.length.toFixed(2)}</code>
                  </span>
                  <span className="score-item">
                    <span>Script</span>
                    <code>{response.selected_candidate.breakdown.target_script.toFixed(2)}</code>
                  </span>
                  <span className="score-item">
                    <span>Confidence</span>
                    <code>{response.selected_candidate.breakdown.confidence.toFixed(2)}</code>
                  </span>
                </div>

                <div className="diagnostics-line">
                  {response.diagnostics.candidate_count} candidates evaluated · {response.diagnostics.selected_strategy} strategy selected · score {response.diagnostics.selected_total_score.toFixed(2)}
                </div>

                {response.retry_used ? <div className="retry-note-inline">Retry was used to improve this result</div> : null}

                <button className="candidate-toggle" type="button" onClick={() => setShowAllCandidates((current) => !current)}>
                  {showAllCandidates ? 'Hide all candidates' : 'Show all candidates'}
                </button>

                {showAllCandidates ? (
                  <div className="candidate-list">
                    {response.candidates.map((candidate) => {
                      const isActive = candidate.candidate_id === response.selected_candidate.candidate_id
                      return (
                        <div key={candidate.candidate_id} className={isActive ? 'candidate-row active' : 'candidate-row'}>
                          <div className="candidate-top">
                            <span>{candidate.strategy}</span>
                            <span className="candidate-score">{candidate.score.toFixed(2)}</span>
                          </div>
                          <p className="candidate-text">{candidate.text}</p>
                          <div className="candidate-notes">{candidate.notes.join(', ')}</div>
                        </div>
                      )
                    })}
                  </div>
                ) : null}
              </>
            ) : (
              <div className="empty-state">Translation will appear here</div>
            )}

            {response ? (
              <div className="output-actions">
                <button type="button" className="action-icon" onClick={copyTranslation} aria-label="Copy translation">
                  <CopyIcon />
                </button>
                <button type="button" className="action-icon" onClick={speakTranslation} aria-label="Speak translation">
                  <SpeakIcon />
                </button>
                <button type="button" className="action-icon" onClick={shareTranslation} aria-label="Share translation">
                  <ShareIcon />
                </button>
              </div>
            ) : null}

            {copied ? <div className="copy-toast">Copied</div> : null}
          </div>
        </article>
      </section>
    </main>
  )
}

export default App