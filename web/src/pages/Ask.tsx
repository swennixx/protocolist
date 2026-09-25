import { useEffect, useRef, useState } from 'react'
import { ArrowUp, MessageSquare, Sparkles, Square } from 'lucide-react'
import { ask, type Hit, type MeetingBrief } from '../api'
import { fmtDate, fmtTime } from '../util'

type Msg = { role: 'user' | 'bot'; text: string; sources: Hit[]; pending?: boolean; error?: boolean }

const SUGGESTIONS = ['Какие решения приняли?', 'Кто за что отвечает и к какому сроку?', 'Какие сроки называли?', 'Какие цифры по бюджету звучали?']

const citeHref = (h: Hit) => `#/m/${h.meeting_id}?t=${Math.floor(h.start)}`

export default function Ask({ meetings }: { meetings: MeetingBrief[] }) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [q, setQ] = useState('')
  const [scope, setScope] = useState('all')
  const end = useRef<HTMLDivElement>(null)
  const abort = useRef<AbortController | null>(null)
  const done = meetings.filter((m) => m.status === 'done')
  const streaming = msgs.at(-1)?.pending ?? false

  useEffect(() => {
    end.current?.scrollIntoView({ block: 'end', behavior: 'smooth' })
  }, [msgs])

  useEffect(() => () => abort.current?.abort(), [])

  const updateLast = (fn: (m: Msg) => Msg) => setMsgs((ms) => [...ms.slice(0, -1), fn(ms[ms.length - 1])])

  const send = async (text: string) => {
    const question = text.trim()
    if (question.length < 2 || streaming) return
    setMsgs((m) => [...m, { role: 'user', text: question, sources: [] }, { role: 'bot', text: '', sources: [], pending: true }])
    setQ('')
    abort.current = new AbortController()
    try {
      await ask(
        question,
        scope === 'all' ? null : scope,
        {
          sources: (h) => updateLast((m) => ({ ...m, sources: h })),
          token: (t) => updateLast((m) => ({ ...m, text: m.text + t })),
        },
        abort.current.signal,
      )
      updateLast((m) => ({ ...m, pending: false }))
    } catch (e) {
      const stopped = abort.current?.signal.aborted
      updateLast((m) => ({ ...m, pending: false, error: !stopped, text: stopped ? m.text : `Не удалось получить ответ: ${(e as Error).message}` }))
    }
  }

  return (
    <div className="page ask">
      <header className="page-head">
        <div>
          <h1>Спросить архив</h1>
          <p className="muted">Ответ строится только по записанным встречам — каждое утверждение со ссылкой на фрагмент</p>
        </div>
        <select className="scope" value={scope} onChange={(e) => setScope(e.target.value)} aria-label="Где искать">
          <option value="all">Все встречи</option>
          {done.map((m) => (
            <option key={m.id} value={m.id}>
              {m.title}
            </option>
          ))}
        </select>
      </header>

      <div className="chat">
        {msgs.length === 0 && (
          <div className="empty">
            <span className="empty-icon">
              <MessageSquare size={26} />
            </span>
            <h2>О чём договорились?</h2>
            <p className="muted">{done.length ? 'Задайте вопрос своими словами — например:' : 'Сначала загрузите хотя бы одну встречу.'}</p>
            {done.length > 0 && (
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {msgs.map((msg, i) =>
          msg.role === 'user' ? (
            <div key={i} className="msg user">
              {msg.text}
            </div>
          ) : (
            <div key={i} className="msg bot">
              <span className="bot-icon">
                <Sparkles size={15} />
              </span>
              <div className="bot-body">
                <p className={msg.error ? 'form-error' : ''}>
                  {msg.text.split(/(\[\d+\])/).map((part, j) => {
                    const n = Number(part.match(/^\[(\d+)\]$/)?.[1])
                    const s = msg.sources[n - 1]
                    if (!n) return part
                    return s ? (
                      <a key={j} className="cite" href={citeHref(s)} title={`${s.meeting_title}, ${fmtTime(s.start)}`}>
                        {n}
                      </a>
                    ) : null // a citation the model invented: drop it
                  })}
                  {msg.pending && !msg.text && <span className="muted">Ищу во встречах…</span>}
                  {msg.pending && <span className="caret" />}
                </p>
                {!msg.pending && msg.sources.length > 0 && (
                  <Sources hits={msg.sources} cited={new Set([...msg.text.matchAll(/\[(\d+)\]/g)].map((x) => Number(x[1])))} />
                )}
              </div>
            </div>
          ),
        )}
        <div ref={end} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          send(q)
        }}
      >
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Спросите о решениях, задачах, сроках…" aria-label="Вопрос" maxLength={1000} />
        {streaming ? (
          <button type="button" className="send" onClick={() => abort.current?.abort()} aria-label="Остановить">
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button className="send" disabled={q.trim().length < 2} aria-label="Отправить">
            <ArrowUp size={18} strokeWidth={2.5} />
          </button>
        )}
      </form>
    </div>
  )
}

function Sources({ hits, cited }: { hits: Hit[]; cited: Set<number> }) {
  const [all, setAll] = useState(false)
  // Show what the answer cites; the rest of the retrieved fragments stay one click away.
  const list = hits.map((h, i) => ({ h, n: i + 1 })).filter(({ n }) => all || cited.has(n) || cited.size === 0)
  return (
    <div className="sources">
      {list.map(({ h, n }) => (
        <a key={n} className="source" href={citeHref(h)}>
          <div className="source-head">
            <span className="cite static">{n}</span>
            <b>{h.meeting_title}</b>
            <span className="muted small">
              {fmtDate(h.date)} · <span className="mono">{fmtTime(h.start)}</span>
            </span>
          </div>
          <p>
            <span className={`speaker-name sp-text-${h.color}`}>{h.speaker}:</span> {h.text}
          </p>
        </a>
      ))}
      {!all && list.length < hits.length && (
        <button className="more" onClick={() => setAll(true)}>
          Показать все найденные фрагменты ({hits.length})
        </button>
      )}
    </div>
  )
}
