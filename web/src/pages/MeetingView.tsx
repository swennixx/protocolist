import { useEffect, useRef, useState } from 'react'
import {
  ArrowLeft, Check, Copy, Download, Gavel, ListChecks, Pause, Pencil, Play, RotateCcw, RotateCw, Search, Sparkles, Trash2,
  Users, X,
} from 'lucide-react'
import { api, watch, type Meeting, type Progress } from '../api'
import { STAGES, fmtDate, fmtTime, initials, isBusy, plural, speakerOf } from '../util'
import { Avatars, Highlight, Pipeline } from './Meetings'

type Tab = 'summary' | 'decisions' | 'tasks' | 'speakers'
const RATES = [0.75, 1, 1.25, 1.5, 2]

export default function MeetingView({ id, startAt, onChange }: { id: string; startAt: number; onChange: () => void }) {
  const [m, setM] = useState<Meeting | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<Progress | null>(null)

  useEffect(() => {
    api.meeting(id).then(setM, (e) => setError(e.message))
  }, [id])

  const busy = m ? isBusy(m) : false
  useEffect(() => {
    if (!busy) return
    let stage = -1
    return watch(id, (p) => {
      setProgress(p)
      // Transcript appears after diarization (stage 3); the rest fills in when analysis finishes.
      const changed = p.stage !== stage
      stage = p.stage
      if (p.status === 'done' || p.status === 'error' || (changed && p.stage >= 3)) api.meeting(id).then(setM)
    })
  }, [busy, id])

  if (error) return <div className="page"><p className="form-error">{error}</p><a href="#/" className="back"><ArrowLeft size={16} /> К списку встреч</a></div>
  if (!m) return <div className="page muted">Загружаем встречу…</div>
  if (!m.segments.length)
    return (
      <div className="page narrow-page">
        <a href="#/" className="back"><ArrowLeft size={16} /> Встречи</a>
        <h1>{m.title}</h1>
        <p className="muted head-gap">Стенограмма появится после диаризации — страница обновится сама.</p>
        <div className="panel">
          <Pipeline p={progress ?? m} />
          {(progress?.error ?? m.error) && <p className="form-error">{progress?.error ?? m.error}</p>}
        </div>
      </div>
    )
  return <Loaded m={m} setM={setM} startAt={startAt} busy={busy} progress={progress} onChange={onChange} />
}

function Loaded({ m, setM, startAt, busy, progress, onChange }: {
  m: Meeting
  setM: (fn: (m: Meeting | null) => Meeting | null) => void
  startAt: number
  busy: boolean
  progress: Progress | null
  onChange: () => void
}) {
  const d = m.duration || (m.segments.at(-1)?.end ?? 1)
  const audio = useRef<HTMLAudioElement>(null)
  const [time, setTime] = useState(startAt)
  const [playing, setPlaying] = useState(false)
  const [rate, setRate] = useState(1)
  const [mode, setMode] = useState<'clean' | 'verbatim'>('clean')
  const [tab, setTab] = useState<Tab>('summary')
  const [q, setQ] = useState('')
  const [editing, setEditing] = useState<number | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const segRefs = useRef<(HTMLDivElement | null)[]>([])
  const follow = useRef(true)

  const patch = (fn: (m: Meeting) => Meeting) => setM((x) => (x ? fn(x) : x))

  const seek = (t: number) => {
    follow.current = true
    const nt = Math.max(0, Math.min(d, t))
    if (audio.current) audio.current.currentTime = nt
    setTime(nt)
  }

  useEffect(() => {
    if (audio.current && startAt) audio.current.currentTime = startAt
  }, [startAt])

  useEffect(() => {
    if (audio.current) audio.current.playbackRate = rate
  }, [rate])

  // Smooth cursor: read currentTime every frame while playing (timeupdate fires only ~4 times a second).
  useEffect(() => {
    if (!playing) return
    let raf = 0
    const tick = () => {
      if (audio.current) setTime(audio.current.currentTime)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing])

  const toggle = () => {
    const a = audio.current
    if (!a) return
    if (a.paused) a.play().catch(() => setPlaying(false))
    else a.pause()
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, select, button')) return
      if (e.code === 'Space') {
        e.preventDefault()
        toggle()
      } else if (e.key === 'ArrowLeft') seek(time - 10)
      else if (e.key === 'ArrowRight') seek(time + 10)
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  })

  const active = m.segments.findIndex((s) => time >= s.start - 0.2 && time <= s.end + 0.3)

  useEffect(() => {
    if (active >= 0 && follow.current) segRefs.current[active]?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [active])

  const text = (s: Meeting['segments'][number]) => (mode === 'clean' ? s.clean : s.text)
  const query = q.trim().toLowerCase()
  const matches = query ? m.segments.flatMap((s, i) => (text(s).toLowerCase().includes(query) ? [i] : [])) : []
  const nextMatch = () => {
    if (!matches.length) return
    const next = matches.find((i) => m.segments[i].start > time + 0.5) ?? matches[0]
    seek(m.segments[next].start)
  }

  const talk = m.speakers.map((s) => {
    const segs = m.segments.filter((x) => x.speaker === s.id)
    return { s, secs: segs.reduce((a, x) => a + x.end - x.start, 0), count: segs.length }
  })
  const talkTotal = talk.reduce((a, x) => a + x.secs, 0) || 1

  // Waveform bars coloured by who speaks at that moment.
  const bars = m.peaks.map((h, i) => {
    const t = ((i + 0.5) / m.peaks.length) * d
    const seg = m.segments.find((s) => t >= s.start && t <= s.end)
    return { h: Math.max(0.06, h), color: seg ? speakerOf(m, seg.speaker)?.color ?? -1 : -1 }
  })

  const owner = (t: Meeting['tasks'][number]) => speakerOf(m, t.owner)
  const segAt = (i: number) => m.segments[Math.min(i, m.segments.length - 1)]

  const copyTasks = async () => {
    const lines = m.tasks
      .filter((t) => !t.done)
      .map((t) => `• ${t.text} — ${owner(t)?.name ?? t.owner_name ?? 'без исполнителя'}${t.due ? `, до ${fmtDate(t.due)}` : ''}`)
    try {
      await navigator.clipboard.writeText(`Задачи по встрече «${m.title}»:\n${lines.join('\n')}`)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* clipboard blocked */
    }
  }

  const remove = async () => {
    if (!confirm(`Удалить встречу «${m.title}» вместе с записью и стенограммой?`)) return
    await api.remove(m.id)
    onChange()
    location.hash = '#/'
  }

  const reprocess = async (stage: 'asr' | 'analyze') => {
    const what = stage === 'asr' ? 'Распознать запись заново? Правки текста и имена спикеров сбросятся.' : 'Пересобрать выжимку, решения и задачи?'
    if (!confirm(what)) return
    setMenuOpen(false)
    try {
      const b = await api.reprocess(m.id, stage)
      setM((x) => (x ? { ...x, ...b, decisions: x.decisions } : x))
      onChange()
    } catch (e) {
      alert((e as Error).message)
    }
  }

  const processingSecs = Object.entries(m.timings).reduce((a, [, v]) => a + (typeof v === 'number' ? v : 0), 0)

  const TimeChip = ({ seg }: { seg: number }) => (
    <button className="chip mono link" onClick={() => seek(segAt(seg).start)} title="Перейти к фрагменту">
      <Play size={10} fill="currentColor" /> {fmtTime(segAt(seg).start)}
    </button>
  )

  return (
    <div className="meeting">
      <header className="meeting-head">
        <a href="#/" className="back">
          <ArrowLeft size={16} /> Встречи
        </a>
        <div className="meeting-title">
          <div>
            <h1>{m.title}</h1>
            <p className="muted">
              {fmtDate(m.date, { day: 'numeric', month: 'long', weekday: 'long' })} · {fmtTime(d)} ·{' '}
              {plural(m.speakers.length, ['участник', 'участника', 'участников'])}
              {processingSecs > 0 && !busy && <> · обработано за {fmtTime(processingSecs)}</>}
            </p>
          </div>
          <div className="head-actions">
            <Avatars speakers={m.speakers} />
            <div className="dropdown">
              <button className="btn" onClick={() => setMenuOpen((o) => !o)} aria-expanded={menuOpen}>
                <Download size={16} /> Экспорт и действия
              </button>
              {menuOpen && (
                <div className="menu" onMouseLeave={() => setMenuOpen(false)}>
                  <a href={api.exportUrl(m.id, 'docx')} download>DOCX · протокол для Word</a>
                  <a href={api.exportUrl(m.id, 'md')} download>Markdown · протокол</a>
                  <a href={api.exportUrl(m.id, 'srt')} download>SRT · субтитры</a>
                  <a href={api.exportUrl(m.id, 'vtt')} download>VTT · субтитры для веба</a>
                  <a href={api.exportUrl(m.id, 'json')} download>JSON · все данные</a>
                  <hr />
                  {m.has_audio && !busy && (
                    <button onClick={() => reprocess('asr')}>
                      <RotateCcw size={15} /> Распознать заново
                    </button>
                  )}
                  {!busy && (
                    <button onClick={() => reprocess('analyze')}>
                      <Sparkles size={15} /> Пересобрать выжимку и задачи
                    </button>
                  )}
                  <button className="danger" onClick={remove}>
                    <Trash2 size={15} /> Удалить встречу
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
        {busy && (
          <p className="notice">
            {(progress ?? m).stage >= 3
              ? 'Идёт анализ: выжимка, решения и задачи появятся через минуту-другую. Стенограмма уже готова.'
              : `Идёт обработка: ${STAGES[(progress ?? m).stage]?.name.toLowerCase() ?? 'в очереди'} — стенограмма обновится сама.`}
          </p>
        )}
      </header>

      {m.has_audio ? (
        <section className="player" aria-label="Плеер">
          <audio
            ref={audio}
            src={api.audioUrl(m.id)}
            preload="metadata"
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            onTimeUpdate={(e) => !playing && setTime(e.currentTarget.currentTime)}
          />
          <div className="controls">
            <button className="icon-btn" onClick={() => seek(time - 10)} aria-label="Назад на 10 секунд">
              <RotateCcw size={18} />
            </button>
            <button className="play" onClick={toggle} aria-label={playing ? 'Пауза' : 'Воспроизвести'}>
              {playing ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
            </button>
            <button className="icon-btn" onClick={() => seek(time + 10)} aria-label="Вперёд на 10 секунд">
              <RotateCw size={18} />
            </button>
            <span className="mono time">
              {fmtTime(time)} <span className="muted">/ {fmtTime(d)}</span>
            </span>
          </div>
          <div
            className="wave"
            role="slider"
            aria-label="Позиция"
            aria-valuemin={0}
            aria-valuemax={Math.round(d)}
            aria-valuenow={Math.round(time)}
            aria-valuetext={fmtTime(time)}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'ArrowLeft') seek(time - 5)
              if (e.key === 'ArrowRight') seek(time + 5)
            }}
            onClick={(e) => {
              const r = e.currentTarget.getBoundingClientRect()
              seek(((e.clientX - r.left) / r.width) * d)
            }}
          >
            {bars.map((b, i) => (
              <i
                key={i}
                className={`${b.color >= 0 ? `sp-${b.color}` : 'silence'} ${(i + 0.5) / bars.length <= time / d ? 'played' : ''}`}
                style={{ height: `${b.h * 100}%` }}
              />
            ))}
            <span className="cursor" style={{ left: `${(time / d) * 100}%` }} />
          </div>
          <button className="rate" onClick={() => setRate(RATES[(RATES.indexOf(rate) + 1) % RATES.length])} title="Скорость">
            {rate}×
          </button>
        </section>
      ) : (
        <p className="notice">Аудио удалено после обработки (настройка приватности) — осталась только стенограмма.</p>
      )}

      <div className="meeting-body">
        <section className="transcript" aria-label="Стенограмма">
          <div className="transcript-bar">
            <div className="seg-control" role="group" aria-label="Вид текста">
              <button className={mode === 'clean' ? 'on' : ''} onClick={() => setMode('clean')} title="Без слов-паразитов">
                Чистый текст
              </button>
              <button className={mode === 'verbatim' ? 'on' : ''} onClick={() => setMode('verbatim')} title="Как распознано">
                Дословно
              </button>
            </div>
            <label className="mini-search">
              <Search size={15} />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && nextMatch()}
                placeholder="Найти в стенограмме"
              />
              {q && <span className="muted small">{matches.length}</span>}
            </label>
          </div>

          <div className="segments" onWheel={() => (follow.current = false)} onTouchMove={() => (follow.current = false)}>
            {m.segments.map((s, i) => {
              const sp = speakerOf(m, s.speaker)
              const sameSpeaker = m.segments[i - 1]?.speaker === s.speaker
              return (
                <div
                  key={s.id}
                  ref={(el) => {
                    segRefs.current[i] = el
                  }}
                  className={`segment ${i === active ? 'active' : ''} ${sameSpeaker ? 'cont' : ''} ${query && !matches.includes(i) ? 'dim' : ''}`}
                >
                  <div className="seg-meta">
                    {!sameSpeaker && <span className={`speaker-name sp-text-${sp?.color ?? 0}`}>{sp?.name}</span>}
                    <button className="ts mono" onClick={() => seek(s.start)}>
                      {fmtTime(s.start)}
                    </button>
                    {s.edited && <span className="muted small">изменено</span>}
                  </div>
                  {editing === i ? (
                    <EditBox
                      value={s.clean}
                      cancel={() => setEditing(null)}
                      save={async (v) => {
                        const r = await api.editSegment(s.id, v)
                        patch((mm) => ({ ...mm, segments: mm.segments.map((x) => (x.id === s.id ? { ...x, clean: r.clean, edited: true } : x)) }))
                        setEditing(null)
                        setMode('clean')
                      }}
                    />
                  ) : (
                    <p onClick={() => seek(s.start)}>
                      <Highlight text={text(s)} q={q} />
                      <button
                        className="edit-btn"
                        onClick={(e) => {
                          e.stopPropagation()
                          setEditing(i)
                        }}
                        aria-label="Исправить текст"
                      >
                        <Pencil size={13} />
                      </button>
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        </section>

        <aside className="side">
          <div className="tabs" role="tablist">
            {(
              [
                ['summary', 'Кратко', Sparkles],
                ['decisions', 'Решения', Gavel],
                ['tasks', 'Задачи', ListChecks],
                ['speakers', 'Спикеры', Users],
              ] as const
            ).map(([tid, label, Icon]) => (
              <button key={tid} role="tab" aria-selected={tab === tid} className={tab === tid ? 'on' : ''} onClick={() => setTab(tid)}>
                <Icon size={15} /> {label}
              </button>
            ))}
          </div>

          <div className="tab-body">
            {tab === 'summary' && (
              <>
                {m.summary.length ? (
                  <ul className="summary">
                    {m.summary.map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted empty-tab">{busy ? 'Выжимка готовится…' : 'Выжимки нет: анализ выключен в настройках.'}</p>
                )}
                {m.topics.length > 0 && (
                  <>
                    <h3 className="section-title">Темы</h3>
                    <ol className="topics">
                      {m.topics.map((t, i) => {
                        const start = segAt(t.seg).start
                        const end = m.topics[i + 1] ? segAt(m.topics[i + 1].seg).start : d
                        return (
                          <li key={t.title + i} className={time >= start && time < end ? 'on' : ''}>
                            <button onClick={() => seek(start)}>
                              <span className="mono">{fmtTime(start)}</span> {t.title}
                            </button>
                          </li>
                        )
                      })}
                    </ol>
                  </>
                )}
              </>
            )}

            {tab === 'decisions' && (
              <ul className="items">
                {m.decisions.length === 0 && <li className="muted">{busy ? 'Анализ ещё идёт…' : 'Решений не зафиксировано.'}</li>}
                {m.decisions.map((x, i) => (
                  <li key={i}>
                    <Gavel size={16} className="accent" />
                    <div>
                      <p>{x.text}</p>
                      <TimeChip seg={x.seg} />
                    </div>
                  </li>
                ))}
              </ul>
            )}

            {tab === 'tasks' && (
              <>
                <ul className="items">
                  {m.tasks.length === 0 && <li className="muted">{busy ? 'Анализ ещё идёт…' : 'Задач не прозвучало.'}</li>}
                  {m.tasks.map((t) => {
                    const sp = owner(t)
                    return (
                      <li key={t.id} className={t.done ? 'done' : ''}>
                        <label className="check">
                          <input
                            type="checkbox"
                            checked={t.done}
                            onChange={async () => {
                              patch((mm) => ({ ...mm, tasks: mm.tasks.map((x) => (x.id === t.id ? { ...x, done: !t.done } : x)) }))
                              await api.setTask(t.id, !t.done)
                              onChange()
                            }}
                          />
                          <span>
                            <Check size={12} strokeWidth={3.5} />
                          </span>
                        </label>
                        <div>
                          <p>{t.text}</p>
                          <div className="task-meta">
                            {sp ? (
                              <span className={`speaker-name sp-text-${sp.color}`}>{sp.name}</span>
                            ) : (
                              <span className="speaker-name muted">{t.owner_name ?? 'Без исполнителя'}</span>
                            )}
                            {t.due && <span className="muted small">до {fmtDate(t.due)}</span>}
                            <TimeChip seg={t.seg} />
                          </div>
                        </div>
                      </li>
                    )
                  })}
                </ul>
                {m.tasks.length > 0 && (
                  <button className="btn wide" onClick={copyTasks}>
                    {copied ? <Check size={16} /> : <Copy size={16} />} {copied ? 'Скопировано' : 'Скопировать для мессенджера'}
                  </button>
                )}
              </>
            )}

            {tab === 'speakers' && (
              <ul className="speakers">
                {talk.map(({ s, secs, count }) => (
                  <li key={s.id}>
                    <span className={`avatar sp-${s.color}`}>{initials(s.name)}</span>
                    <div className="grow">
                      <SpeakerName
                        name={s.name}
                        save={async (name) => {
                          patch((mm) => ({ ...mm, speakers: mm.speakers.map((x) => (x.id === s.id ? { ...x, name } : x)) }))
                          await api.renameSpeaker(s.id, name)
                          onChange()
                        }}
                      />
                      <div className="bar thin">
                        <i className={`bg-sp-${s.color}`} style={{ width: `${(secs / talkTotal) * 100}%` }} />
                      </div>
                      <span className="muted small">
                        {Math.round((secs / talkTotal) * 100)}% времени · {plural(count, ['реплика', 'реплики', 'реплик'])}
                      </span>
                    </div>
                  </li>
                ))}
                <li className="muted small hint">Нажмите на имя, чтобы переименовать. Имя меняется во всей встрече, в задачах, поиске и экспорте.</li>
              </ul>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}

function SpeakerName({ name, save }: { name: string; save: (v: string) => void }) {
  const [v, setV] = useState(name)
  useEffect(() => setV(name), [name])
  const commit = () => {
    const t = v.trim()
    if (t && t !== name) save(t)
    else setV(name)
  }
  return (
    <input
      className="name-input"
      value={v}
      maxLength={100}
      aria-label="Имя спикера"
      onChange={(e) => setV(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') e.currentTarget.blur()
        if (e.key === 'Escape') {
          setV(name)
          e.currentTarget.blur()
        }
      }}
    />
  )
}

function EditBox({ value, save, cancel }: { value: string; save: (v: string) => Promise<void>; cancel: () => void }) {
  const [v, setV] = useState(value)
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    if (!v.trim()) return
    setBusy(true)
    try {
      await save(v)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="edit-box">
      <textarea
        autoFocus
        value={v}
        onChange={(e) => setV(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') cancel()
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit()
        }}
        rows={3}
      />
      <div className="edit-actions">
        <button className="btn small" onClick={cancel}>
          <X size={14} /> Отмена
        </button>
        <button className="btn primary small" onClick={submit} disabled={busy}>
          <Check size={14} /> Сохранить
        </button>
      </div>
    </div>
  )
}
