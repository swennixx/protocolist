import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Check, Clock, FileAudio, LoaderCircle, RotateCcw, Search, Trash2, Upload, X } from 'lucide-react'
import { api, upload, watch, type Hit, type MeetingBrief, type Progress, type Speaker } from '../api'
import { STAGES, fmtDate, fmtDuration, fmtSize, fmtTime, initials, isBusy, overall, plural } from '../util'

export function Highlight({ text, q }: { text: string; q: string }) {
  const query = q.trim()
  if (!query) return <>{text}</>
  const re = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  return <>{text.split(re).map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : p))}</>
}

export function Avatars({ speakers }: { speakers: Speaker[] }) {
  return (
    <div className="avatars">
      {speakers.map((s) => (
        <span key={s.id} className={`avatar sp-${s.color}`} title={s.name}>
          {initials(s.name)}
        </span>
      ))}
    </div>
  )
}

export default function Meetings({ meetings, refresh }: { meetings: MeetingBrief[] | null; refresh: () => void }) {
  const [q, setQ] = useState('')
  const [mode, setMode] = useState<'exact' | 'smart'>('smart')
  const [hits, setHits] = useState<Hit[] | null>(null)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [dropFile, setDropFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    setHits(null)
    if (!q.trim()) return
    let stale = false
    const id = setTimeout(() => {
      api.search(q, mode).then((h) => !stale && setHits(h), () => !stale && setHits([]))
    }, 300)
    return () => {
      stale = true
      clearTimeout(id)
    }
  }, [q, mode])

  const list = meetings ?? []
  const done = list.filter((m) => m.status === 'done')
  const total = done.reduce((a, m) => a + m.duration, 0)
  const openTasks = done.reduce((a, m) => a + m.open_tasks, 0)
  const shown = q ? list.filter((m) => m.title.toLowerCase().includes(q.trim().toLowerCase())) : list

  return (
    <div
      className="page"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(e) => e.currentTarget === e.target && setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const f = e.dataTransfer.files[0]
        if (f) {
          setDropFile(f)
          setUploadOpen(true)
        }
      }}
    >
      <header className="page-head">
        <div>
          <h1>Встречи</h1>
          <p className="muted">
            {meetings === null
              ? 'Загружаем…'
              : `${plural(done.length, ['встреча', 'встречи', 'встреч'])} · ${fmtDuration(total)} записей · ${plural(openTasks, ['открытая задача', 'открытые задачи', 'открытых задач'])}`}
          </p>
        </div>
        <button className="btn primary" onClick={() => setUploadOpen(true)}>
          <Upload size={17} /> Загрузить запись
        </button>
      </header>

      <div className="searchbar">
        <Search size={18} className="muted" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Поиск по всем встречам: «сроки релиза», «бюджет на рекламу»…"
          aria-label="Поиск по встречам"
        />
        {q && (
          <button className="icon-btn" onClick={() => setQ('')} aria-label="Очистить">
            <X size={16} />
          </button>
        )}
        <div className="seg-control" role="group" aria-label="Режим поиска">
          <button className={mode === 'smart' ? 'on' : ''} onClick={() => setMode('smart')} title="Слова и смысл: находит перефразировки">
            По смыслу
          </button>
          <button className={mode === 'exact' ? 'on' : ''} onClick={() => setMode('exact')} title="Только совпадения слов (с учётом падежей)">
            По словам
          </button>
        </div>
      </div>

      {q && (
        <section className="hits">
          <h2 className="section-title">Фрагменты{hits ? ` · ${hits.length}` : ' · ищем…'}</h2>
          {hits?.length === 0 && <p className="muted">Ничего не нашлось. Попробуйте переформулировать или другой режим поиска.</p>}
          {hits?.map((h) => (
            <a key={h.meeting_id + h.seg} className="hit" href={`#/m/${h.meeting_id}?t=${Math.floor(h.start)}`}>
              <div className="hit-meta">
                <span className="hit-title">{h.meeting_title}</span>
                <span className="chip mono">{fmtTime(h.start)}</span>
                <span className={`speaker-name sp-text-${h.color}`}>{h.speaker}</span>
              </div>
              <p>
                <Highlight text={h.text} q={q} />
              </p>
            </a>
          ))}
        </section>
      )}

      {meetings?.length === 0 && (
        <button className="empty-upload" onClick={() => setUploadOpen(true)}>
          <FileAudio size={34} />
          <b>Загрузите первую запись</b>
          <span className="muted">Перетащите файл встречи сюда или нажмите, чтобы выбрать. Всё обрабатывается на этом компьютере.</span>
        </button>
      )}

      <div className="cards">
        {shown.map((m) => (
          <MeetingCard key={m.id} m={m} refresh={refresh} />
        ))}
      </div>

      {dragging && (
        <div className="drop-overlay">
          <FileAudio size={40} />
          <b>Отпустите, чтобы загрузить запись</b>
        </div>
      )}

      {uploadOpen && (
        <UploadDialog
          initialFile={dropFile}
          refresh={refresh}
          close={() => {
            setUploadOpen(false)
            setDropFile(null)
            refresh()
          }}
        />
      )}
    </div>
  )
}

function MeetingCard({ m, refresh }: { m: MeetingBrief; refresh: () => void }) {
  const date = fmtDate(m.date, { day: 'numeric', month: 'long', weekday: 'short' })
  const remove = async () => {
    if (confirm(`Удалить «${m.title}»?`)) {
      await api.remove(m.id)
      refresh()
    }
  }
  if (m.status === 'uploading') {
    return (
      <div className="card processing">
        <div className="card-top">
          <span className="date">{date}</span>
          <span className="status">
            <Upload size={13} /> Загрузка файла
          </span>
        </div>
        <h3>{m.title}</h3>
        <div className="stage-line">
          <span>
            {fmtSize(m.received)} из {fmtSize(m.size)}
          </span>
          <button className="icon-btn" onClick={remove} aria-label="Удалить" title="Если окно с загрузкой закрыли — удалите и загрузите файл заново">
            <Trash2 size={16} />
          </button>
        </div>
        <div className="bar">
          <i style={{ width: `${(m.received / m.size) * 100}%` }} />
        </div>
      </div>
    )
  }
  if (m.status === 'error') {
    const retry = async () => {
      await api.retry(m.id)
      refresh()
    }
    return (
      <div className="card problem">
        <div className="card-top">
          <span className="date">{date}</span>
          <span className="status bad">
            <AlertTriangle size={13} /> Ошибка
          </span>
        </div>
        <h3>{m.title}</h3>
        <p className="preview">{m.error}</p>
        <div className="card-foot">
          <button className="btn small" onClick={retry}>
            <RotateCcw size={14} /> Повторить
          </button>
          <button className="icon-btn" onClick={remove} aria-label="Удалить">
            <Trash2 size={16} />
          </button>
        </div>
      </div>
    )
  }
  if (isBusy(m)) {
    const stage = STAGES[Math.min(m.stage, STAGES.length - 1)]
    return (
      <a className="card processing" href={`#/m/${m.id}`}>
        <div className="card-top">
          <span className="date">{date}</span>
          <span className="status">
            <LoaderCircle size={14} className="spin" /> {m.status === 'queued' ? 'В очереди' : 'Обработка'}
          </span>
        </div>
        <h3>{m.title}</h3>
        <div className="stage-line">
          <span>{m.status === 'queued' ? 'Ждёт свободного обработчика' : stage.name}</span>
          <span className="mono">{Math.round(overall(m) * 100)}%</span>
        </div>
        <div className="bar">
          <i style={{ width: `${overall(m) * 100}%` }} />
        </div>
      </a>
    )
  }
  return (
    <a className="card" href={`#/m/${m.id}`}>
      <div className="card-top">
        <span className="date">{date}</span>
        <span className="date mono">
          <Clock size={13} /> {fmtTime(m.duration)}
        </span>
      </div>
      <h3>{m.title}</h3>
      <p className="preview">{m.summary[0] ?? 'Стенограмма готова.'}</p>
      <div className="card-foot">
        <Avatars speakers={m.speakers} />
        <span className="muted small">
          {plural(m.decisions, ['решение', 'решения', 'решений'])} ·{' '}
          {m.open_tasks ? plural(m.open_tasks, ['задача', 'задачи', 'задач']) : 'задачи закрыты'}
        </span>
      </div>
    </a>
  )
}

export function Pipeline({ p }: { p: Progress }) {
  return (
    <ol className="pipeline">
      {STAGES.map((s, i) => {
        const state = p.status === 'done' || i < p.stage ? 'done' : i === p.stage && p.status === 'processing' ? 'active' : 'wait'
        return (
          <li key={s.name} className={state}>
            <span className="pipe-icon">
              {state === 'done' ? <Check size={14} strokeWidth={3} /> : state === 'active' ? <LoaderCircle size={16} className="spin" /> : i + 1}
            </span>
            <div className="pipe-body">
              <div className="pipe-row">
                <b>{s.name}</b>
                {state === 'active' && <span className="mono small">{Math.round(p.stage_progress * 100)}%</span>}
              </div>
              <span className="muted small">{s.hint}</span>
              {state === 'active' && (
                <div className="bar thin">
                  <i style={{ width: `${p.stage_progress * 100}%` }} />
                </div>
              )}
            </div>
          </li>
        )
      })}
    </ol>
  )
}

function UploadDialog({ initialFile, refresh, close }: { initialFile: File | null; refresh: () => void; close: () => void }) {
  const [file, setFile] = useState<File | null>(initialFile)
  const [title, setTitle] = useState(initialFile ? initialFile.name.replace(/\.[^.]+$/, '') : '')
  const [lang, setLang] = useState('auto')
  const [speakers, setSpeakers] = useState('auto')
  const [id, setId] = useState<string | null>(null)
  const [sent, setSent] = useState(0)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [started, setStarted] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close()
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [close])

  useEffect(() => () => abort.current?.abort(), [])

  useEffect(() => {
    if (!id || !file || sent < file.size) return
    return watch(id, (p) => {
      setProgress(p)
      if (p.status === 'done' || p.status === 'error') refresh()
    })
  }, [id, file, sent, refresh])

  const pick = (f: File | undefined) => {
    if (!f) return
    setFile(f)
    setError(null)
    if (!title) setTitle(f.name.replace(/\.[^.]+$/, ''))
  }

  const start = async () => {
    if (!file) return
    setError(null)
    try {
      const m = await api.create({
        title: title.trim() || file.name,
        filename: file.name,
        size: file.size,
        language: lang === 'auto' ? null : lang,
        num_speakers: speakers === 'auto' ? null : Number(speakers),
      })
      setId(m.id)
      setStarted(Date.now())
      refresh()
      abort.current = new AbortController()
      await upload(m.id, file, setSent, abort.current.signal)
    } catch (e) {
      if (!abort.current?.signal.aborted) setError((e as Error).message)
    }
  }

  const uploading = id && file && sent < file.size
  const done = progress?.status === 'done'
  const elapsed = (Date.now() - started) / 1000
  const frac = progress ? overall(progress) : 0
  const eta = progress?.status === 'processing' && frac > 0.08 ? Math.max(0, elapsed / frac - elapsed) : null

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="upload-title">
        <div className="modal-head">
          <h2 id="upload-title">{id ? title.trim() || file?.name : 'Новая запись'}</h2>
          <button className="icon-btn" onClick={close} aria-label="Закрыть">
            <X size={18} />
          </button>
        </div>

        {!id ? (
          <>
            <button
              className={`dropzone ${file ? 'has-file' : ''}`}
              onClick={() => input.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                e.stopPropagation()
                pick(e.dataTransfer.files[0])
              }}
            >
              <FileAudio size={28} />
              {file ? (
                <>
                  <b>{file.name}</b>
                  <span className="muted small">{fmtSize(file.size)} · нажмите, чтобы заменить</span>
                </>
              ) : (
                <>
                  <b>Перетащите файл или нажмите для выбора</b>
                  <span className="muted small">MP3, M4A, WAV, OGG, OPUS, WEBM, MP4, MOV · до 2 ГБ</span>
                </>
              )}
            </button>
            <input ref={input} type="file" accept="audio/*,video/*" hidden onChange={(e) => pick(e.target.files?.[0])} />

            <label className="field">
              <span>Название</span>
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Например, планёрка отдела продаж" maxLength={300} />
            </label>
            <div className="field-row">
              <label className="field">
                <span>Язык</span>
                <select value={lang} onChange={(e) => setLang(e.target.value)}>
                  <option value="auto">Как в настройках</option>
                  <option value="ru">Русский</option>
                  <option value="en">Английский</option>
                </select>
              </label>
              <label className="field">
                <span>Спикеров</span>
                <select value={speakers} onChange={(e) => setSpeakers(e.target.value)}>
                  <option value="auto">Определить автоматически</option>
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {error && <p className="form-error">{error}</p>}
            <div className="modal-foot">
              <span className="muted small">Файл обрабатывается локально</span>
              <button className="btn primary" disabled={!file} onClick={start}>
                Начать обработку
              </button>
            </div>
          </>
        ) : (
          <>
            {uploading || !progress ? (
              <div className="upload-progress">
                <div className="pipe-row">
                  <b>Загрузка файла</b>
                  <span className="mono small">
                    {fmtSize(sent)} / {fmtSize(file!.size)}
                  </span>
                </div>
                <div className="bar">
                  <i style={{ width: `${(sent / file!.size) * 100}%` }} />
                </div>
                <span className="muted small">Если связь прервётся, загрузка продолжится с того же места</span>
              </div>
            ) : (
              <Pipeline p={progress} />
            )}
            {(error || progress?.error) && <p className="form-error">{error ?? progress?.error}</p>}
            <div className="modal-foot">
              <span className="muted small">
                {done ? 'Готово' : progress?.status === 'queued' ? 'В очереди' : eta !== null ? `Осталось около ${fmtTime(eta)}` : uploading ? 'Загружаем…' : 'Оцениваем время…'}
              </span>
              {done ? (
                <a className="btn primary" href={`#/m/${id}`}>
                  Открыть встречу
                </a>
              ) : (
                <button className="btn" onClick={close} disabled={!!uploading}>
                  {uploading ? 'Дождитесь загрузки' : 'Свернуть'}
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
