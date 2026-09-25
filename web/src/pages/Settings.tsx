import { useEffect, useRef, useState } from 'react'
import { Check, Lock, X } from 'lucide-react'
import { api, type Settings as S } from '../api'

const MODELS = [
  { id: 'tiny', size: '75 МБ', speed: 5, quality: 1, note: 'Черновик за секунды' },
  { id: 'small', size: '480 МБ', speed: 4, quality: 2, note: 'Слабые машины без GPU' },
  { id: 'medium', size: '1,5 ГБ', speed: 3, quality: 3, note: 'Баланс для CPU' },
  { id: 'large-v3-turbo', size: '1,6 ГБ', speed: 4, quality: 4, note: 'Рекомендуется' },
  { id: 'large-v3', size: '3,1 ГБ', speed: 2, quality: 5, note: 'Максимальная точность' },
]

function Dots({ n, label }: { n: number; label: string }) {
  return (
    <span className="dots" aria-label={`${label}: ${n} из 5`}>
      <span className="muted small">{label}</span>
      {Array.from({ length: 5 }, (_, i) => (
        <i key={i} className={i < n ? 'on' : ''} />
      ))}
    </span>
  )
}

export default function Settings() {
  const [s, setS] = useState<S | null>(null)
  const [term, setTerm] = useState('')
  const [status, setStatus] = useState('Изменения сохраняются автоматически')
  const dirty = useRef(false)
  const set = <K extends keyof S>(k: K, v: S[K]) => {
    dirty.current = true
    setS((x) => (x ? { ...x, [k]: v } : x))
  }

  useEffect(() => {
    api.settings().then(setS, (e) => setStatus(e.message))
  }, [])

  useEffect(() => {
    if (!s || !dirty.current) return
    const id = setTimeout(() => {
      api.saveSettings(s).then(
        () => setStatus('Сохранено'),
        (e) => setStatus(`Не сохранилось: ${e.message}`),
      )
    }, 400)
    return () => clearTimeout(id)
  }, [s])

  const addTerm = () => {
    const t = term.trim()
    if (t && s && !s.glossary.includes(t)) set('glossary', [...s.glossary, t])
    setTerm('')
  }

  if (!s) return <div className="page muted">{status === 'Изменения сохраняются автоматически' ? 'Загружаем настройки…' : status}</div>

  return (
    <div className="page settings">
      <header className="page-head">
        <div>
          <h1>Настройки</h1>
          <p className="muted">{status}</p>
        </div>
      </header>

      <section className="panel">
        <h2>Модель распознавания</h2>
        <p className="muted">
          Whisper работает на этом устройстве. Модели, кроме large-v3-turbo, нужно один раз скачать:{' '}
          <code>python -m app.download medium</code>
        </p>
        <div className="models">
          {MODELS.map((m) => (
            <label key={m.id} className={`model ${s.model === m.id ? 'on' : ''}`}>
              <input type="radio" name="model" checked={s.model === m.id} onChange={() => set('model', m.id)} />
              <div className="model-head">
                <b className="mono">{m.id}</b>
                {s.model === m.id && <Check size={16} className="accent" />}
              </div>
              <span className="muted small">
                {m.note} · {m.size}
              </span>
              <Dots n={m.speed} label="Скорость" />
              <Dots n={m.quality} label="Точность" />
            </label>
          ))}
        </div>
        <label className="field narrow">
          <span>Язык по умолчанию</span>
          <select value={s.lang} onChange={(e) => set('lang', e.target.value)}>
            <option value="auto">Определять автоматически</option>
            <option value="ru">Русский</option>
            <option value="en">Английский</option>
          </select>
        </label>
      </section>

      <section className="panel">
        <h2>Разделение по спикерам</h2>
        <p className="muted">
          {s.pyannote_available
            ? 'pyannote точнее разбирает перебивания и шум; встроенный алгоритм в 5–10 раз быстрее.'
            : 'pyannote недоступна: примите условия модели на Hugging Face и выполните python -m app.download. Пока работает встроенный алгоритм.'}
        </p>
        <div className="seg-control">
          {(
            [
              ['auto', 'Авто'],
              ['pyannote', 'pyannote'],
              ['clustering', 'Встроенная, быстрее'],
            ] as const
          ).map(([id, label]) => (
            <button key={id} className={s.diarization === id ? 'on' : ''} onClick={() => set('diarization', id)} disabled={id === 'pyannote' && !s.pyannote_available}>
              {label}
            </button>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Анализ встречи</h2>
        <div className="field-row">
          <label className="field">
            <span>Языковая модель</span>
            <select value={s.llm} onChange={(e) => set('llm', e.target.value)}>
              <option value="qwen3:8b">Qwen 3 8B · Ollama (рекомендуется)</option>
              <option value="qwen2.5:7b">Qwen 2.5 7B · Ollama</option>
              <option value="off">Выключено — только стенограмма</option>
            </select>
          </label>
          <div className="field">
            <span>Шаблон выжимки</span>
            <div className="seg-control">
              {(
                [
                  ['short', 'Краткий'],
                  ['full', 'Подробный'],
                  ['protocol', 'Протокол'],
                ] as const
              ).map(([id, label]) => (
                <button key={id} className={s.template === id ? 'on' : ''} onClick={() => set('template', id)} disabled={s.llm === 'off'}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Словарь терминов</h2>
        <p className="muted">Названия, фамилии и жаргон компании. Подсказываются модели и повышают точность распознавания.</p>
        <div className="tags">
          {s.glossary.map((t) => (
            <span key={t} className="tag">
              {t}
              <button onClick={() => set('glossary', s.glossary.filter((x) => x !== t))} aria-label={`Удалить «${t}»`}>
                <X size={12} />
              </button>
            </span>
          ))}
          <input
            value={term}
            onChange={(e) => setTerm(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addTerm()}
            onBlur={addTerm}
            placeholder="Добавить термин…"
            aria-label="Новый термин"
          />
        </div>
      </section>

      <section className="panel">
        <h2>Приватность</h2>
        <label className="switch-row">
          <div>
            <b>Удалять аудио после обработки</b>
            <span className="muted small">Останутся только текст, выжимка и задачи</span>
          </div>
          <input type="checkbox" className="switch" checked={s.deleteAudio} onChange={(e) => set('deleteAudio', e.target.checked)} />
        </label>
        <div className="privacy-note">
          <Lock size={16} />
          Обработка не делает сетевых запросов: аудио, текст и векторный индекс хранятся на этом сервере.
        </div>
      </section>
    </div>
  )
}
