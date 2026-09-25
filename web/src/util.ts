import type { Meeting, MeetingBrief } from './api'

export const STAGES = [
  { name: 'Подготовка', hint: 'ffmpeg · 16 кГц моно · нормализация громкости' },
  { name: 'Распознавание речи', hint: 'Whisper large-v3-turbo · пословные таймкоды' },
  { name: 'Диаризация', hint: 'кто когда говорит · голосовые эмбеддинги' },
  { name: 'Анализ', hint: 'локальная LLM · выжимка, решения, задачи' },
  { name: 'Индексация', hint: 'multilingual-e5 · pgvector' },
]

/** Overall pipeline progress 0..1 (stages weighted by typical duration). */
const WEIGHTS = [0.05, 0.45, 0.15, 0.3, 0.05]
export function overall(m: Pick<MeetingBrief, 'status' | 'stage' | 'stage_progress'>) {
  if (m.status === 'done') return 1
  return WEIGHTS.slice(0, m.stage).reduce((a, w) => a + w, 0) + (WEIGHTS[m.stage] ?? 0) * m.stage_progress
}

export const isBusy = (m: Pick<MeetingBrief, 'status'>) => m.status === 'queued' || m.status === 'processing'

export function fmtTime(s: number) {
  s = Math.max(0, Math.floor(s))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`
}

export function fmtDuration(s: number) {
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return h ? `${h} ч ${m} мин` : `${m} мин`
}

export const fmtDate = (iso: string, opts: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'long' }) =>
  new Date(iso).toLocaleDateString('ru-RU', opts)

export const fmtSize = (bytes: number) =>
  bytes > 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} ГБ` : `${(bytes / 1024 ** 2).toFixed(1)} МБ`

export function plural(n: number, forms: [string, string, string]) {
  const n10 = n % 10
  const n100 = n % 100
  const f = n10 === 1 && n100 !== 11 ? 0 : n10 >= 2 && n10 <= 4 && (n100 < 10 || n100 >= 20) ? 1 : 2
  return `${n} ${forms[f]}`
}

export const initials = (name: string) => (/^Спикер \d+$/.test(name) ? name.replace('Спикер ', 'С') : name.slice(0, 1))

export const speakerOf = (m: Meeting, id: number | null) => m.speakers.find((s) => s.id === id)

export function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v ? (JSON.parse(v) as T) : fallback
  } catch {
    return fallback
  }
}

export function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* storage unavailable: preference lives for this session only */
  }
}
