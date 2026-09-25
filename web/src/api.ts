// Typed client for the FastAPI backend (see backend/app/main.py).

export type Status = 'uploading' | 'queued' | 'processing' | 'done' | 'error'
export type Speaker = { id: number; name: string; color: number }

export type MeetingBrief = {
  id: string
  title: string
  date: string
  status: Status
  stage: number
  stage_progress: number
  error: string | null
  duration: number
  size: number
  received: number
  speakers: Speaker[]
  summary: string[]
  decisions: number
  open_tasks: number
}

export type Segment = { id: number; speaker: number; start: number; end: number; text: string; clean: string; edited: boolean }
export type Decision = { text: string; seg: number }
export type Task = { id: number; text: string; owner: number | null; owner_name: string | null; due: string | null; seg: number; done: boolean }
export type Topic = { title: string; seg: number }

export type Meeting = Omit<MeetingBrief, 'decisions'> & {
  language: string | null
  has_audio: boolean
  peaks: number[]
  topics: Topic[]
  timings: Record<string, number | string>
  segments: Segment[]
  decisions: Decision[]
  tasks: Task[]
}

export type Hit = {
  meeting_id: string
  meeting_title: string
  date: string
  seg: number
  start: number
  text: string
  speaker: string
  color: number
}

export type Settings = {
  model: string
  lang: string
  llm: string
  template: string
  glossary: string[]
  deleteAudio: boolean
}

export type Progress = Pick<MeetingBrief, 'status' | 'stage' | 'stage_progress' | 'error' | 'received' | 'size'>

export class ApiError extends Error {}

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!r.ok) {
    const detail = await r.json().then((j) => j.detail, () => null)
    throw new ApiError(typeof detail === 'string' ? detail : `Ошибка сервера (${r.status})`)
  }
  return r.status === 204 ? (undefined as T) : r.json()
}

export const api = {
  meetings: () => req<MeetingBrief[]>('GET', '/api/meetings'),
  meeting: (id: string) => req<Meeting>('GET', `/api/meetings/${id}`),
  create: (b: { title: string; filename: string; size: number; language: string | null; num_speakers: number | null }) =>
    req<MeetingBrief>('POST', '/api/meetings', b),
  rename: (id: string, title: string) => req<MeetingBrief>('PATCH', `/api/meetings/${id}`, { title }),
  remove: (id: string) => req<void>('DELETE', `/api/meetings/${id}`),
  retry: (id: string) => req<MeetingBrief>('POST', `/api/meetings/${id}/retry`),
  reprocess: (id: string, stage: 'asr' | 'diarize' | 'analyze') => req<MeetingBrief>('POST', `/api/meetings/${id}/reprocess?stage=${stage}`),
  renameSpeaker: (id: number, name: string) => req<Speaker>('PATCH', `/api/speakers/${id}`, { name }),
  editSegment: (id: number, clean: string) => req<{ clean: string }>('PATCH', `/api/segments/${id}`, { clean }),
  setTask: (id: number, done: boolean) => req<unknown>('PATCH', `/api/tasks/${id}`, { done }),
  search: (q: string, mode: 'smart' | 'exact') => req<Hit[]>('GET', `/api/search?q=${encodeURIComponent(q)}&mode=${mode}`),
  settings: () => req<Settings>('GET', '/api/settings'),
  saveSettings: (s: Settings) => req<Settings>('PUT', '/api/settings', s),
  audioUrl: (id: string) => `/api/meetings/${id}/audio`,
  exportUrl: (id: string, format: string) => `/api/meetings/${id}/export?format=${format}`,
}

const CHUNK = 8 * 1024 * 1024

/** Resumable upload: sends the file in chunks; after a network error it asks the server where to resume. */
export async function upload(id: string, file: File, onProgress: (sent: number) => void, signal?: AbortSignal) {
  let offset = 0
  let failures = 0
  while (offset < file.size) {
    try {
      const r = await fetch(`/api/meetings/${id}/upload?offset=${offset}`, {
        method: 'PUT',
        body: file.slice(offset, offset + CHUNK),
        signal,
      })
      if (r.status === 409) {
        offset = (await api.meeting(id)).received
        continue
      }
      if (!r.ok) throw new ApiError((await r.json().catch(() => ({}))).detail ?? `Ошибка загрузки (${r.status})`)
      offset = (await r.json()).received
      failures = 0
      onProgress(offset)
    } catch (e) {
      if (signal?.aborted || e instanceof ApiError || ++failures > 5) throw e
      await new Promise((res) => setTimeout(res, 1000 * failures))
      offset = (await api.meeting(id)).received
    }
  }
}

/** Processing progress over Server-Sent Events. Returns a function that closes the stream. */
export function watch(id: string, onUpdate: (p: Progress) => void) {
  const es = new EventSource(`/api/meetings/${id}/events`)
  es.onmessage = (e) => {
    const p: Progress = JSON.parse(e.data)
    onUpdate(p)
    if (p.status === 'done' || p.status === 'error') es.close()
  }
  return () => es.close()
}

/** Streams an answer from /api/ask (POST + SSE body). */
export async function ask(
  question: string,
  meetingId: string | null,
  on: { sources: (h: Hit[]) => void; token: (t: string) => void },
  signal?: AbortSignal,
) {
  const r = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question, meeting_id: meetingId }),
    signal,
  })
  if (!r.ok || !r.body) throw new ApiError(`Ошибка сервера (${r.status})`)
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buf += value
    let i
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, i)
      buf = buf.slice(i + 2)
      const event = block.match(/^event: (.*)$/m)?.[1]
      const data = block.match(/^data: (.*)$/m)?.[1]
      if (!data) continue
      if (event === 'sources') on.sources(JSON.parse(data))
      else if (event === 'token') on.token(JSON.parse(data))
    }
  }
}
