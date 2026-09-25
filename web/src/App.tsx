import { useCallback, useEffect, useState } from 'react'
import { AudioLines, MessageSquare, Monitor, Moon, Settings as Cog, ShieldCheck, Sun } from 'lucide-react'
import { api, type MeetingBrief } from './api'
import { isBusy, load, save } from './util'
import Meetings from './pages/Meetings'
import MeetingView from './pages/MeetingView'
import Ask from './pages/Ask'
import Settings from './pages/Settings'

type Theme = 'system' | 'light' | 'dark'

function useHash() {
  const [hash, setHash] = useState(location.hash.slice(1) || '/')
  useEffect(() => {
    const on = () => {
      setHash(location.hash.slice(1) || '/')
      window.scrollTo(0, 0)
    }
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  return hash
}

export default function App() {
  const [meetings, setMeetings] = useState<MeetingBrief[] | null>(null)
  const [offline, setOffline] = useState(false)
  const [theme, setTheme] = useState<Theme>(() => load('theme', 'system'))
  const hash = useHash()

  useEffect(() => {
    if (theme === 'system') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.dataset.theme = theme
    save('theme', theme)
  }, [theme])

  const refresh = useCallback(async () => {
    try {
      setMeetings(await api.meetings())
      setOffline(false)
    } catch {
      setOffline(true)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh, hash])

  // While something is being processed, keep the list fresh (cards show stage and percent).
  const busy = meetings?.some((m) => isBusy(m) || m.status === 'uploading') ?? false
  useEffect(() => {
    if (!busy && !offline) return
    const id = setInterval(refresh, busy ? 1500 : 5000)
    return () => clearInterval(id)
  }, [busy, offline, refresh])

  const [path, query] = hash.split('?')
  const params = new URLSearchParams(query)
  let page
  let section = 'meetings'
  if (path.startsWith('/m/')) {
    const id = path.slice(3)
    page = <MeetingView key={id + (params.get('t') ?? '')} id={id} startAt={Number(params.get('t') ?? 0)} onChange={refresh} />
  } else if (path === '/ask') {
    section = 'ask'
    page = <Ask meetings={meetings ?? []} />
  } else if (path === '/settings') {
    section = 'settings'
    page = <Settings />
  } else {
    page = <Meetings meetings={meetings} refresh={refresh} />
  }

  const nextTheme: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' }
  const ThemeIcon = { system: Monitor, light: Sun, dark: Moon }[theme]
  const themeLabel = { system: 'Как в системе', light: 'Светлая тема', dark: 'Тёмная тема' }[theme]

  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="#/">
          <span className="brand-mark">
            <AudioLines size={18} strokeWidth={2.4} />
          </span>
          Протоколист
        </a>
        <nav className="nav">
          <a href="#/" className={section === 'meetings' ? 'active' : ''}>
            <AudioLines size={18} /> <span>Встречи</span>
          </a>
          <a href="#/ask" className={section === 'ask' ? 'active' : ''}>
            <MessageSquare size={18} /> <span>Спросить</span>
          </a>
          <a href="#/settings" className={section === 'settings' ? 'active' : ''}>
            <Cog size={18} /> <span>Настройки</span>
          </a>
        </nav>
        <div className="sidebar-foot">
          <div className="privacy">
            <ShieldCheck size={18} />
            <div>
              <b>Локальный режим</b>
              <span>Записи и текст не покидают устройство</span>
            </div>
          </div>
          <button className="theme-btn" onClick={() => setTheme(nextTheme[theme])} title="Сменить тему">
            <ThemeIcon size={16} /> {themeLabel}
          </button>
        </div>
      </aside>
      <main className="main">
        {offline && <div className="offline">Сервер недоступен. Проверьте, что запущены API и база данных — переподключаемся…</div>}
        {page}
      </main>
    </div>
  )
}
