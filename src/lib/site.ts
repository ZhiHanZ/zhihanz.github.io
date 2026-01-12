import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import YAML from 'yaml'

export type SiteConfig = {
  baseURL: string
  title: string
  description: string
  author: string
  pagination: number
  social: {
    github?: string
    email?: string
    twitter?: string
  }
  menu: {
    main: Array<{ name: string; url: string; weight: number }>
  }
  giscus: {
    repo: string
    repoId: string
    category: string
    categoryId: string
  }
}

const REPO_ROOT = process.cwd()

let cachedSite: SiteConfig | null = null

export function loadSite(): SiteConfig {
  if (cachedSite) return cachedSite

  const raw = readFileSync(join(REPO_ROOT, 'site.yaml'), 'utf8')
  cachedSite = YAML.parse(raw) as SiteConfig
  return cachedSite
}

export function formatDate(date: Date): { day: string; monthYear: string; full: string } {
  const day = String(date.getUTCDate()).padStart(2, '0')
  const monthYear = new Intl.DateTimeFormat('en-US', {
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC'
  }).format(date)
  const full = new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    timeZone: 'UTC'
  }).format(date)

  return { day, monthYear, full }
}

export function paginate<T>(items: T[], pageSize: number, page: number): { items: T[]; totalPages: number } {
  const totalPages = Math.ceil(items.length / pageSize)
  const start = (page - 1) * pageSize
  const end = start + pageSize

  return {
    items: items.slice(start, end),
    totalPages
  }
}
