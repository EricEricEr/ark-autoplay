// 零依赖 HTTP JSON 封装：统一 15s 超时与「说人话」的网络错误。
// fetchImpl 可注入（selftest 用假 fetch 替换网络层，不 mock 任何内部函数）。

export const DEFAULT_TIMEOUT_MS = 15000

export class HttpError extends Error {
  constructor(message, { status = null, url = '', bodySnippet = '' } = {}) {
    super(message)
    this.name = 'HttpError'
    this.status = status
    this.url = url
    this.bodySnippet = bodySnippet
  }
}

export function explainNetError(err) {
  const name = err?.name || ''
  const code = err?.cause?.code || err?.code || ''
  if (name === 'TimeoutError' || name === 'AbortError')
    return '请求超时（15 秒无响应）——网络不通，请检查网络/代理'
  if (code === 'ENOTFOUND' || code === 'EAI_AGAIN')
    return '域名解析失败——网络不通，请检查网络/代理'
  if (code === 'ECONNREFUSED')
    return '连接被拒绝——网络不通，请检查代理/TUN 模式'
  if (code === 'ECONNRESET' || code === 'ETIMEDOUT' || code === 'EPIPE' || code === 'UND_ERR_CONNECT_TIMEOUT')
    return '连接被重置或超时——网络不通，请检查网络/代理'
  if (name === 'TypeError')
    return `网络请求失败（${err?.cause?.message || err.message}）——请检查网络/代理`
  return null
}

export async function httpJson(url, { method = 'GET', headers = {}, body, timeoutMs = DEFAULT_TIMEOUT_MS } = {}, fetchImpl = globalThis.fetch) {
  let res
  try {
    res = await fetchImpl(url, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
      redirect: 'follow',
    })
  }
  catch (err) {
    throw new HttpError(explainNetError(err) || `网络请求失败：${err?.message || err}`, { url })
  }
  const text = await res.text()
  if (!res.ok)
    throw new HttpError(`服务器返回 HTTP ${res.status}`, { status: res.status, url, bodySnippet: text.slice(0, 200) })
  try {
    return JSON.parse(text)
  }
  catch {
    throw new HttpError('响应不是合法 JSON（可能被网关拦截）', { status: res.status, url, bodySnippet: text.slice(0, 200) })
  }
}
