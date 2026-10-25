// 森空岛 / 鹰角通行证 API 客户端（零依赖 Node ESM 移植版）。
// 依据就地代码仓库（勿改）：
//   .research/signature.ts                    —— skland-kit 签名实现，本文件 buildSignedHeaders 逐字节对照
//   .research/client_collections_hypergrayph.ts —— 扫码登录与 grant 的端点/信封形态 {status,msg,data}
//   .research/SklandImpl.java                 —— 一图流生产实现：扫码状态语义（100/101/102 轮询、0 带 scanCode）、
//                                                generate_cred_by_code 的 body {kind:1, code}
//   .research/HgImpl.java                     —— 签名另一独立实现（LinkedHashMap 键序 platform,timestamp,dId,vName），旁证
//   skland-kit src/client/core.ts / src/utils/crypto.ts —— signIn/refresh 端点与 md5/hmacSha256 均为小写 hex
import { createHash, createHmac } from 'node:crypto'
import { httpJson, HttpError } from './http.mjs'

export const APP_CODE = '4ca99fa6b56cc2ba' // 森空岛 appCode（skland-kit src/constants.ts SKLAND_APP_CODE）
export const AS_BASE = 'https://as.hypergryph.com'
export const ZONAI_BASE = 'https://zonai.skland.com'

export const SKLAND_APP_UA = 'Mozilla/5.0 (Linux; Android 12; SM-A5560 Build/V417IR; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/101.0.4951.61 Safari/537.36; SKLand/1.52.1'
export const SKLAND_WEB_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
export const DEEP_LINK_PREFIX = 'hypergryph://scan_login?scanId='

// signature.ts: SERVER_TIMESTAMP_OFFSET = 2 * 1000（签名时间向前偏移 2s，规避服务端时间校验过严）
const SERVER_TIMESTAMP_OFFSET = 2000

// crypto.ts: md5(string) = MD5(UTF8(s)) 小写 hex；hmacSha256(key, data) = HMAC-SHA256(UTF8(key), UTF8(data)) 小写 hex
export function md5Hex(s) {
  return createHash('md5').update(String(s), 'utf8').digest('hex')
}

export function hmacSha256Hex(key, data) {
  return createHmac('sha256', String(key)).update(String(data), 'utf8').digest('hex')
}

// 与 .research/signature.ts 的 signRequest 逐字节对应：
//   timestamp = (Date.now() - 2000).toString().slice(0, -3)   —— 秒级字符串
//   signatureHeaders = { platform:'3', timestamp, dId, vName:'1.0.0' }  —— 键序即签名内容，不可调整
//   str = pathname + query + (body ? JSON.stringify(body) : '') + timestamp + JSON.stringify(signatureHeaders)
//   sign = md5(hmacSha256(token, str))
// query 是不带 '?' 的 URLSearchParams 序列化；body 仅 POST JSON 时出现。
export function buildSignedHeaders({ token, cred, dId, pathname, query = '', bodyObj = null, nowMs = Date.now() }) {
  const timestamp = String(nowMs - SERVER_TIMESTAMP_OFFSET).slice(0, -3)
  const signatureHeaders = { platform: '3', timestamp, dId, vName: '1.0.0' }
  const str = `${pathname}${query}${bodyObj ? JSON.stringify(bodyObj) : ''}${timestamp}${JSON.stringify(signatureHeaders)}`
  const sign = md5Hex(hmacSha256Hex(token, str))
  return {
    'user-agent': SKLAND_APP_UA,
    'accept-encoding': 'gzip',
    'connection': 'close',
    'x-requested-with': 'com.hypergryph.skland',
    ...signatureHeaders,
    sign,
    cred,
  }
}

// ---------- 鹰角通行证 as.hypergryph.com（扫码登录 + grant，无需 sign） ----------

function hgHeaders(dId) {
  return {
    'user-agent': SKLAND_APP_UA,
    'content-type': 'application/json',
    'dId': dId,
    'x-requested-with': 'com.hypergryph.skland',
    'origin': 'https://user.hypergryph.com',
    'referer': 'https://user.hypergryph.com/',
  }
}

function hgUnwrap(res, what) {
  if (res && typeof res === 'object' && res.status === 0 && res.data !== undefined)
    return res.data
  const suffix = res?.msg ? `：${res.msg}` : `（status=${res?.status}）`
  const err = new Error(`${what}失败${suffix}`)
  err.hgStatus = res?.status
  throw err
}

// skland-kit parseOAuthToken：token 字段偶为 JSON 字符串（真正 token 在 data.content）
export function parseHgToken(token) {
  const t = String(token ?? '').trim()
  try {
    const parsed = JSON.parse(t)
    if (typeof parsed?.data?.content === 'string')
      return parsed.data.content
  }
  catch { /* 纯 token 字符串 */ }
  return t
}

export async function hgGenScanLogin(fetchImpl, dId) {
  const res = await httpJson(`${AS_BASE}/general/v1/gen_scan/login`, {
    method: 'POST',
    headers: hgHeaders(dId),
    body: { appCode: APP_CODE },
  }, fetchImpl)
  return hgUnwrap(res, '申请扫码二维码') // { scanId, scanUrl? }
}

// 扫码状态信封原样返回：SklandImpl.java 以顶层 status 为进度（100/101/102 继续轮询，0 时 data.scanCode）；
// skland-kit 则认为 data.scanStatus 为进度。interpretScanEnvelope 两种形态都接。
export async function hgScanStatus(fetchImpl, dId, scanId) {
  const q = new URLSearchParams({ scanId }).toString()
  return httpJson(`${AS_BASE}/general/v1/scan_status?${q}`, {
    headers: hgHeaders(dId),
  }, fetchImpl)
}

export function interpretScanEnvelope(res) {
  const data = res?.data ?? {}
  if (data?.scanCode)
    return { done: true, scanCode: data.scanCode }
  const msg = [res?.msg, data?.scanStatus].filter(s => typeof s === 'string' && s && s !== 'OK')[0] || ''
  return { done: false, msg }
}

export async function hgTokenByScanCode(fetchImpl, dId, scanCode) {
  const res = await httpJson(`${AS_BASE}/user/auth/v1/token_by_scan_code`, {
    method: 'POST',
    headers: hgHeaders(dId),
    body: { scanCode },
  }, fetchImpl)
  const data = hgUnwrap(res, '扫码换取鹰角通行证 token')
  if (!data?.token)
    throw new Error('扫码换取鹰角通行证 token 失败：响应缺少 token')
  return parseHgToken(data.token)
}

export async function hgGrant(fetchImpl, dId, hgToken) {
  const res = await httpJson(`${AS_BASE}/user/oauth2/v2/grant`, {
    method: 'POST',
    headers: hgHeaders(dId),
    body: { appCode: APP_CODE, token: hgToken, type: 0 },
  }, fetchImpl)
  const data = hgUnwrap(res, '鹰角通行证授权验证（grant）')
  if (!data?.code)
    throw new Error('鹰角通行证授权验证（grant）失败：响应缺少一次性 code')
  return data // { code, uid }
}

// ---------- 森空岛 zonai.skland.com ----------

// core.ts signIn：本请求不带 sign，headers 与之一致（WEB UA / referer / platform / timestamp / vName）
export async function sklandCredByCode(fetchImpl, dId, code, nowMs = Date.now()) {
  const res = await httpJson(`${ZONAI_BASE}/web/v1/user/auth/generate_cred_by_code`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'user-agent': SKLAND_WEB_UA,
      'referer': 'https://www.skland.com/',
      'origin': 'https://www.skland.com',
      'dId': dId,
      'platform': '3',
      'timestamp': `${Math.floor(nowMs / 1000)}`,
      'vName': '1.0.0',
    },
    body: { code, kind: 1 },
  }, fetchImpl)
  if (res?.code === 0 && res.data?.cred && res.data?.token)
    return res.data // { cred, userId, token }
  throw new Error(`换取森空岛凭证失败：${res?.message || res?.msg || `code=${res?.code}`}`)
}

// core.ts refresh：GET /web/v1/auth/refresh，走完整 signRequest（用旧 token 签名 + cred）
export async function sklandRefresh(fetchImpl, session) {
  const pathname = '/web/v1/auth/refresh'
  const headers = {
    'content-type': 'application/json',
    'referer': 'https://www.skland.com/',
    'origin': 'https://www.skland.com',
    ...buildSignedHeaders({ ...session, pathname }),
    'user-agent': SKLAND_WEB_UA, // core.ts 显式 UA 优先于 signature.ts 默认值
  }
  const res = await httpJson(`${ZONAI_BASE}${pathname}`, { headers }, fetchImpl)
  if (res?.code === 0 && res.data?.token)
    return res.data // { token }
  const err = new Error(`刷新森空岛 token 失败：${res?.message || res?.msg || `code=${res?.code}`}`)
  err.sklandCode = res?.code
  throw err
}

function sklandUnwrap(res, what) {
  if (res?.code === 0)
    return res.data
  const err = new Error(`${what}失败：${res?.message || res?.msg || `code=${res?.code}`}`)
  err.sklandCode = res?.code
  throw err
}

export async function sklandBinding(fetchImpl, session) {
  const pathname = '/api/v1/game/player/binding'
  const headers = buildSignedHeaders({ ...session, pathname })
  const res = await httpJson(`${ZONAI_BASE}${pathname}`, { headers }, fetchImpl)
  return sklandUnwrap(res, '获取游戏绑定角色列表')
}

export async function sklandPlayerInfo(fetchImpl, session, uid) {
  const pathname = '/api/v1/game/player/info'
  const query = new URLSearchParams({ uid }).toString()
  const headers = buildSignedHeaders({ ...session, pathname, query })
  const res = await httpJson(`${ZONAI_BASE}${pathname}?${query}`, { headers }, fetchImpl)
  return sklandUnwrap(res, '获取玩家全干员练度')
}

export function isAuthFailure(err) {
  if (err instanceof HttpError)
    return err.status === 401 || err.status === 403
  return err?.sklandCode != null || err?.hgStatus != null
}
