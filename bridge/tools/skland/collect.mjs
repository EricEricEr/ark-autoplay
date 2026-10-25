#!/usr/bin/env node
// 既有森空岛扫码通道的采集入口：一次命令完成登录和名册导出，无 npm 依赖。
// 保留全部技能/模组 ID、等级及 player/info 原始数据，不依赖静态名录筛选干员。
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import {
  DEEP_LINK_PREFIX, hgGenScanLogin, hgGrant, hgScanStatus, hgTokenByScanCode,
  interpretScanEnvelope, isAuthFailure, sklandBinding, sklandCredByCode,
  sklandPlayerInfo, sklandRefresh,
} from './lib/skland.mjs'
import { renderTerminalQr } from './lib/qr_terminal.mjs'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const DEFAULT_CRED = path.join(ROOT, '.secrets/skland_cred.json')
// 沿用本项目已验证的客户端指纹常量，不是用户凭证。
const DEVICE_ID = 'BqbjCY5HN8T+MFmjvDT4ceaUO6zSMuvdts2+67sjAL4QTA3GfE7M1rDkuUo8Hbhl03557VponqkJW2Z1wh+/nYA=='
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

async function readJson(filename) {
  return JSON.parse(await readFile(filename, 'utf8'))
}

async function saveJson(filename, doc) {
  await mkdir(path.dirname(filename), { recursive: true })
  const temporary = `${filename}.tmp-${process.pid}`
  await writeFile(temporary, JSON.stringify(doc, null, 2) + '\n', { mode: 0o600 })
  await rename(temporary, filename)
}

export function pickRole(binding, uid) {
  const entry = (binding?.list ?? []).find(e => e.appCode === 'arknights')
  const roles = entry?.bindingList ?? []
  if (uid != null) {
    const hit = roles.find(r => String(r.uid) === String(uid))
    if (!hit) throw new Error('指定 UID 不在本次扫码账号的绑定列表中')
    return hit
  }
  const role = roles.find(r => String(r.uid) === String(entry?.defaultUid))
    ?? roles.find(r => r.isDefault) ?? roles[0]
  if (!role) throw new Error('该扫码账号没有绑定明日方舟角色')
  return role
}

export function buildRoster(info, role = {}, now = new Date()) {
  if (!Array.isArray(info?.chars) || info.chars.length === 0)
    throw new Error('player/info 未返回非空 chars，不覆盖已有名册')
  const unresolved = []
  const ops = info.chars.map(c => {
    const name = info.charInfoMap?.[c.charId]?.name ?? c.name ?? null
    if (!name) unresolved.push({ opId: c.charId, reason: 'name_missing' })
    const main = c.mainSkillLvl ?? null
    const skills = (c.skills ?? []).map(s => ({
      ...s,
      skill_level: s.specializeLevel > 0 ? 7 + s.specializeLevel : main,
    }))
    return {
      opId: c.charId, name, elite: c.evolvePhase ?? null, level: c.level ?? null,
      potential: c.potentialRank == null ? null : c.potentialRank + 1,
      mainSkillLvl: main,
      // 顶层仅表示普通技能等级，绝不把某一个技能专精代替全部技能。
      skill_level: main,
      skills, equip: c.equip ?? [], defaultEquipId: c.defaultEquipId ?? null,
      raw: c,
    }
  })
  return {
    schema: 'skland_roster/1.0', source: 'skland', updated: now.toISOString(),
    player: {
      uid: role.uid ?? info.status?.uid ?? null,
      nickname: info.status?.name ?? role.nickName ?? null,
      server: role.channelName ?? null,
    },
    ops, unresolved, raw_player_info: info,
  }
}

export async function login({ fetchImpl = globalThis.fetch, log = console.log,
  pause = sleep, timeoutMs = 180000, now = Date.now } = {}) {
  const scan = await hgGenScanLogin(fetchImpl, DEVICE_ID)
  if (!scan?.scanId) throw new Error('扫码响应缺少 scanId')
  log('请用手机森空岛 APP 扫码并确认，程序随后自动导出练度：')
  log(renderTerminalQr(scan.scanUrl || `${DEEP_LINK_PREFIX}${scan.scanId}`))
  const deadline = now() + timeoutMs
  let scanCode
  while (now() < deadline) {
    const state = interpretScanEnvelope(await hgScanStatus(fetchImpl, DEVICE_ID, scan.scanId))
    if (state.done) { scanCode = state.scanCode; break }
    await pause(2000)
  }
  if (!scanCode) throw new Error('扫码过期或未确认，请重新运行命令')
  const token = await hgTokenByScanCode(fetchImpl, DEVICE_ID, scanCode)
  const grant = await hgGrant(fetchImpl, DEVICE_ID, token)
  const cred = await sklandCredByCode(fetchImpl, DEVICE_ID, grant.code)
  return {
    cred: cred.cred, token: cred.token, dId: DEVICE_ID,
    savedAt: new Date().toISOString(),
  }
}

export async function runCollect({ outPath, uid = null, credPath = DEFAULT_CRED,
  fixturePath = null, forceLogin = false, fetchImpl = globalThis.fetch,
  loginImpl = login, log = console.log } = {}) {
  if (!outPath) throw new Error('必须指定 --out 输出名册路径')
  let info, role = {}
  if (fixturePath) {
    const fixture = await readJson(fixturePath)
    info = fixture?.code === 0 ? fixture.data : fixture
  }
  else {
    let credential = null
    if (!forceLogin) {
      try { credential = await readJson(credPath) }
      catch (err) { if (err.code !== 'ENOENT') throw new Error('本地凭证不可读；用 --login 重新扫码') }
    }
    const reauthenticate = async () => {
      credential = await loginImpl({ fetchImpl, log })
      await saveJson(credPath, credential)
    }
    if (!credential) await reauthenticate()
    const call = async fn => {
      try { return await fn(credential) }
      catch (err) { if (!isAuthFailure(err)) throw err }
      try {
        const refreshed = await sklandRefresh(fetchImpl, credential)
        credential = { ...credential, token: refreshed.token, savedAt: new Date().toISOString() }
        await saveJson(credPath, credential)
      }
      catch { await reauthenticate() }
      return fn(credential)
    }
    role = pickRole(await call(c => sklandBinding(fetchImpl, c)), uid)
    log(`本次名册：${role.channelName ?? ''} ${role.nickName ?? ''} UID=${role.uid}`)
    info = await call(c => sklandPlayerInfo(fetchImpl, c, role.uid))
  }
  const roster = buildRoster(info, role)
  await saveJson(path.resolve(outPath), roster)
  log(`已导出 ${roster.ops.length} 名干员的练度、全部技能和模组记录 -> ${outPath}`)
  return roster
}

export async function main(args = process.argv.slice(2)) {
  if (args.includes('--help') || args.includes('-h')) {
    console.log('node tools/skland/collect.mjs --out <名册.json> [--uid <游戏UID>] [--login] [--cred <本地凭证.json>] [--fixture <离线样例.json>]')
    console.log('首次显示二维码；确认后自动导出。--login 强制重新扫码。--fixture 离线、不读凭证。')
    return
  }
  const options = {}
  const values = { '--out': 'outPath', '--uid': 'uid', '--cred': 'credPath', '--fixture': 'fixturePath' }
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--login') { options.forceLogin = true; continue }
    const key = values[args[i]]
    if (!key || !args[i + 1] || args[i + 1].startsWith('--'))
      throw new Error(`参数无效或缺值：${args[i]}`)
    options[key] = args[++i]
  }
  await runCollect(options)
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(err => { console.error(`练度导出失败：${err.message}`); process.exitCode = 1 })
}
