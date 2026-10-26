// 桌面端扫码桥接：二维码在本机生成，凭据只写私有目录，角色由玩家点选。
import { readFile, writeFile, rename, mkdir } from 'node:fs/promises'
import path from 'node:path'
import { login, runCollect, pickRole } from './collect.mjs'
import qrcode from './lib/vendor/qrcode-generator.js'

const options = {}
const args = process.argv.slice(2)
for (let i = 0; i < args.length; i += 2) options[args[i].replace(/^--/, '')] = args[i + 1]
const pause = ms => new Promise(resolve => setTimeout(resolve, ms))
async function emit(event) {
  await mkdir(path.dirname(options.events), { recursive: true })
  const temporary = `${options.events}.${process.pid}.tmp`
  await writeFile(temporary, JSON.stringify({ ...event, updated_at: new Date().toISOString() }))
  for (;;) {
    try { await rename(temporary, options.events); break }
    catch (error) {
      if (!['EACCES', 'EPERM', 'EBUSY'].includes(error.code)) throw error
      await pause(50)
    }
  }
}
async function onQr(url) {
  const qr = qrcode(0, 'M')
  qr.addData(url, 'Byte')
  qr.make()
  const n = qr.getModuleCount()
  const matrix = Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => qr.isDark(r, c)))
  await emit({ status: 'scan', message: '请用森空岛APP扫码并确认', matrix })
}
async function rolePicker(binding, uid) {
  const account = (binding?.list ?? []).find(item => item.appCode === 'arknights')
  const roles = account?.bindingList ?? []
  if (roles.length < 2) return pickRole(binding, uid)
  await emit({ status: 'choose_role', message: '请选择当前模拟器登录的账号', roles: roles.map(role => ({
    uid: role.uid, name: role.nickName ?? '已绑定角色', server: role.channelName ?? '',
  })) })
  for (;;) {
    try {
      const chosen = JSON.parse(await readFile(options.selection, 'utf8'))
      if (roles.some(role => String(role.uid) === String(chosen.uid))) return pickRole(binding, chosen.uid)
    } catch (error) {
      if (error.code && error.code !== 'ENOENT') throw error
    }
    await pause(300)
  }
}
try {
  await emit({ status: 'connecting', message: '正在连接森空岛' })
  const roster = await runCollect({ outPath: options.out, credPath: options.cred,
    forceLogin: options.login === 'true',
    loginImpl: supplied => login({ ...supplied, onQr }), rolePicker,
    log: () => {} })
  await emit({ status: 'done', message: `已获取${roster.ops.length}名干员的练度`,
    count: roster.ops.length, player: roster.player })
} catch (error) {
  await emit({ status: 'error', message: error.message })
  process.exitCode = 1
}
