// 合成数据、假网络：不读取账号凭证、不访问真实服务。
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { buildRoster, login, pickRole, runCollect } from './collect.mjs'

const fixture = {
  status: { uid: 'SYNTHETIC_UID', name: '合成博士' },
  charInfoMap: { synthetic_op: { name: '合成干员' } },
  chars: [{
    charId: 'synthetic_op', evolvePhase: 2, level: 80, potentialRank: 5,
    mainSkillLvl: 7, skills: [
      { id: 'synthetic_s1', specializeLevel: 0 },
      { id: 'synthetic_s2', specializeLevel: 3 },
    ],
    defaultEquipId: 'synthetic_e2', equip: [
      { id: 'synthetic_e1', level: 1, locked: false },
      { id: 'synthetic_e2', level: 3, locked: false },
    ],
  }],
}
const binding = { list: [{ appCode: 'arknights', defaultUid: 'u1', bindingList: [
  { uid: 'u1', nickName: '合成甲' }, { uid: 'u2', nickName: '合成乙' },
] }] }
const respond = data => ({ ok: true, status: 200, text: async () => JSON.stringify(data) })
async function temporary(t) {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'ark-skland-test-'))
  t.after(() => rm(dir, { recursive: true, force: true }))
  return dir
}

test('all skills and modules are preserved; ordinary level 7 stays 7', () => {
  const roster = buildRoster(fixture)
  assert.equal(roster.ops[0].skill_level, 7)
  assert.deepEqual(roster.ops[0].skills.map(s => s.skill_level), [7, 10])
  assert.deepEqual(roster.ops[0].equip, fixture.chars[0].equip)
  assert.equal(roster.ops[0].potential, 6)
  assert.equal(roster.ops[0].defaultEquipId, 'synthetic_e2')
  assert.deepEqual(roster.raw_player_info, fixture)
})

test('unknown names are retained without requiring an old operator catalog', () => {
  const roster = buildRoster({ chars: [{ charId: 'synthetic_future', level: 1 }] })
  assert.equal(roster.ops.length, 1)
  assert.equal(roster.ops[0].name, null)
  assert.equal(roster.ops[0].mainSkillLvl, null)
  assert.equal(roster.unresolved[0].reason, 'name_missing')
})

test('explicit UID cannot silently fall back to a different account', () => {
  assert.equal(pickRole(binding).uid, 'u1')
  assert.equal(pickRole(binding, 'u2').uid, 'u2')
  assert.throws(() => pickRole(binding, 'unknown'))
})

test('offline fixture never logs in, reads credentials, or sends a request', async t => {
  const dir = await temporary(t)
  const input = path.join(dir, 'input.json'), outPath = path.join(dir, 'roster.json')
  await writeFile(input, JSON.stringify({ code: 0, data: fixture }))
  const fail = () => { throw new Error('Network/login must not be called') }
  await runCollect({ outPath, fixturePath: input, credPath: dir,
    fetchImpl: fail, loginImpl: fail, log: () => {} })
  assert.deepEqual(JSON.parse(await readFile(outPath, 'utf8')).raw_player_info, fixture)
})

test('first use logs in once and keeps login tokens out of exported roster', async t => {
  const dir = await temporary(t), messages = []
  const outPath = path.join(dir, 'roster.json'), credPath = path.join(dir, '.secrets/cred.json')
  let logins = 0
  const loginImpl = async () => {
    logins++
    return { cred: 'FAKE_PRIVATE_CRED', token: 'FAKE_PRIVATE_TOKEN', dId: 'FAKE_DID' }
  }
  const fetchImpl = async url => {
    if (url.includes('/player/binding')) return respond({ code: 0, data: binding })
    if (url.includes('/player/info')) return respond({ code: 0, data: fixture })
    throw new Error('Unexpected fake request')
  }
  await runCollect({ outPath, credPath, uid: 'u2', fetchImpl, loginImpl,
    log: s => messages.push(s) })
  const output = await readFile(outPath, 'utf8')
  assert.equal(logins, 1)
  assert.equal(JSON.parse(output).player.uid, 'u2')
  assert.ok(!output.includes('FAKE_PRIVATE_'))
  assert.ok(!messages.join('\n').includes('FAKE_PRIVATE_'))
  assert.equal(JSON.parse(await readFile(credPath, 'utf8')).cred, 'FAKE_PRIVATE_CRED')
})

test('existing session refreshes once and does not require another scan', async t => {
  const dir = await temporary(t), credPath = path.join(dir, 'cred.json')
  await writeFile(credPath, JSON.stringify({ cred: 'FAKE_CRED', token: 'OLD', dId: 'D' }))
  let bindings = 0, refreshes = 0
  const fetchImpl = async url => {
    if (url.includes('/player/binding')) {
      bindings++
      return respond(bindings === 1 ? { code: 10001 } : { code: 0, data: binding })
    }
    if (url.includes('/auth/refresh')) {
      refreshes++
      return respond({ code: 0, data: { token: 'REFRESHED' } })
    }
    if (url.includes('/player/info')) return respond({ code: 0, data: fixture })
    throw new Error('Unexpected fake request')
  }
  await runCollect({ outPath: path.join(dir, 'out.json'), credPath, fetchImpl,
    loginImpl: () => { throw new Error('No QR expected') }, log: () => {} })
  assert.equal(refreshes, 1)
  assert.equal(bindings, 2)
  assert.equal(JSON.parse(await readFile(credPath, 'utf8')).token, 'REFRESHED')
})

test('QR confirmation, token exchange and grant run with mocked endpoints', async () => {
  const fetchImpl = async url => {
    let data
    if (url.includes('/gen_scan/login')) data = { status: 0, data: { scanId: 'SYN_SCAN' } }
    else if (url.includes('/scan_status')) data = { status: 0, data: { scanCode: 'SYN_CODE' } }
    else if (url.includes('/token_by_scan_code')) data = { status: 0, data: { token: 'SYN_HG_TOKEN' } }
    else if (url.includes('/oauth2/v2/grant')) data = { status: 0, data: { code: 'SYN_GRANT' } }
    else if (url.includes('/generate_cred_by_code')) data = {
      code: 0, data: { cred: 'SYN_PRIVATE_CRED', token: 'SYN_PRIVATE_TOKEN' },
    }
    else throw new Error('Unexpected fake request')
    return respond(data)
  }
  const messages = []
  const cred = await login({ fetchImpl, log: s => messages.push(s) })
  assert.equal(cred.cred, 'SYN_PRIVATE_CRED')
  assert.ok(!messages.join('\n').includes('SYN_PRIVATE_'))
})

test('invalid/empty response does not overwrite an existing export', async t => {
  const dir = await temporary(t), outPath = path.join(dir, 'roster.json')
  const input = path.join(dir, 'empty.json')
  await writeFile(outPath, '{"previous":true}')
  await writeFile(input, '{"chars":[]}')
  await assert.rejects(runCollect({ outPath, fixturePath: input, log: () => {} }))
  assert.equal(await readFile(outPath, 'utf8'), '{"previous":true}')
})

test('--help is available without login or static game data', () => {
  const script = fileURLToPath(new URL('./collect.mjs', import.meta.url))
  const result = spawnSync(process.execPath, [script, '--help'], { encoding: 'utf8' })
  assert.equal(result.status, 0)
  assert.ok(result.stdout.includes('--out'))
})
