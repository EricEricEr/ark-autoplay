// 终端二维码渲染：本地生成（不经过任何第三方服务），半块字符两模块压一行。
// vendored: scripts/skland/lib/vendor/qrcode-generator.js（qrcode-generator 1.4.4，MIT，Kazuhiko Arase）
// 取代 2026-10-03 前的 api.qrserver.com 外链方案——登录深链内容不再出本机。
import qrcode from './vendor/qrcode-generator.js'

export function renderTerminalQr(data) {
  const qr = qrcode(0, 'M') // 0=自动选版本，M 容错可扫码（含 logo 遮挡余量）
  qr.addData(data, 'Byte')
  qr.make()
  const n = qr.getModuleCount()
  const isDark = (r, c) => (r >= 0 && r < n && c >= 0 && c < n ? qr.isDark(r, c) : false)
  const lines = []
  // 2 模块静区（手机扫码必需的空白边）
  for (let r = -2; r < n + 2; r += 2) {
    let line = ''
    for (let c = -2; c < n + 2; c++) {
      const top = isDark(r, c)
      const bot = isDark(r + 1, c)
      line += top ? (bot ? '█' : '▀') : bot ? '▄' : ' '
    }
    lines.push(line)
  }
  return lines.join('\n')
}
