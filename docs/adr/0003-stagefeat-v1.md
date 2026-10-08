# ADR-0003：关卡静态特征 v1（stagefeat）

- 状态：已接受（2026-10-09）
- 范围：`src/ark_core/datapipe/stagefeat.py`、`configs/tile_vocab.yaml`、`configs/data.yaml`（stagefeat 小节）、`tests/unit/test_datapipe_stagefeat.py`
- 数据版本：`77.6.0+a550f5e+ak_dataset_v0.1`
- 产物：`stagefeat.parquet`（2459 关 × 20 列）

## 背景

datapipe v1 已把**干员 / 敌人数值**（featvec）、**关卡元信息**（stages_registry）、
**作业动作**（episodes）转为模型可消费产物，但**关卡本身的空间结构**（地形、路线、
出怪时序）一直停留在 `stages.jsonl` 的嵌套 JSON 形态：格子是字符串 token、路线是变长
checkpoint 列表、出怪是 (t, enemy, route) 三元组。模型无法直接消费。

本次把这一块补齐。范围严格限定在**静态数据**——即"这张图长什么样、敌人什么时候从哪
来"，不含任何运行时观测（战场实时状态属感知层，另行处理）。

## 决策

### 1. 产物形态：一关一行，三个张量列

| 列 | 内容 |
|---|---|
| `grid` | row-major 展平的格子矩阵，每格 5 通道：`h` / `build` / `deployable` / `is_start` / `is_end` |
| `routes` | 路线列表，每条含 `mode` / `start` / `end` / `n_points` / `points`（扁平 [r,c,...]）/ `wait_ms` |
| `spawns` | 出怪时间表（`t_ms` 升序），含 `wave` / `route` / `enemy_id` / `enemy_matched` |

外加 `rows` / `cols` / `n_routes` / `n_spawns` 与 5 个 `opt_*` 关卡参数（费用、生命、编队上限）。

### 2. 变长而非 padding 到定长

实测网格 6×9 ~ 40×40（平均约 10×10，共 244040 格）。若一律补齐到 64×64 需 10,072,064 格，
**放大 41.3 倍**且绝大部分是 padding。故 `grid` 按 `rows × cols` 原样展平，消费方按行内
自带的 `rows` / `cols` reshape；路线同理按条变长。

超上限（网格 64×64 / 单路线 256 点）**报错而非静默截断**——截断会丢地形且不留痕。

### 3. 坐标系统一为 `[row, col]`（本 ADR 最重要的一条）

真实数据里存在**两套坐标系**，已用数据交叉验证：

| 来源 | 坐标序 | 证据 |
|---|---|---|
| `grid` 索引 / `routes` 的 `start`/`end`/`pos` | **`[row, col]`** | route.start 按 [row,col] 解读命中起点格 3990 次，按 [col,row] 仅 342 次 |
| MAA 作业 deploy 的 `location` | **`[x, y] = [col, row]`** | 28076 条真实 deploy 中，按此解读 96.8% 落在可部署格，按 [row,col] 仅 41.7% |

产物统一输出 `[row, col]`（与 GameData 原生一致），并提供
`stagefeat.normalize_deploy_location([x,y]) -> (row,col)` 做显式转换。
**凡消费 episodes 中 deploy 位置的代码都必须先过该函数**；混用坐标系是 ReviewBench
rubric「Correctness」类列名的缺陷（本项目方案 §2.2 明确举例"坐标系混用"）。

### 4. 数字编码的语义翻译（编码表即配置）

`stages.jsonl` 的格子 / 路线字段存在字符串与数字两套写法，且**同关内不混用**
（实测 3279 关：纯字符串 3047 / 全数字 232 / 混用 **0**）。数字码语义不在数据里，
由交叉验证反推，全部落在 `configs/tile_vocab.yaml`：

| 字段 | 数字码 | 语义 | 判定依据 |
|---|---|---|---|
| `h` | `0` / `1` | LOWLAND / HIGHLAND | `1` 仅配 NONE/RANGED，与 `tile_wall`(HIGHLAND,RANGED) 一致 |
| `build` | `0`/`1`/`2`/`3` | NONE/MELEE/RANGED/ALL | `1`↔`tile_road`(MELEE)、`2`↔`tile_wall`(RANGED)、`0`↔tile_start/end/hole(NONE) |
| route `mode` | `0` / `1` | WALK / FLY | `1` 的起点格中 tile_flystart/tile_hole 占比显著高于 `0` |

实测构建结果：`unknown_h_codes` 与 `unknown_build_codes` **均为空**（零未解析），
`mode` 仅 `E_NUM` 9644 条落 unknown——它是纯占位路线，从不被 spawns 引用（见下）。

### 5. checkpoint 分类：wait 类的 `[0,0]` 不得混入路径

checkpoint 同样有字符串与数字两套类型码。按形态交叉验证分为三类：

| 类别 | 字符串码 | 数字码 | 形态证据 | 处理 |
|---|---|---|---|---|
| move | MOVE / PATROL_MOVE / APPEAR_AT_POS / MAP_OFFSET_MOVE | `0` / `6` | pos 恒为真实坐标、time=0 | 贡献路径点 |
| wait | WAIT_FOR_SECONDS / WAIT_CURRENT_{FRAGMENT,WAVE}_TIME / WAIT_BOSSRUSH_WAVE | `1` / `3` / `4` | pos=[0,0] 占比 92.8%~100%、time>0 | 只记时长到 `wait_ms` |
| other | DISAPPEAR | `5` | pos=[0,0]、time=0 | 丢弃并计数 |

**若把 wait 类的 `[0,0]` 当作路径点，路线几何会被拖到地图左上角**——这是本模块的
核心正确性约束，单测 `test_wait_checkpoints_do_not_pollute_path_geometry` 固化。

### 6. 关卡变体不重复入表

`#f#`（四星）775 关 + `#s`（六星）45 关，与其主关卡共用同一 `level_id`，且
grid / routes / spawns / options **实测完全一致**（888 组变体、0 差异）。
默认跳过（`stagefeat.include_variants: false`），避免同图重复放大采样权重。
产物中保留 `is_variant` 列以便追溯。

## 实测构建结果

```
n_stages            2459   （3279 全量 - 820 变体）
n_routes            99024
n_spawns            143421
n_grid_cells        244040
n_null_routes       599    （routes 数组内的 null 占位，从不被 spawns 引用）
max_grid            40×40  （act2multi_tr02）
max_route_points    125
unknown_h/build     空     （编码表 100% 覆盖）
```

产物验证（`data-validation/verify_stagefeat.py`，把张量还原成 ASCII 地图再与源 JSON 对照）：

- 1-7 还原地图的三条通道与左侧终点、右侧三个出生点位置正确；
- **可部署位逐格比对 77 格，不一致 0**；
- 路线数 27 vs 源 27、出怪数 41 vs 源 41、出怪时刻逐一比对**不一致 0**；
- 全量不变式：grid 长度自洽、`n_points` 与 `points` 自洽、`spawns.route` 越界 0。

## 已知边界（如实记录，未解决）

1. **敌人等级档未接入**：`spawns.enemy_id` 直接命中 featvec `enemy_id` 99.4%
   （143421 条中未命中约 1121 条，118 种，多为 `_a`/`_b`/`_c` 变体与活动特供敌）。
   但**同一 enemy_id 在 featvec 里有 1~3 个等级档**（1923 个 id 单档 / 217 个双档 /
   23 个三档），具体用哪一档由 `level_*.gamedata.json` 的 `enemyDbRefs`
   （`{id, level}`，实测 24300 条，level 分布 0:23010 / 1:816 / 2:474）决定。
   本 v1 **未消费 enemyDbRefs**，即产物给出了敌人 id 但未给出该关的实际等级档。
   TODO(M1)：接入 gamedata-levels 的 enemyDbRefs 以闭合数值口径。
2. **E_NUM 路线未解释**：9644 条 mode=E_NUM 的路线 start/end 恒为 [0,0]、
   checkpoints 为空，且从不被 spawns 引用，判定为占位并保留 unknown 计数。
3. **`routes` 的 599 个 null 占位**：与 E_NUM 同理，跳过并计数。
4. **出怪时序对齐未做**：`spawns` 的 `t` 是波次内的相对时刻，而"下一波何时推进"
   取决于阻塞敌人清场情况（`waves_level_*.json` 的 `waveGate` 模型有描述，
   见泰拉演算台 simulation 数据）。v1 未做波次推进推算。
5. **敌人路线进度**（协议 State 里的 `pos.route + progress`）需由路线几何 + 移动
   速度实时推算，属感知层，不在本静态转换范围。

## 后果

- 正：关卡空间结构首次成为模型可直接消费的整型张量；编码口径全部入 configs；
  坐标系混用这一高危缺陷被显式收敛到单一函数并有单测固化；产物体积仅 1.4MB。
- 负：`grid` 变长意味着训练侧的 collate 需按 `rows`/`cols` 自行 padding，
  本模块不代做（避免在数据层引入不可见的信息损失）。
- 验证基线：2459 关 / 99024 路线 / 143421 出怪 / 244040 格 —— 计数可复现
  （`build_report.json` 的 `sections.stagefeat` 可见）。
