"""战场标准状态（State）的 schema 定义。

由 perception/state_builder 从 bridge 原始输出构建，供决策模型与轨迹落盘消费；
敌人位置使用"路线 + 进度"而非像素，干员 / 敌人以数值特征向量（featvec）标识。
"""

from pydantic import BaseModel, Field

from ark_core.protocol.version import PROTOCOL_VERSION


class State(BaseModel):
    """战场标准状态（简化 stub，字段类型从简）。

    完整字段见 RFC-0001（docs/rfc/0001-state-action-protocol.md）。
    当前仅覆盖最小可校验字段集合，子结构暂以 dict 占位。
    """

    protocol_version: str = Field(default=PROTOCOL_VERSION, description="协议版本号（semver）")
    t_ms: int = Field(default=0, description="距开局的毫秒时间戳")
    stage: dict = Field(default_factory=dict, description="关卡信息：code / family")
    cost: int = Field(default=0, description="当前费用")
    life: int = Field(default=0, description="剩余生命值")
    kill_count: int = Field(default=0, description="击杀数")
    tiles: list[dict] = Field(default_factory=list, description="格子列表（r/c/kind/device）")
    hand: list[dict] = Field(default_factory=list, description="待部署干员卡（featvec/费用/冷却）")
    deployed: list[dict] = Field(default_factory=list, description="已部署干员（位置/朝向/血技比）")
    enemies: list[dict] = Field(default_factory=list, description="敌人（种类/路线进度/血比/速度）")

    # TODO(M1): 按 RFC-0001 拆出 Tile / HandCard / DeployedOperator / Enemy 子模型，
    #           并将 dict 字段替换为强类型；旧版数据由 migrations/ 迁移。
