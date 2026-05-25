import os
import json
import random
import requests
from dotenv import load_dotenv
from typing import Optional, Dict
from utils.config_handler import agent_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_root
from langchain_core.tools import tool
from RAG.rag_service import RagSummarizerService

load_dotenv()

BAIDU_MAP_AK = os.getenv("BAIDU_MAP_AK", "")
BAIDU_MAP_API = "https://api.map.baidu.com"

# 检查AK
if not BAIDU_MAP_AK:
    logger.warning("未配置BAIDU_MAP_AK，天气功能将使用模拟数据")

# 城市 -> 行政区划代码
CITY_TO_DISTRICT_ID = {
    "北京": "110000", "上海": "310000", "深圳": "440300", "广州": "440100",
    "成都": "510100", "杭州": "330100", "武汉": "420100", "西安": "610100",
    "南京": "320100", "重庆": "500000", "天津": "120000", "苏州": "320500",
    "郑州": "410100", "长沙": "430100", "青岛": "370200", "宁波": "330200",
    "无锡": "320200", "厦门": "350200",
}

rag = RagSummarizerService()
user_ids = [f"100{i}" for i in range(1, 11)]
month_arr = [f"2025-{i:02d}" for i in range(1, 13)]

external_data = {}


def get_city_district_id(city_name: str) -> Optional[str]:
    """获取城市行政区划代码"""
    # 优先使用内置映射
    city_name = city_name.replace("市", "").strip()
    if city_name in CITY_TO_DISTRICT_ID:
        return CITY_TO_DISTRICT_ID[city_name]
    return None


def get_ip_location() -> Optional[str]:
    """通过IP获取当前城市"""
    if not BAIDU_MAP_AK:
        return None

    try:
        url = f"{BAIDU_MAP_API}/location/ip"
        params = {"ip": "", "ak": BAIDU_MAP_AK, "coor": "bd09ll"}
        res = requests.get(url, params=params, timeout=5).json()

        if res.get("status") == 0:
            parts = res.get("address", "").split("|")
            if len(parts) >= 3:
                city = parts[-2] or parts[-3]
                return city.replace("市", "")
    except Exception as e:
        logger.error(f"IP定位失败: {e}")
    return None


def get_real_weather(city: str, data_type: str = "all") -> Dict:
    """调用百度地图天气API（修复版）"""
    if not BAIDU_MAP_AK:
        return {"use_mock": True}

    district_id = get_city_district_id(city)
    if not district_id:
        ip_city = get_ip_location()
        district_id = get_city_district_id(ip_city) if ip_city else None
        if not district_id:
            return {"error": "不支持该城市", "use_mock": True}

    try:
        url = f"{BAIDU_MAP_API}/weather/v1/"
        params = {
            "ak": BAIDU_MAP_AK,
            "district_id": district_id,
            "data_type": data_type
        }
        res = requests.get(url, params=params, timeout=10).json()
        if res.get("status") == 0:
            return res.get("result", {})
        return {"error": res.get("message"), "use_mock": True}
    except Exception as e:
        logger.error(f"天气API请求失败: {e}")
        return {"error": str(e), "use_mock": True}


def format_weather_message(weather_data: Dict, city: str) -> str:
    """格式化天气信息"""
    if weather_data.get("use_mock"):
        mock = random.choice([
            f"天气晴朗，25°C，湿度45%，南风2级，空气质量优",
            f"多云，22°C，湿度55%，东北风1级，空气质量良",
            f"小雨，18°C，湿度80%，北风2级，空气质量优",
        ])
        return f"📍{city} {mock}（演示数据）"

    try:
        now = weather_data.get("now", {})
        return (
            f"📍{weather_data.get('location', {}).get('city', city)} | {now.get('text')} | "
            f"🌡️{now.get('temp')}°C | 💧湿度{now.get('rh')}% | 🌬️{now.get('wind_dir')}"
        )
    except:
        return f"📍{city} 天气数据解析失败"


def get_real_user_location() -> str:
    city = get_ip_location()
    return city if city else random.choice(list(CITY_TO_DISTRICT_ID.keys()))


# ==================== 工具函数（修复版） ====================

@tool(description="从向量库检索参考资料")
def rag_summarize(query: str) -> str:
    try:
        return rag.rag_summarize(query)
    except Exception as e:
        logger.error(f"RAG失败: {e}")
        return f"检索失败：{str(e)}"


@tool(description="获取城市实时天气，输入：城市名")
def get_weather(city: str) -> str:
    try:
        data = get_real_weather(city)
        return format_weather_message(data, city)
    except:
        return f"📍{city} 天气服务暂时不可用"


@tool(description="获取用户当前所在城市（IP定位）")
def get_user_location() -> str:
    return get_real_user_location()


@tool(description="随机获取一个用户ID")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="随机获取一个月份，格式：2025-01")
def get_random_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """加载外部CSV数据（修复拼写错误）"""
    global external_data
    if external_data:
        return

    try:
        path = get_abs_root(agent_config["external_data_path"])  # 修复拼写
        if not os.path.exists(path):
            logger.warning("外部数据文件不存在")
            return

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                arr = line.split(",")
                if len(arr) < 6:
                    continue
                uid, feat, eff, consum, comp, tm = [i.strip() for i in arr]
                external_data.setdefault(uid, {})[tm] = {
                    "特征": feat, "效率": eff, "耗材": consum, "对比": comp
                }
        logger.info(f"外部数据加载完成：{len(external_data)} 个用户")
    except Exception as e:
        logger.error(f"加载外部数据失败: {e}")


@tool(description="获取用户指定月份使用记录，参数：user_id、month")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"无数据：{user_id} {month}")
        return ""


@tool(description="为报告生成注入上下文")
def fill_context_for_report() -> str:
    return "[上下文注入成功]"