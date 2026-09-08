# -*- coding: utf-8 -*-
"""推理统一入口：4bit 加载 CareBot（可挂 LoRA），纯文本前缀生成回答。

训练与推理使用同一种前缀格式：
    {instruction}
    患者描述：{ask}
    [检索到的医学知识]
    建议：

生成后会对输出做一轮“去语料味”清理，压掉网上问答语料常见的客套、
复读和跑题尾巴：
  1. 若模型把提示词前缀又复述了一遍，只保留最后一个“建议：”之后的内容；
  2. 出现“答案是 / 参考文献 / 医生询问”等转场标记时，从此处截断；
  3. 整句去重，删除“谢谢 / 祝您 / 仅供参考 / AI身份声明 / 再见”等客套与跑题句；
  4. 过长时按句号边界截断，保证回答紧凑可读。

另外提供两个检索保护函数，供 app_demo 在注入上下文前使用：
  - clean_context_docs：清洗检索片段（只留解答主体、去模板头、去客套尾）；
  - is_context_relevant：关键词相关性门，避免把不相关语料喂给模型。
"""
from __future__ import annotations

import re
from typing import Optional

_INSTRUCTION = (
    "你是一名专业的中文医生助手。请根据患者的描述直接给出结论和可执行的日常建议："
    "先说结论，再给 2~3 条要点。不要复述问题，不要客套话，"
    "不要写“仅供参考”、“祝您早日康复”之类的结尾。"
)

_EXAMPLE = (
    "示例：\n"
    "患者描述：最近总失眠、白天没精神，怎么办？\n"
    "建议：先固定起床和睡觉时间，白天适度运动、午后不喝咖啡，睡前少看手机，"
    "观察两周。如果仍然严重影响白天状态，建议到神经内科或睡眠门诊就诊评估。"
)

_ANSWER_GUIDE = "请用一段简洁的中文直接给出建议（正文不超过250字），说完就停。"

# 出现这些片段说明模型开始转场/跑题/接续别的回答，直接从此截断
_TRANSITION_MARKERS = (
    "答案是", "答案就是", "答案：", "参考答案", "参考文献",
    "最后提醒", "最后再", "医生询问", "欢迎点击", "以上就是我",
    "但由于我是一个", "因为我是一个AI", "我是AI助手",
    "希望对您的帮助", "以上回答", "以上建议", "以上意见", "以上内容",
    "再次提醒", "再次强调", "希望我的回应", "希望我的回复",
    "如果实际状况", "如果实际情况", "作为一个AI", "我是一个AI",
    "我无法提供", "不能替代真正的", "无法替代真正的", "AI助手",
    "非常抱歉", "再见", "注意事项", "回答必须", "以上为我的回答",
)

# 客套/免责结尾特征词（用于删除整句）
_POLITE_TAIL = (
    "仅供参考", "以上回答", "以上建议", "以上意见", "以上信息", "以上内容",
    "谢谢", "感谢", "祝您", "祝你", "祝愿", "祝福", "祈福", "谢谢合作",
    "早日康复", "身体健康", "希望以上", "希望我的回答", "希望能帮",
    "如果还有", "如果有其他", "如有疑问", "如有其他", "如有问题",
    "欢迎随时", "随时欢迎", "竭诚为您", "医生询问",
)

# 以这些开头的句子是语料式客套/免责/跑题句，整句删除
_SENT_DROP_STARTS = (
    "最后提醒", "再提醒", "再次提醒", "再次强调",
    "以上回答", "以上建议", "以上意见", "以上内容", "以上信息",
    "希望以上", "希望对您", "希望这些", "希望我的回答", "希望我的回应",
    "希望我的回复", "希望能够帮助", "如果还有其他", "如果还有任何",
    "如果有任何", "如有疑问", "如有问题", "如需更多", "如果实际",
    "欢迎随时", "请随时联系", "请不要犹豫", "强烈推荐咨询", "推荐咨询",
    "谢谢您的", "感谢您的", "祝您", "祝你", "祝愿", "祝福", "祈福",
    "再见", "非常抱歉", "很抱歉", "抱歉", "由于我是一个", "作为一个AI",
    "我是一个AI", "作为AI助手", "AI助手", "这只是", "这只是一个",
)


def build_medical_prefix(ask: str, instruction: Optional[str] = None,
                         context: Optional[str] = None) -> str:
    """组装生成前缀；格式与训练数据保持一致。"""
    inst = instruction or _INSTRUCTION
    parts = [inst, "", _EXAMPLE, "", f"患者描述：{ask}"]
    if context:
        parts.append(f"检索到的医学知识：\n{context.strip()}")
    parts.append(_ANSWER_GUIDE)
    if context:
        parts.append("（检索资料里与问题直接相关的医学结论请直接采用；资料与问题无关时，请忽略资料，按你的医学常识回答。）")
    parts.append("建议：")
    return "\n".join(parts)


def load_generation_model(model_dir, lora_path: Optional[str] = None,
                          quantize: str = "4bit"):
    """加载基座（4bit），可选挂 LoRA 适配器，返回 (model, tokenizer)。"""
    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16,
                  attn_implementation="sdpa")
    if quantize == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, ask: str, max_new_tokens: int = 220,
                    context: Optional[str] = None, temperature: float = 0.1,
                    instruction: Optional[str] = None) -> str:
    """根据患者描述生成回答，返回清理后的紧凑文本。"""
    import torch
    prefix = build_medical_prefix(ask, instruction=instruction, context=context)
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True)
    return clean_generated(raw)


def clean_generated(raw: str) -> str:
    """清理模型输出：去前缀复读 → 去转场 → 去重复 → 去客套 → 限长。"""
    text = (raw or "").strip()
    if not text:
        return text

    # 去掉“答：/回答：/结论：/A：”式开头
    text = re.sub(r"^\s*(?:答|回答|结论|A)\s*[:：]\s*", "", text)

    # 若模型把提示词整段复述出来，只保留最后一个“建议：”之后的内容
    if "患者描述" in text:
        cut = text.rfind("建议：")
        if cut >= 0:
            text = text[cut + len("建议："):].strip()
        else:
            # 只复读了开头没写“建议：”，从第一个“患者描述：”之后开始保留
            m = re.search(r"患者描述：?", text)
            if m:
                text = text[m.end():].strip()

    # 出现转场标记（开始回答另一个问题 / 列参考 / 客套模板）就截断
    positions = [text.find(m) for m in _TRANSITION_MARKERS]
    positions = [p for p in positions if p >= 0]
    if positions:
        text = text[:min(positions)]

    # 重复句去重：第一次出现与前面相同的整句就停
    text = trim_repetition(text)
    mid = text

    # 删除回答末尾的客套句（至少保留第一句，避免空结果）
    text = _strip_tail_politeness(text)

    # 删除夹在正文里的客套短句、AI身份声明、再见等跑题句
    text = _drop_fluff_sentences(text)

    # 超长按句号边界截断，保持紧凑
    text = _cap_length(text, 300)
    if not text.strip():
        # 极端情况清理后为空，退回原始开头，保证有内容可看
        return (mid or raw or "").strip()[:200]
    return text


def trim_repetition(text: str) -> str:
    """以句子为单位，第一次出现重复句就截断。"""
    if not text:
        return text
    parts = re.split(r"(?<=[。！？；!?；])", text)
    seen = set()
    kept = []
    for seg in parts:
        norm = "".join(seg.split())
        if not norm:
            continue
        if norm in seen:
            break
        seen.add(norm)
        kept.append(seg)
    return "".join(kept).strip()


def _strip_tail_politeness(text: str) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[。！？；!?；])", text)]
    parts = [p for p in parts if p]
    while len(parts) > 1 and any(k in parts[-1] for k in _POLITE_TAIL):
        parts.pop()
    return "".join(parts)


def _drop_fluff_sentences(text: str) -> str:
    """删除客套/免责/跑题整句（开头命中特征词，或为含客套特征的短句）。"""
    parts = [p.strip() for p in re.split(r"(?<=[。！？；!?；])", text)]
    parts = [p for p in parts if p]
    kept = []
    for part in parts:
        norm = "".join(part.split())
        if part.startswith(_SENT_DROP_STARTS):
            continue
        if any(k in part for k in (
                "仅供参考", "医生询问", "希望以上", "AI助手", "角色扮演",
                "我无法提供", "不能提供真正的", "不能替代真正的", "无法替代真正的",
                "因为我无法", "无法进行面诊", "无法进行诊断", "了解完整的情况",
                "我将随时", "如需进一步", "我强烈推荐", "强烈推荐您",
                "很高兴为您服务", "竭诚为您服务")):
            continue
        if len(norm) <= 40 and any(k in part for k in _POLITE_TAIL):
            continue
        kept.append(part)
    return "".join(kept)


def _cap_length(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    chunks = re.split(r"(?<=[。！？；!?；])", text)
    buf = ""
    for chunk in chunks:
        if buf and len(buf) + len(chunk) > limit:
            break
        buf += chunk
    return buf if buf else text[:limit]


# ==================== 检索上下文的质量保护 ====================
# 这套医疗语料里混了不少“营销式客套尾巴”和“多条问答拼接”的脏数据，
# 直接整段喂给模型会把回答带偏。因此在注入上下文前先清洗片段，
# 再用关键词相关性判断是否真的与问题相关；不相关就不注入。

_CTX_FIELDS = ("【科室】", "【主题】", "【问题】", "【解答】", "【答案】")
# 整词停用：只在算相关性前整体去掉
_CTX_STOP_WORDS = (
    "患者", "请问", "医生", "医院", "什么", "怎么", "怎么办", "为什么",
    "如何", "是否", "能不能", "可以", "应该", "问题", "症状", "建议",
    "情况", "最近", "总是", "一直", "经常", "有点", "有没有", "需要",
    "这个", "那个", "这些", "那些", "还有", "反复", "复发", "长期", "吃",
)
# 单字虚词；刻意不删“反/复”等医学用字，避免把“反酸”拆坏
_CTX_STOP_CHARS = frozenset(
    "的吗呢了啊吧哦嗯的了和与或及把被从向在是也会还就要对到都你我他她它们"
    "人点多给来去上问想")


def clean_context_docs(docs) -> list:
    """清洗检索片段：保留解答主体，去掉模板字段与客套尾巴，单条限长。"""
    cleaned = []
    for raw in docs or []:
        if not raw:
            continue
        d = raw.strip()
        # 优先取【解答】之后的主体；没有【解答】字段就取第一个字段之后
        if "【解答】" in d:
            d = d.split("【解答】", 1)[1]
        else:
            idxs = [d.find(f) for f in _CTX_FIELDS]
            idxs = [i for i in idxs if i >= 0]
            if idxs:
                d = d[min(idxs):]
        # 若又出现第二个字段头，说明混入多条问答，截断
        for f in _CTX_FIELDS:
            j = d.find(f)
            if j >= 0:
                d = d[:j]
                break
        # 去掉遗留字段标签
        d = re.sub(r"【(?:科室|主题|问题|解答|答案)】", "", d)
        # 从第一句客套句处截断（模型不需要答案尾巴的客套话）
        parts = [p.strip() for p in re.split(r"(?<=[。！？；!?；])", d)]
        kept = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if (p.startswith(_SENT_DROP_STARTS)
                    or any(k in p for k in _POLITE_TAIL)
                    or any(k in p for k in ("仅供参考", "医生询问", "AI助手",
                                            "角色扮演"))):
                break
            kept.append(p)
        body = "".join(kept)[:400].strip(" \n·，：")
        if body:
            cleaned.append(body)
    return cleaned


def _context_grams(text: str):
    """把问题/片段归一成“内容二字组”集合，去掉标点与停用表达。"""
    t = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    for w in _CTX_STOP_WORDS:
        t = t.replace(w, "")
    t = "".join(ch for ch in t if ch not in _CTX_STOP_CHARS)
    if len(t) < 2:
        return set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def is_context_relevant(question: str, doc: str) -> bool:
    """关键词门：问题和候选片段至少共享一个二字片段才算相关。

    注意：请传“原始检索片段”（含主题/问题字段）来判断相关性，
    因为清洗后的解答往往把同义词改写掉了，会误伤有效召回。
    """
    if not doc:
        return False
    return bool(_context_grams(question) & _context_grams(doc))