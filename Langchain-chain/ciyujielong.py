from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
import re
import random

# ===================== 【1】加载本地成语库（语料库） =====================
def load_idiom_lib(file_path="chenyu.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except:
        print("❌ 未找到成语文件！")
        exit()

    idioms = re.findall(r"[\u4e00-\u9fa5]{4}", text)
    idiom_list = list(set([i.strip() for i in idioms if len(i.strip()) == 4]))
    return set(idiom_list), {idiom: True for idiom in idiom_list}

IDIOM_SET, IDIOM_DICT = load_idiom_lib()

# ===================== 【2】定义 LangChain 工具（查词库专用） =====================
@tool
def search_idiom(start_char: str) -> list:
    """
    从本地成语库中检索 以指定汉字开头 的所有四字成语
    只能从这里返回的列表里选成语
    """
    candidates = [idiom for idiom in IDIOM_SET if idiom[0] == start_char]
    return candidates

@tool
def validate_and_remove(idiom: str) -> bool:
    """验证成语是否在库中，用过就删除，防止重复"""
    if idiom in IDIOM_SET:
        IDIOM_SET.remove(idiom)
        return True
    return False

# ===================== 【3】接入 Ollama 大模型 =====================
llm = OllamaLLM(model="qwen3.5:9b")  # 换成你本地的模型

# 提示词：强制模型必须用语料库的词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业成语接龙AI。\n"
               "规则：\n"
               "1. 必须使用工具 search_idiom 查到的成语\n"
               "2. 只输出4字成语，不要解释\n"
               "3. 必须以 {char} 开头"),
    ("user", "请接一个以 {char} 开头的成语，从工具返回列表选择")
])

# ===================== 【4】构建 LangChain 调用链 =====================
# 链结构：用户输入 → 模型 → 调用工具查词库 → 模型选择 → 返回最终成语
idiom_chain = (
    {"char": RunnablePassthrough()}
    | prompt
    | llm
)

# ===================== 【5】AI 接龙主函数 =====================
def ai_play(start_char: str):
    # 1. 工具查词库（强制用语料库）
    candidates = search_idiom.invoke(start_char)
    if not candidates:
        return {"ok": False, "idiom": ""}

    # 2. 把候选词给大模型，让模型选一个
    final_idiom = idiom_chain.invoke({
        "char": start_char,
        "candidates": candidates
    }).strip()[:4]  # 强制只取4个字

    # 3. 验证并删除用过的成语
    if validate_and_remove.invoke(final_idiom):
        return {"ok": True, "idiom": final_idiom}
    return {"ok": False, "idiom": ""}

# ===================== 【6】游戏主逻辑 =====================
def play_game():
    print("🎉 成语接龙（LangChain + Ollama + 本地词库）")
    print("✅ AI 必须从你的成语库中查找成语\n")

    current_idiom = random.choice(list(IDIOM_SET))
    IDIOM_SET.remove(current_idiom)
    print(f"AI：{current_idiom}")

    while True:
        need_char = current_idiom[-1]
        user = input(f"\n你【{need_char}】开头：").strip()
        if user == "退出": break

        # 校验用户输入
        if len(user)!=4 or user not in IDIOM_DICT or user[0]!=need_char:
            print("❌ 违规，你输了！")
            break
        IDIOM_SET.remove(user)

        # AI 调用链 + 查词库 接龙
        ai_res = ai_play(user[-1])
        if not ai_res["ok"]:
            print("🏆 AI 接不上，你赢了！")
            break
        current_idiom = ai_res["idiom"]
        print(f"AI：{current_idiom}")

if __name__ == "__main__":
    play_game()