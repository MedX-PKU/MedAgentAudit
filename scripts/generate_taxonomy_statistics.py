import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
# 设置绘图样式
sns.set_theme(style="whitegrid")

def natural_sort_key(code):
    """
    为 '1.10', 'S2.1' 等编号生成自然排序的键。
    - "Other" 始终排在最后。
    - 'S' 开头的成功模式和数字开头的失败模式分开处理。
    - 将点分隔的数字转换为整数元组以进行正确比较 (例如 '1.10' > '1.2')。
    """
    code = str(code)
    if code == "Other":
        # 将 "Other" 放在列表末尾
        return (float('inf'),)

    is_success = code.startswith('S')
    # 移除前缀以获取数字部分
    numeric_part = code[1:] if is_success else code

    try:
        # 分割编号并转换为整数
        parts = [int(p) for p in numeric_part.split('.')]
        # 返回一个元组用于排序，用 0/1 区分成功/失败模式
        return (1 if is_success else 0, *parts)
    except (ValueError, AttributeError):
        # 处理意外格式，将其排在数字编号之后但在 "Other" 之前
        return (float('inf') - 1, code)

# 加载Code映射
def load_code_to_name_mapping(file_path):
    """加载Code到标准名称的映射关系"""
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("警告: 未找到 code_to_name_mapping.json 文件")
        return {}

def map_code_to_name(code, mapping):
    """将Code映射到标准名称"""
    return mapping.get(code, "Other")

def load_simplified_code_mapping(file_path):
    """加载将各种原始/变体 code 归一到数值分级 code 的映射 (simplified_code.json)"""
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("警告: 未找到 simplified_code.json 文件，将直接使用原始 code")
        return {}

def iter_annotation_files(root: Path):
    """遍历所有annotation文件"""
    for p in root.rglob("*.json"):
        if not p.is_file():
            continue
        yield p

def load_records(root: Path, code_mapping, simplified_mapping, model_name=""):
    """加载记录，基于Code进行映射"""
    records = []
    for fp in iter_annotation_files(root):
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            is_correct = bool(data.get("is_correct"))
            # only fetch the failure records
            if not is_correct:
                # 获取primary_classification的code
                code = None
                pc = data.get("primary_classification")
                if isinstance(pc, dict):
                    code = pc.get("code")

                if not code:
                    code = "Unknown"

                # 第一步：通过simplified_code.json映射为简化code
                simplified_code = simplified_mapping.get(code, None)
                if simplified_code: # make sure the code is the categories we need
                    # 第二步：通过code_mapping.json将简化code映射为标准名称
                    mapped_name = map_code_to_name(simplified_code, code_mapping)

                    records.append({
                        "file": str(fp),
                        "is_correct": is_correct,
                        "name": mapped_name,
                        "code": simplified_code,  # 使用简化后的code
                        "original_code": code,    # 保存原始code
                        "model": model_name
                    })
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            records.append({
                "file": str(fp),
                "is_correct": False,
                "name": "Other",
                "code": "Unknown",
                "original_code": "Unknown",
                "model": model_name
            })
    return records # [{file:..., is_correct:..., name:..., code:..., original_code:..., model:...}, ... ]

def aggregate_all_records(all_records):
    """聚合所有记录，并在程序侧进行口径容错：
    - 成功统计只计入以 S 开头的代码；
    - 失败统计只计入以 1/2/3/4 开头的代码,进行筛选，只筛选我们需要的编码类型，其他的删除；
    同时统计口径不一致的异常样本数量。
    """
    success = Counter()
    failure = Counter()
    total_success = 0
    total_failure = 0

    # 按模型分别统计
    model_stats = defaultdict(lambda: {"success": Counter(), "failure": Counter(), "total_success": 0, "total_failure": 0})

    # 异常计数：成功里带失败码 / 失败里带成功码 / 其他
    anomalies = {
        "success_with_failure_code": 0,
        "failure_with_success_code": 0,
        "unclassified_code": 0
    }

    def is_success_code(code: str) -> bool:
        return isinstance(code, str) and code.startswith("S")

    def is_failure_code(code: str) -> bool:
        if not isinstance(code, str) or not code:
            return False
        return code[0] in {"1", "2", "3", "4"}

    for r in all_records:
        model = r.get("model", "Unknown")
        code = r.get("code")
        name = r.get("name", "Other") # in this step, if code can't be mapped, name is "Other" , so we need to clean the code format
        key = (code, name)
        if r["is_correct"]:
            if is_success_code(code):
                success[key] += 1
                model_stats[model]["success"][key] += 1
                total_success += 1
                model_stats[model]["total_success"] += 1
            elif is_failure_code(code):
                anomalies["success_with_failure_code"] += 1
        else:
            if is_failure_code(code):
                failure[key] += 1
                model_stats[model]["failure"][key] += 1
                total_failure += 1
                model_stats[model]["total_failure"] += 1
            elif is_success_code(code):
                anomalies["failure_with_success_code"] += 1

    return success, failure, total_success, total_failure, model_stats, anomalies

def compute_overall_code_level_proportions(all_records):
    """统计整体占比（基于 simplified_code.json）：
    - level3: 使用 simplified 后的 三级数值代码（如 1.1.1）作为键统计；
    - level1/level2: 由上述 level3 代码的前缀聚合得到（即只以 level3 子类求和）。
    仅统计简化后仍为数值型且具有三级结构的代码。
    返回: {"level1": {code: {count, ratio}}, "level2": {...}, "level3": {code: {count, ratio}} }"""
    level3 = Counter()

    def is_numeric_code(code: str) -> bool:
        return isinstance(code, str) and len(code) > 0 and code[0].isdigit()

    # 先按记录统计简化后的 终端代码：
    # - 若有三级（a.b.c），使用 a.b.c
    # - 若仅二级（a.b），使用 a.b
    # - 若仅一级（a），使用 a
    for r in all_records:
        code = r.get("code") or ""
        if not is_numeric_code(code):
            continue
        parts = [p for p in code.split(".") if p]
        if len(parts) >= 3:
            norm_term = f"{parts[0]}.{parts[1]}.{parts[2]}"
        elif len(parts) == 2:
            norm_term = f"{parts[0]}.{parts[1]}"
        elif len(parts) == 1:
            norm_term = parts[0]
        else:
            continue
        level3[norm_term] += 1

    considered_total = sum(level3.values())

    # 由 终端代码 聚合得到 level2 与 level1
    level2 = Counter()
    level1 = Counter()
    for l3_code, cnt in level3.items():
        p = l3_code.split(".")
        l1_code = p[0]
        # 若有二级或三级，给 level2 聚合；只有一级则不计入 level2
        if len(p) >= 2:
            l2_code = f"{p[0]}.{p[1]}"
            level2[l2_code] += cnt
        # level1 总是聚合
        level1[l1_code] += cnt

    def to_ratio_dict(counter: Counter):
        items = {}
        if considered_total == 0:
            return items
        
        # === DEBUG 开始: 捕获并打印导致报错的数据 ===
        try:
            # 尝试执行原来的排序逻辑
            sorted_items = sorted(
                counter.items(),
                key=lambda x: [int(s) if s.isdigit() else s for s in x[0].split('.')]
            )
        except TypeError as e:
            print("\n" + "="*50)
            print("【DEBUG信息】捕获到排序错误，正在列出当前所有待排序的 Code：")
            print(f"错误详情: {e}")
            print("-" * 50)
            
            # 生成排序键并打印，方便肉眼检查
            # 我们将其转换为字符串打印，避免这里也报错
            debug_list = []
            for code, count in counter.items():
                # 模拟出错的那行 lambda 代码生成的键
                sort_key = [int(s) if s.isdigit() else s for s in code.split('.')]
                debug_list.append((code, sort_key))
            
            # 简单按字符串排序展示，方便对比
            debug_list.sort(key=lambda x: str(x[0]))
            
            for code, sort_key in debug_list:
                print(f"Code: {code:<20} | 转换后的排序键: {sort_key}")
            
            print("-" * 50)
            print("【原因分析】请检查上方列表中的‘转换后的排序键’。")
            print("Python 无法比较数字和字符串。")
            print("例子: 如果同时存在 [1, 2] 和 [1, 'Unknown']，")
            print("      在比较第2位时，2(int) < 'Unknown'(str) 就会报错。")
            print("="*50 + "\n")
            raise e  # 重新抛出错误终止程序
        # === DEBUG 结束 ===

        for k, v in sorted(
            counter.items(),
            key=lambda x: [int(s) if s.isdigit() else s for s in x[0].split('.')]
        ):
            items[k] = {
                "count": v,
                "ratio": v / considered_total
            }
        return items

    result = {
        "total_considered": considered_total,
        "level1": to_ratio_dict(level1),
        "level2": to_ratio_dict(level2),
        "level3": to_ratio_dict(level3)
    }
    return result

def create_pie_chart(counter: defaultdict, title: str, out_path: Path, min_percentage=1, fix_overlap=False, startangle=120):
    """创建美化的饼图，支持按大类着色、将编号与百分比显示在饼图外部"""
    if not counter:
        print(f"No data to plot for {title}")
        return

    # 按数量排序，用于决定哪些项目合并到"Other"
    items_by_value = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    total = sum(cnt for _, cnt in items_by_value)

    # 合并小分类到Other
    main_items = []
    other_count = 0

    for (code, name), cnt in items_by_value:
        # percentage = cnt / total * 100 if total > 0 else 0
        # if percentage >= min_percentage and name != "Other": # logic here needs to be fixed，we now don't have 'Other' category
        main_items.append(((code, name), cnt))
        # else:
        #     other_count += cnt

    # # 如果有小分类，添加Other
    # if other_count > 0:
    #     main_items.append((("Other", "Other"), other_count))

    # *** 核心修改：将用于绘图的列表按类别编号进行自然排序 ***
    main_items.sort(key=lambda item: natural_sort_key(item[0][0])) # main_items: [ ( (code, name), count ), ... ]

    if not main_items:
        print(f"No data to plot for {title}")
        return

    # 准备数据
    labels_with_codes = [item[0] for item in main_items]# [ (code, name), ... ]
    sizes = [item[1] for item in main_items]
    total = sum(sizes) # 重新计算总数

    # --- 2. 根据大类生成颜色 (Color logic remains the same) ---
    codes = [lc[0] for lc in labels_with_codes] # [1.1.1, 1.1.2, ...]
    is_success_chart = any(str(c).startswith('S') for c in codes if c and c != "Other")

    major_categories = sorted(list(set([str(c).split('.')[0] for c in codes if c != "Other"])), key=natural_sort_key) # ['1', '2', ...]

    # 为不同的大类选择不同的基准色
    if is_success_chart:
        base_palette = sns.color_palette("Set1", n_colors=len(major_categories))
    else:
        base_palette = sns.color_palette("Set2", n_colors=len(major_categories))

    major_color_map = {cat: color for cat, color in zip(major_categories, base_palette)} # {'1': red, '2':green, ...}

    # 为每个子类生成颜色
    colors = []  # colors = [color1, color2, ...] it involves all the color of the third-level categories
    sub_category_shades = {}
    for code, name in labels_with_codes:
        if code == "Other":
            colors.append("lightgrey")
            continue

        major_cat = str(code).split('.')[0] # major category: '1', '2', ...
        if major_cat not in sub_category_shades:
            count = sum(1 for c, n in labels_with_codes if c != "Other" and str(c).startswith(major_cat)) # how many third-level catgories under the first level
            base_color = major_color_map.get(major_cat, "grey") # Fallback
            # generate sub-category colors
            shades = sns.light_palette(base_color, n_colors=count + 2, reverse=False)[1:] # from base color generate gradient colors, and remove the lightest one
            sub_category_shades[major_cat] = list(shades) # sub_category_shades: {'1': [shade1, shade2, ...], '2': [...]}

        if sub_category_shades.get(major_cat):
            colors.append(sub_category_shades[major_cat].pop(0))
        else:
            colors.append("grey") # Fallback

    # --- 3. 绘制饼图并手动添加外部标签 ---

    # 计算 explode 参数
    explode_list = [0.05 if (val/total)*100 < min_percentage*2 else 0 for val in sizes]
    if fix_overlap:
        # A slightly larger explode for better separation if fixing overlap
        explode_list = [e * 1.1 for e in explode_list]

    # 创建饼图 (不使用 autopct)
    plt.figure(figsize=(16, 12))
    # plt.pie returns (wedges, texts) when autopct is not used.
    wedges, _ = plt.pie(
        sizes,
        startangle=startangle,
        colors=colors,
        counterclock=False, # 让饼图顺时针排列
        explode=explode_list
    )

    # --- 4. 手动添加外部标签 (代码: 百分比%) ---
    # 标签距离圆心的半径
    label_radius = 1.2
    label_fontsize = 48 # 保持和原 autotexts 大小一致

    for i, wedge in enumerate(wedges):
        # 提取数据
        (code, name), count = main_items[i]
        pct = (count / total) * 100

        # 仅为满足最小百分比的项目添加标签
        if pct < min_percentage:
            # 对于不满足最小百分比的项，尝试将百分比显示在饼图内部，但只显示百分比
            if pct > 0:
                ang_deg = (wedge.theta2 + wedge.theta1) / 2.
                rad = np.deg2rad(ang_deg)
                x_inner = 0.5 * np.cos(rad)
                y_inner = 0.5 * np.sin(rad)

                plt.text(
                    x_inner,
                    y_inner,
                    f'{pct:.1f}%',
                    ha='center',
                    va='center',
                    fontsize=label_fontsize, # 稍微小一点的字体
                    color='black'
                )
            continue

        # 计算标签位置
        ang = (wedge.theta2 + wedge.theta1) / 2.  # 中间角度

        # 将角度调整到 -180 到 180 范围
        while ang > 180:
            ang -= 360
        while ang < -180:
            ang += 360

        # 转换为弧度
        rad = np.deg2rad(ang)

        # 标签的坐标
        x_label = label_radius * np.cos(rad)
        y_label = label_radius * np.sin(rad)

        # 确定对齐方式：左侧使用右对齐，右侧使用左对齐
        if x_label > 0:
            ha = 'left' # 右半边
        else:
            ha = 'right' # 左半边

        # 标签内容: code: percentage%
        code = "F" + code if code[0] in {"1", "2", "3", "4"} else code
        label_text = f"{code}: {pct:.1f}%" if code != "Other" else f"Other: {pct:.1f}%"

        # 添加文本
        plt.text(
            x_label,
            y_label,
            label_text,
            ha=ha,
            va='center',
            fontsize=label_fontsize,
            weight='bold'
        )

    # 确保饼图是圆形的
    plt.axis('equal')

    # 原图例相关的代码被移除或保持注释

    # 调整布局以适应图表
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=500, format='pdf', facecolor='white')
    plt.close()
    print(f"Chart saved to {out_path}")

def print_detailed_statistics(success, failure, total_success, total_failure, model_stats):
    """打印详细的统计信息"""
    print(f"\n=== 总体统计 ===")
    print(f"总记录数: {total_success + total_failure}")
    print(f"成功: {total_success}, 失败: {total_failure}")
    print(f"成功率: {total_success / (total_success + total_failure) * 100:.1f}%")

    print(f"\n=== 总体成功类别统计 ===")
    sorted_success = sorted(success.items(), key=lambda item: natural_sort_key(item[0][0]))
    for (code, name), count in sorted_success:
        percentage = count / total_success * 100 if total_success > 0 else 0
        print(f"[{code}] {name}: {count} ({percentage:.1f}%)")

    print(f"\n=== 总体失败类别统计 ===")
    sorted_failure = sorted(failure.items(), key=lambda item: natural_sort_key(item[0][0]))
    for (code, name), count in sorted_failure:
        percentage = count / total_failure * 100 if total_failure > 0 else 0
        print(f"[{code}] {name}: {count} ({percentage:.1f}%)")

    for model, stats in model_stats.items():
        print(f"\n=== {model} 模型统计 ===")
        print(f"成功: {stats['total_success']}, 失败: {stats['total_failure']}")
        if stats['total_success'] + stats['total_failure'] > 0:
            success_rate = stats['total_success'] / (stats['total_success'] + stats['total_failure']) * 100
            print(f"成功率: {success_rate:.1f}%")

def main():
    project_root = Path(__file__).resolve().parents[2] # get to the root file which has the log files
    log_dir = project_root / "logs"
    annot_root = log_dir / "annotation"
    date = time.strftime("%Y%m%d")
    out_dir = log_dir / "taxonomy_statistics" / date 
    out_dir.mkdir(parents=True, exist_ok=True)
    
    code_to_name_file_path = Path(__file__).resolve().parent / "code_to_name_mapping.json"
    code_mapping = load_code_to_name_mapping(file_path = code_to_name_file_path)
    print(f"加载了 {len(code_mapping)} 个Code映射关系")

    simplified_mapping_file_path = Path(__file__).resolve().parent / "simplified_code.json"
    simplified_mapping = load_simplified_code_mapping(file_path = simplified_mapping_file_path)
    print(f"加载了 {len(simplified_mapping)} 个简化Code映射关系")

    all_records = []


    deepseek_root = annot_root / "deepseek-chat"
    if deepseek_root.exists():
        deepseek_records = load_records(deepseek_root, code_mapping, simplified_mapping, "DeepSeek")
        all_records.extend(deepseek_records)
        print(f"加载了 {len(deepseek_records)} 条 DeepSeek 记录")
    else:
        print("警告: 未找到 deepseek-chat 目录")

    gemini_root = annot_root / "gemini-2.5-flash"
    if gemini_root.exists():
        gemini_records = load_records(gemini_root, code_mapping, simplified_mapping, "Gemini")
        all_records.extend(gemini_records)
        print(f"加载了 {len(gemini_records)} 条 Gemini 记录")
    else:
        print("警告: 未找到 gemini-2.5-flash 目录")

    gpt_root = annot_root / "gpt-5"
    if gpt_root.exists():
        gpt_records = load_records(gpt_root, code_mapping, simplified_mapping, "GPT")
        all_records.extend(gpt_records)
        print(f"加载了 {len(gpt_records)} 条 GPT 记录")
    else:
        print("警告: 未找到 gpt-5 目录")

    if not all_records:
        print("错误: 没有找到任何记录")
        return
    
    # print the code in all_records printed for debug
    print("\n" + "="*20 + " DEBUG: Current Unique Codes " + "="*20)
    # 提取 code，转为字符串以防 None，使用 set 去重
    unique_codes = set(str(r.get("code")) for r in all_records)
    # 使用脚本已有的排序逻辑进行排序
    sorted_codes = sorted(unique_codes, key=natural_sort_key)
    
    for c in sorted_codes:
        print(f"'{c}'")
    print("="*63 + "\n")
    success, failure, total_success, total_failure, model_stats, anamolies = aggregate_all_records(all_records) 

    anomolies_file = out_dir / "anomolies.json"
    with anomolies_file.open("w", encoding="utf-8") as f:
        json.dump(anamolies, f, ensure_ascii=False, indent=2)
        
    print_detailed_statistics(success, failure, total_success, total_failure, model_stats)

    # create_pie_chart(
    #     success,
    #     title="success_pie",
    #     out_path=out_dir / "pie_success.pdf",
    #     min_percentage=1,
    #     fix_overlap=True,
    #     startangle=0
    # )

    create_pie_chart(
        failure,
        title="failure_pie",
        out_path=out_dir / "pie_failure.pdf",
        min_percentage=1.5,
        fix_overlap=True
    )

    # for model, stats in model_stats.items():
    #     if stats['total_success'] > 0:
    #         create_pie_chart(
    #             stats['success'],
    #             title=f"{model} - Distribution of Success Patterns",
    #             out_path=out_dir / f"{model.lower()}_success.pdf",
    #             min_percentage=1.5,
    #             fix_overlap=False
    #         )

    #     if stats['total_failure'] > 0:
    #         create_pie_chart(
    #             stats['failure'],
    #             title=f"{model} - Distribution of Failure Modes",
    #             out_path=out_dir / f"{model.lower()}_failures.pdf",
    #             min_percentage=1.5,
    #             fix_overlap=True
    #         )

    overall_level_stats = compute_overall_code_level_proportions(all_records)
    json_out = out_dir / "overall_code_level_distribution.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(overall_level_stats, f, ensure_ascii=False, indent=2)
    print(f"Overall code level distribution saved to {json_out}")

if __name__ == "__main__":
    main()