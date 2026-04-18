# MathorCup 2026 C 题建模与复现项目

## TODO:
1. 痰湿积分动力学里对于降低积分：频率达到5的阈值才开始有效，一级强度是否对降低积分有效？
2. 最优化方案：预算成本的权重是怎么来的？可不可以对题目理解如下：应以最大程度降低痰湿积分为主要目标，预算仅为边界约束条件，不应作为双目标去进行优化。在积分降低程度相同时再去选择预算低的方案。若二者都相同则选择时间成本最小的？
3. 在这六个月中如果过完某个月后所选方案是不是可以改变？不用选一个方案然后去


## 项目简介

本仓库是 MathorCup 2026 C 题《中老年人群高血脂症的风险预警及干预方案优化》的完整建模与复现工程，包含：

- 论文正文与 LaTeX 模板
- 三道问题对应的 Python 实现
- 样例数据
- 已生成的图表与表格结果

项目整体采用“数据清洗与质量诊断 -> 关键指标筛选 -> 三级风险预警 -> 6 个月个性化干预优化”的建模主线，既可用于论文整理与提交，也可用于结果复现、组内协作和后续维护。

论文基本信息如下：

- 题号：C
- 题目：中老年人群高血脂症的风险预警及干预方案优化
- 队伍编号：`MC2606970`

## 研究内容与三道题对应关系

### Q1 关键指标筛选与体质贡献度分析

目标是从血常规指标和中老年人活动量表评分中筛选出：

- 能表征痰湿体质严重程度的关键指标
- 能预警高血脂发病风险的关键指标
- 九种中医体质对高血脂发病风险的贡献差异

仓库对应脚本：

- `src/pyCode/problem1.py`

### Q2 三级风险预警模型

目标是建立低风险、中风险、高风险三级预警体系，并说明分层阈值与融合逻辑。

仓库对应脚本：

- `src/pyCode/problem2.py`

### Q3 6 个月个性化干预方案优化

目标是针对痰湿体质患者，在预算、年龄、活动能力等约束下，为 6 个月干预过程给出个体化最优策略。

仓库对应脚本：

- `src/pyCode/problem3.py`

## 方法框架与核心结论

### 整体方法框架

论文主体围绕以下四步展开：

1. 对样本数据做质量检查、标签一致性检查和异常值诊断。
2. 针对问题一进行候选特征构建、多重共线性诊断和多方法集成排序。
3. 针对问题二构建“规则层 + 模型层”融合的三级风险预警模型。
4. 针对问题三建立 6 个月有限时域动态规划模型，输出个性化干预方案。

### 数据质量与预处理结论

- 样本规模为 1000 例。
- 数据中未发现缺失值与重复样本 ID。
- ADL/IADL 子项之和与总分完全一致。
- 检出 65 例体质标签与最高积分体质不一致的判定歧义样本，分析时保留并在论文中说明。
- 诊断标签与“血脂异常项数 >= 1”一致率为 100%，这对后续风险建模的特征使用方式有直接影响。

### Q1 方法与结论摘要

问题一采用以下方法组合：

- VIF 多重共线性诊断
- Spearman 秩相关
- 互信息
- LASSO 稀疏回归
- 随机森林置换重要度
- 偏相关分析
- Borda 排序聚合

代表性结果如下：

- 表征痰湿体质严重程度的 Top-5 指标为：`TC`、`ADL吃饭`、`BMI`、`TG`、`ADL洗澡`
- 预警高血脂发病风险的 Top-5 指标为：`TG`、`TC`、`血尿酸`、`HDL-C`、`LDL-C`
- 九种体质的发病率差异整体不显著
- 本数据中痰湿体质发病率为 77.3%，低于整体均值 79.3%

### Q2 方法与结论摘要

问题二采用“规则层 + 模型层”融合预警：

- 模型层以 L2 正则化 Logistic 回归为主
- 随机森林和 GBDT 作为对照模型
- 规则层结合血脂异常项数、痰湿积分、活动评分和 BMI 进行分级
- 最终风险等级由规则层和模型层取较高等级得到

代表性结果如下：

- Logistic 回归 5 折交叉验证 AUC 为 `0.974`
- 最终分层结果为：低风险 `146` 人、中风险 `236` 人、高风险 `618` 人
- 三层风险的实际发病率分别为：`0%`、`74.2%`、`100%`
- 痰湿体质高风险核心组合之一为“痰湿积分 >= 60 且血脂异常 >= 1”

### Q3 方法与结论摘要

问题三建立了 6 个月有限时域动态规划模型：

- 状态核心为痰湿积分
- 每月决策包括调理分级、活动强度和周训练频次
- 同时约束预算、年龄组、活动能力和训练强度可行范围
- 目标综合考虑疗效、成本和可耐受性

代表性结果如下：

- 共为 278 名痰湿体质患者求解方案
- 年龄越小、活动能力越强的患者，更倾向于较高强度方案
- 活动受限或年龄较高的患者，只能选择较低强度干预
- 论文中展示的 3 个样本，6 个月痰湿积分降幅分别约为 `21.0%`、`31.0%`、`43.2%`

## 仓库结构

```text
Reposit
├─ main.tex
├─ MathorCupmodeling.cls
├─ README.md
├─ main.pdf
├─ src
│  ├─ Data
│  │  └─ 附件1：样例数据.xlsx
│  ├─ pyCode
│  │  ├─ common.py
│  │  ├─ problem1.py
│  │  ├─ problem2.py
│  │  └─ problem3.py
│  └─ outputs
│     ├─ figures
│     └─ tables
└─ .gitignore
```

各部分职责说明：

- `main.tex`：论文主文档，包含摘要、正文、参考文献与附录。
- `MathorCupmodeling.cls`：MathorCup 论文模板类文件。
- `src/Data`：样例数据输入目录。当前脚本会在该目录下查找文件名包含“样例数据.xlsx”的 Excel 文件。
- `src/pyCode/common.py`：公共路径、绘图配置、随机种子和数据读取工具函数。
- `src/pyCode/problem1.py`：问题一脚本，输出相关图表、排名表与贡献度结果。
- `src/pyCode/problem2.py`：问题二脚本，输出交叉验证结果、特征重要度、风险分层结果与图表。
- `src/pyCode/problem3.py`：问题三脚本，输出样本路径、全体痰湿体质患者优化结果与图表。
- `src/outputs/figures`：保存生成的 PNG 图像。
- `src/outputs/tables`：保存生成的 CSV 和 TXT 结果表。

## 环境与运行方式

### Python 依赖

本仓库当前脚本至少依赖以下 Python 包：

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `openpyxl`

如果需要手动安装，可按需执行：

```bash
pip install numpy pandas matplotlib scipy scikit-learn statsmodels openpyxl
```

### LaTeX 环境

当前仓库已实际使用 XeLaTeX 编译论文，`main.log` 中记录的环境为：

- XeTeX / XeLaTeX
- TeX Live 2026

因此推荐使用 XeLaTeX 作为默认编译方式。

### 数据输入与输出路径

根据 `src/pyCode/common.py` 的路径约定：

- 输入数据目录：`src/Data`
- 图像输出目录：`src/outputs/figures`
- 表格输出目录：`src/outputs/tables`

三个脚本运行时会自动创建输出目录。

### Python 脚本运行

建议在仓库根目录下执行以下命令：

```bash
python src/pyCode/problem1.py
python src/pyCode/problem2.py
python src/pyCode/problem3.py
```

说明：

- `problem1.py` 负责问题一的数据质量分析、特征筛选与体质贡献度分析。
- `problem2.py` 负责问题二的三级风险预警建模与结果输出。
- `problem3.py` 负责问题三的动态规划求解与干预方案优化。

### 推荐复现顺序

如需完整复现论文结果，推荐顺序如下：

1. 先确认 `src/Data` 中的样例数据文件存在且未被 Excel 占用。
2. 依次运行 `problem1.py`、`problem2.py`、`problem3.py` 生成图表与表格。
3. 使用 XeLaTeX 编译 `main.tex`，将 `src/outputs` 中的结果纳入论文。

## 结果产出说明

仓库中已经包含一批已生成结果，可直接用于检查与插图复用。

### 已生成图像

`src/outputs/figures` 中当前可见的典型图像包括：

- `Q1_corr_heatmap.png`
- `Q1_summary.png`
- `Q2_summary.png`
- `Q3_summary.png`
- `Q3_demo_trajectories.png`

### 已生成表格

`src/outputs/tables` 中当前可见的典型结果包括：

- `Q1_vif_setA.csv`
- `Q1_vif_setB.csv`
- `Q1_phlegm_ranking.csv`
- `Q1_risk_ranking.csv`
- `Q1_tizhi_contribution.csv`
- `Q2_cv_results.csv`
- `Q2_feature_importance.csv`
- `Q2_risk_full.csv`
- `Q2_core_combo.csv`
- `Q2_tree_rules.txt`
- `Q3_demo_3patients.csv`
- `Q3_all_phlegm.csv`

这些文件可以用于：

- 与论文中的数值结论进行交叉核对
- 直接插入论文图表
- 支持后续修改模型后的结果对比

## 论文编译说明

推荐在仓库根目录下执行：

```bash
xelatex main.tex
xelatex main.tex
```

补充说明：

- 论文主文件为 `main.tex`
- 模板类文件为 `MathorCupmodeling.cls`
- `main.tex` 已将图像路径设置为 `src/outputs/figures/`
- 若更新了脚本输出，重新编译即可将新图表纳入论文
- 如果交叉引用或目录未更新完整，通常再编译一次即可

当前仓库根目录下已存在 `main.pdf`，可作为最近一次编译结果参考。

---

## 附录 A：VSCode 使用说明

### 1. 基本认识

VSCode 本质上是一个编辑器，本身不直接提供所有语言的编译和运行能力，通常依赖：

- 对应语言的解释器或编译器
- 系统环境变量 `PATH`
- 你在 VSCode 中选择的运行环境

因此，不论是写 LaTeX、Python 还是其他代码，都需要先确认本机环境安装正确。

### 2. 需要特别注意的几个点

- 右下角的 Python 解释器是否选对，尤其是虚拟环境是否正确。
- 文件编码建议统一为 `UTF-8`，避免出现中文乱码。
- LaTeX、Python 等插件需要与本机环境配合使用，插件装好不代表环境一定可用。
- 文件命名要清晰，注释要简洁，方便团队成员协作。

### 3. 如果团队有统一配置文件

原协作流程中提到可以导入统一的 VSCode Profile，例如 `ychconfig.code-profile`。

如果团队后续仍提供这类配置文件，可按以下步骤导入：

1. 点击左下角齿轮图标。
2. 进入 `Profiles`。
3. 选择 `Import Profile`。
4. 导入团队提供的配置文件。

如果当前仓库中没有该文件，说明它并未随仓库分发，需要向组内成员单独获取。

---

## 附录 B：Bash / PowerShell 打开方式

在 Windows 下可以使用以下方式打开命令行：

1. 按 `Win + R`
2. 输入 `cmd` 或 `powershell`
3. 回车进入终端

如果你主要在 VSCode 中工作，也可以直接使用“终端 -> 新建终端”。

---

## 附录 C：Git 与 GitHub 协作说明

### 1. 安装 Git

直接在浏览器中下载并安装 Git 即可，默认安装选项通常就够用。

### 2. 检查 Git 是否可用

在 PowerShell 或 CMD 中输入：

```bash
git --version
```

如果能显示版本号，说明 Git 已经在环境变量中生效。

### 3. 配置用户名和邮箱

首次使用 Git 时，建议先配置全局身份信息：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

说明：

- 用户名可以使用团队内约定名称。
- 邮箱建议使用你平时登录 GitHub 的邮箱，便于识别提交记录。

### 4. 初始化本地仓库并关联远端

典型流程如下：

```bash
git init
git remote add origin https://github.com/M0onesy/Mathor_2026.git
```

建议步骤：

1. 先创建本地项目文件夹，例如 `Reposit`。
2. 用 VSCode 打开该文件夹。
3. 在 VSCode 终端中执行初始化命令。
4. 添加远端仓库地址，建立与 GitHub 项目的关联。

可用以下命令查看当前远端配置：

```bash
git remote -v
```

如果远端地址配置错了，可以移除后重新添加：

```bash
git remote remove origin
```

### 5. 每天开始工作前先同步

为了减少冲突，建议每次开始写代码之前先拉取远端最新内容：

```bash
git pull origin main
```

注意：

- 请先确认当前分支名是否确实为 `main`。
- 如果远端默认分支是 `master`，则把命令中的 `main` 改成 `master`。

### 6. 每天完成工作的标准提交流程

建议按照下面的顺序操作：

```bash
git add .
git commit -m "这里写清楚本次更新内容"
git pull origin main
git push origin main
```

这套流程的含义是：

1. `git add .` 把当前修改加入暂存区。
2. `git commit -m "..."` 形成一次本地提交。
3. `git pull origin main` 在推送前再次同步远端，减少“你写的时候别人已经更新了”的冲突风险。
4. `git push origin main` 把本地提交推送到远端。

### 7. 如果拉取时出现冲突

如果 `git pull` 过程中出现 `CONFLICT`，说明你和其他成员修改了同一文件的相近位置。

处理建议：

1. 打开冲突文件。
2. 找到 Git 标记出的冲突区域。
3. 手动决定保留哪部分内容，或将两边内容合理合并。
4. 修改完成后重新执行：

```bash
git add .
git commit -m "解决合并冲突"
git push origin main
```

### 8. 一个简化版速查流程

```bash
cd "你的项目路径"
git pull origin main
git add .
git commit -m "日常更新"
git push origin main
```

如果你准备推送前已经离上次同步过去了较长时间，仍然建议在 `commit` 后再执行一次 `git pull origin main` 做安全检查。

### 9. 常见补充命令

```bash
git remote -v
git remote remove origin
git pull origin main --allow-unrelated-histories
```

其中：

- `git remote -v` 用于查看当前远端关联
- `git remote remove origin` 用于删除错误的远端配置
- `git pull origin main --allow-unrelated-histories` 仅在本地仓库和远端仓库历史彼此无关时使用

### 10. 关于高风险 Git 操作的说明

原 README 中对部分命令的风险表述比较混杂，这里统一说明：

- 日常协作中不要把强制推送当作常规操作。
- 真正高风险的通常是 `git push --force` 这类会改写远端历史的命令。
- `git push -u origin main` 更多用于首次设置上游分支，不适合作为每天的固定更新命令，但它本身不等于强制覆盖远端。
- 不确定时，优先先 `pull` 再 `push`，并先确认当前分支与提交记录。

---

## 附录 D：Git GUI 协作说明

图形界面适合完成以下高频操作：

- 查看文件修改
- 编写提交说明
- 提交本地版本
- 推送到远端
- 拉取最新代码

它的优势是直观，但在复杂冲突、分支整理和异常状态处理上，命令行通常更稳定。

### 1. 本地无修改，只需要同步远端

这种情况下，直接执行同步或拉取即可。

适用场景：

- 你只是想获取队友最新更新
- 本地没有未提交改动

### 2. 远端无修改，需要把本地修改推上去

推荐流程：

1. 在 GUI 中查看改动。
2. 填写提交说明并提交。
3. 执行推送。

### 3. 本地和远端都有修改

这时不要直接无脑推送，建议按以下顺序处理：

1. 如果本地还有未完成工作，先暂存或 stash。
2. 先拉取远端更新。
3. 如果出现冲突，先解决冲突并确认合并结果。
4. 再进行提交和推送。

### 4. 如何理解图形界面中的常见状态

常见情况包括：

- `origin/main` 在你当前分支前面：说明远端比本地更新，你需要先同步。
- 本地分支领先远端：说明你本地有提交还没推送。
- 本地和远端各自领先：说明双方都改过，通常需要先拉取再处理冲突。

### 5. 版本差异比较

如果你想比较两个版本的文件，可以采用以下做法：

1. 准备两份需要比较的文件，放在同一目录下。
2. 确保文件名不同，避免覆盖。
3. 在 VSCode 或其他工具中选中两个文件并执行比较。

这样可以快速查看：

- 哪些位置被修改了
- 是新增、删除还是替换
- 合并时应该保留哪一部分内容

---

## 维护建议

如果后续继续更新本仓库，建议优先保持以下三点一致：

- README 中的项目说明与 `main.tex` 摘要和章节内容一致
- README 中的运行命令与 `src/pyCode` 中真实脚本入口一致
- README 中列出的输出文件类型与 `src/outputs` 中的实际结果一致
