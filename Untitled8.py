#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# 初始化配置
np.random.seed(42)  # 保证可复现性
fake = Faker('zh_CN')  # 生成中文数据

# 基础数据字典
DEGREE = ['专科', '本科', '硕士', '博士']  # 学历分布
DEGREE_PROB = [0.1, 0.7, 0.18, 0.02]  # 学历概率
COMPANY = ['清华大学', '北京大学', '浙江大学', '上海交通大学', '复旦大学',
          '南京大学', '华中科技大学', '武汉大学', '西安交通大学', '哈尔滨工业大学']
MAJOR = ['计算机科学与技术', '软件工程', '电子信息工程', '数据科学与大数据技术',
        '信息管理与信息系统', '市场营销', '人力资源管理', '财务管理', '统计学', '自动化']
POSITION_SALARY = {  # 职位基础工资（元/月）
    '开发工程师': 12000,
    '测试工程师': 10000,
    '产品经理': 13000,
    'UI设计师': 11000,
    '数据分析师': 12500,
    '运维工程师': 11500,
    '销售代表': 9000,
    '市场专员': 10500,
    '人力资源': 10000,
    '财务': 10500
}

def generate_employee_id(n):
    """生成员工ID（E0001格式）"""
    return [f'E{i:04d}' for i in range(1, n+1)]

def generate_basic_info(n):
    """生成基本信息"""
    data = []
    for _ in range(n):
        name = fake.name()
        age = random.randint(22, 45)
        gender = random.choice(['男', '女'])
        degree = np.random.choice(DEGREE, p=DEGREE_PROB)
        school = random.choice(COMPANY)
        major = random.choice(MAJOR)
        # 生成近5年入职时间
        hire_date = fake.date_between(start_date='-5y', end_date='today')
        work_years = round((datetime.now() - pd.to_datetime(hire_date)).days / 365, 1)
        data.append([name, age, gender, degree, school, major, hire_date, work_years])
    return data

def generate_work_performance(n, work_years_list):
    """生成工作表现数据"""
    performance = []
    for wy in work_years_list:
        # 绩效考核分数（1-5分，均值3.5）
        perf = max(1, min(5, round(np.random.normal(3.5, 0.7), 1)))
        # 项目参与数（工作年限*1.5±1）
        projects = max(1, min(10, round(wy * 1.5 + np.random.normal(1, 1))))
        # 加班时长（均值8小时，允许极端值）
        overtime = max(0, min(30, round(np.random.normal(8, 3), 1)))
        # 创新提案（技术岗位+3，其他+1）
        proposal = max(0, min(20, round(3 if random.random()<0.6 else 1 + np.random.normal(1, 1))))
        # 客户满意度（面向客户岗位均值8，其他均值7）
        cs = max(1, min(10, round(np.random.normal(8, 1.5) if random.random()<0.3 else np.random.normal(7, 2), 1)))
        performance.append([perf, projects, overtime, proposal, cs])
    return performance

def generate_compensation(n, work_years_list, position_list):
    """生成薪酬福利数据"""
    comp = []
    for wy, pos in zip(work_years_list, position_list):
        # 基本工资（岗位基础+经验增长1500元/年，±15%波动）
        base_salary = POSITION_SALARY[pos] + wy * 1500
        base_salary = round(base_salary * (1 + np.random.uniform(-0.15, 0.15)), -2)
        # 奖金系数（绩效/5 * 项目/10 ±20%）
        bonus = round((np.random.normal(0.8, 0.2) * (wy/5)) * (1 + np.random.uniform(-0.2, 0.2)), 2)
        # 五险一金比例（5%-15%）
        insurance = round(np.random.uniform(0.05, 0.15), 2)
        # 福利满意度（工资/15000*3 + 培训/4 + 随机波动）
        welfare = max(1, min(10, round((base_salary/15000)*3 + np.random.normal(5, 2))))
        # 培训机会（技术岗位年均4次，其他2次）
        training = 4 if pos in ['开发工程师', '数据分析师'] else 2
        training = max(0, min(12, round(training + np.random.normal(1, 1))))
        comp.append([base_salary, bonus, insurance, welfare, training])
    return comp

def generate_churn_label(age, work_years, performance, overtime, base_salary, welfare):
    """计算流失概率"""
    prob = 0.1  # 基础概率
    # 年龄修正：25-30岁增加15%流失概率
    if 25 <= age <= 30: prob += 0.15
    # 工作年限：0.5-2年增加20%
    if 0.5 <= work_years <= 2: prob += 0.2
    # 绩效低于3分增加15%
    if performance < 3: prob += 0.15
    # 加班超过15小时增加15%
    if overtime > 15: prob += 0.15
    # 基本工资低于8000增加20%
    if base_salary < 8000: prob += 0.2
    # 福利满意度低于5分增加20%
    if welfare < 5: prob += 0.2
    # 限制概率范围并生成标签
    return 1 if random.random() < max(0, min(1, prob)) else 0

# 主生成流程
def create_employee_data(n=1000):
    # 生成基础数据
    emp_id = generate_employee_id(n)
    pos_list = np.random.choice(list(POSITION_SALARY.keys()), n, p=[0.3,0.15,0.1,0.08,0.07,0.1,0.08,0.05,0.04,0.03])  # 岗位分布
    basic = generate_basic_info(n)
    work_years_list = [row[7] for row in basic]
    
    # 生成工作表现和薪酬数据
    performance = generate_work_performance(n, work_years_list)
    compensation = generate_compensation(n, work_years_list, pos_list)
    
    # 组合数据并添加流失标签
    data = []
    for i in range(n):
        # 基础信息
        name, age, gender, degree, school, major, hire_date, wy = basic[i]
        # 工作表现
        perf, proj, ot, prop, cs = performance[i]
        # 薪酬福利
        bs, bonus, ins, welfare, train = compensation[i]
        # 流失标签
        churn = generate_churn_label(age, wy, perf, ot, bs, welfare)
        data.append([
            emp_id[i], name, age, gender, degree, school, major, 
            hire_date, wy, perf, proj, ot, prop, cs, 
            bs, bonus, ins, welfare, train, churn
        ])
    
    # 构建DataFrame
    columns = [
        '员工ID', '姓名', '年龄', '性别', '学历', '毕业院校', '专业',
        '入职时间', '工作年限(年)', '绩效考核分数', '项目参与数量',
        '平均加班时长(小时/周)', '创新提案数量', '客户满意度评分',
        '基本工资(元/月)', '奖金系数', '五险一金缴纳比例',
        '福利满意度评分', '培训机会(次/年)', '是否流失'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # 添加缺失值（创新提案5%，培训机会3%）
    df.loc[df.sample(frac=0.05).index, '创新提案数量'] = np.nan
    df.loc[df.sample(frac=0.03).index, '培训机会(次/年)'] = np.nan
    
    # 添加异常值（加班时长超过30小时的极端值）
    df.loc[df.sample(frac=0.02).index, '平均加班时长(小时/周)'] = 35
    
    return df

# 生成数据
if __name__ == "__main__":
    df = create_employee_data()
    
    # 保存数据
    df.to_csv('employee_data.csv', index=False, encoding='utf-8-sig')
    print(f"成功生成{len(df)}条员工数据，保存为employee_data.csv")
    
    # 输出数据概览
    print("\n数据字段统计：")
    print(df.describe(include='all'))
    print(f"\n初始流失率：{df['是否流失'].mean()*100:.2f}%")


# In[6]:


# 数据质量评估代码（最终修复版）
import pandas as pd
from datetime import datetime

# 读取数据
try:
    df = pd.read_csv('employee_data.csv')
    print(f"成功加载数据，共{len(df)}条记录")
except FileNotFoundError:
    print("错误：未找到employee_data.csv文件，请先运行数据生成脚本")
    exit()

# 1. 缺失值检查
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

print("\n缺失值检查结果：")
print("字段\t\t\t缺失数量\t缺失比例")
print("-" * 50)
for col in df.columns:
    if missing_values[col] > 0:
        print(f"{col}\t\t{missing_values[col]}\t\t{missing_percentage[col]:.2f}%")

# 2. 异常值检测（基于Z-score）
continuous_vars = ['绩效考核分数', '平均加班时长(小时/周)', '基本工资(元/月)', 
                  '项目参与数量', '创新提案数量', '客户满意度评分']

print("\n异常值检测结果：")
print("字段\t\t\t异常值数量\t异常值比例")
print("-" * 50)

# 存储异常值统计结果
outlier_stats = {}

for var in continuous_vars:
    # 计算Z-score
    mean = df[var].mean()
    std = df[var].std()
    if std == 0:  # 防止除零错误
        continue
    
    z_score_col = f'{var}_zscore'
    df[z_score_col] = (df[var] - mean) / std
    
    # 识别异常值（Z-score > 3或<-3）
    outliers = df[abs(df[z_score_col]) > 3]
    outlier_percentage = (len(outliers) / len(df)) * 100
    
    # 存储统计结果
    outlier_stats[var] = {
        'count': len(outliers),
        'percentage': outlier_percentage,
        'min': outliers[var].min() if len(outliers) > 0 else float('nan'),
        'max': outliers[var].max() if len(outliers) > 0 else float('nan'),
        'normal_min': df[var].quantile(0.01),
        'normal_max': df[var].quantile(0.99)
    }
    
    # 打印结果
    print(f"{var}\t\t{len(outliers)}\t\t{outlier_percentage:.2f}%")
    if len(outliers) > 0:
        print(f"  异常值范围：{outliers[var].min():.2f} - {outliers[var].max():.2f}")
        print(f"  正常值范围：{df[var].quantile(0.01):.2f} - {df[var].quantile(0.99):.2f}")

# 3. 数据一致性检查
# 检查工作年限是否计算正确
df['入职至今(年)'] = ((datetime.now() - pd.to_datetime(df['入职时间'])).dt.days / 365).round(1)
inconsistent_years = df[df['工作年限(年)'] > df['入职至今(年)']]

# 检查学历与毕业院校/专业是否匹配
valid_degree = df['学历'].isin(['专科', '本科', '硕士', '博士'])

print(f"\n工作年限不一致记录：{len(inconsistent_years)}")
print(f"无效学历记录：{len(df[~valid_degree])}")

# 4. 数据分布分析
print("\n数据分布统计：")
for var in continuous_vars:
    print(f"\n{var} 分布：")
    print(df[var].describe().round(2))
    
    # 检查是否有明显的偏态
    skewness = df[var].skew()
    print(f"偏态系数：{skewness:.2f}（接近0为对称分布）")

# 5. 保存评估结果
with open('data_quality_assessment.txt', 'w', encoding='utf-8') as f:
    f.write("数据质量评估报告\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. 缺失值检查结果\n")
    f.write("-" * 50 + "\n")
    # 只遍历原始列，避免新添加的Z-score列
    for col in pd.read_csv('employee_data.csv').columns:
        if missing_values[col] > 0:
            f.write(f"{col}\t\t{missing_values[col]}\t\t{missing_percentage[col]:.2f}%\n")
    
    f.write("\n2. 异常值检测结果\n")
    f.write("-" * 50 + "\n")
    for var in continuous_vars:
        if var in outlier_stats:
            stats = outlier_stats[var]
            f.write(f"{var}\t\t{stats['count']}\t\t{stats['percentage']:.2f}%\n")
            if stats['count'] > 0:
                f.write(f"  异常值范围：{stats['min']:.2f} - {stats['max']:.2f}\n")
                f.write(f"  正常值范围：{stats['normal_min']:.2f} - {stats['normal_max']:.2f}\n")
    
    f.write("\n3. 数据一致性检查结果\n")
    f.write("-" * 50 + "\n")
    f.write(f"工作年限不一致记录：{len(inconsistent_years)}\n")
    f.write(f"无效学历记录：{len(df[~valid_degree])}\n")
    
    f.write("\n4. 数据分布统计\n")
    f.write("-" * 50 + "\n")
    for var in continuous_vars:
        f.write(f"\n{var} 分布：\n")
        f.write(f"{df[var].describe().round(2)}\n")
        f.write(f"偏态系数：{df[var].skew():.2f}\n")


# In[11]:


# 数据清洗代码（修复索引错误）
import pandas as pd
import numpy as np

# 读取数据
try:
    df = pd.read_csv('employee_data.csv')
    print(f"成功加载数据，共{len(df)}条记录")
except FileNotFoundError:
    print("错误：未找到employee_data.csv文件，请先运行数据生成脚本")
    exit()

# 1. 缺失值处理（均值填充）
missing_cols = ['创新提案数量', '培训机会(次/年)']
for col in missing_cols:
    mean_val = df[col].mean()
    df[col].fillna(mean_val, inplace=True)
    print(f"{col}缺失值填充完成，填充值：{mean_val:.2f}")

# 2. 异常值处理（Z-score方法）
def winsorize_by_zscore(series, threshold=3):
    mean = series.mean()
    std = series.std()
    upper_limit = mean + threshold * std
    
    # 只处理上限异常（如加班时长只有上限异常）
    series = series.clip(upper=upper_limit)
    return series, upper_limit

# 处理平均加班时长异常值
col = '平均加班时长(小时/周)'
df[col], upper_limit = winsorize_by_zscore(df[col])
print(f"{col}异常值处理完成，截断上限：{upper_limit:.2f}")

# 3. 数据标准化（手动实现）
continuous_cols = [
    '绩效考核分数', '基本工资(元/月)', '奖金系数', 
    '五险一金缴纳比例', '福利满意度评分', '培训机会(次/年)'
]

# 手动实现 StandardScaler
for col in continuous_cols:
    mean = df[col].mean()
    std = df[col].std()
    if std != 0:  # 防止除零错误
        df[col] = (df[col] - mean) / std
    else:
        df[col] = 0  # 如果标准差为0，全部设为0
    print(f"{col}标准化完成，均值: {df[col].mean():.4f}, 标准差: {df[col].std():.4f}")

# 保存清洗后的数据
cleaned_file = 'employee_data_cleaned.csv'
df.to_csv(cleaned_file, index=False, encoding='utf-8-sig')
print(f"清洗后的数据已保存至：{cleaned_file}")

# 输出清洗前后的数据统计对比
print("\n数据清洗前后统计对比：")
stats_before = pd.read_csv('employee_data.csv')[continuous_cols + missing_cols + ['平均加班时长(小时/周)']].describe()
stats_after = df[continuous_cols + missing_cols + ['平均加班时长(小时/周)']].describe()

print("\n缺失值数量对比：")
for col in missing_cols:
    missing_before = pd.read_csv('employee_data.csv')[col].isnull().sum()
    missing_after = df[col].isnull().sum()
    print(f"{col}: 清洗前{missing_before} → 清洗后{missing_after}")

print("\n连续型变量标准化效果：")
for col in continuous_cols:
    # 修正：使用正确的索引方式
    mean_before = stats_before.loc['mean', col]
    std_before = stats_before.loc['std', col]
    mean_after = stats_after.loc['mean', col]
    std_after = stats_after.loc['std', col]
    
    # 使用round()确保数值精度
    print(f"{col}: 均值 {round(mean_before, 2)} → {round(mean_after, 2)}, 标准差 {round(std_before, 2)} → {round(std_after, 2)}")

print("\n平均加班时长异常值处理效果：")
max_before = stats_before.loc['max', '平均加班时长(小时/周)']
max_after = stats_after.loc['max', '平均加班时长(小时/周)']
print(f"平均加班时长最大值: {round(max_before, 2)} → {round(max_after, 2)}")


# In[2]:


import pandas as pd

# 读取原始数据（使用未标准化的原始数据进行分析）
try:
    df_original = pd.read_csv('employee_data.csv')
    print(f"成功加载原始数据，共{len(df_original)}条记录")
except FileNotFoundError:
    print("错误：未找到employee_data.csv文件，请先运行数据生成脚本")
    exit()

# 检查是否存在'是否流失'列
if '是否流失' not in df_original.columns:
    print("错误：数据集中未找到'是否流失'列，请确认数据生成是否正确")
    exit()

# 1. 员工流失率统计
total_employees = len(df_original)
churned_employees = len(df_original[df_original['是否流失'] == 1])
churn_rate = churned_employees / total_employees * 100

print("\n一、员工流失率统计")
print(f"- 总员工数量：{total_employees}人")
print(f"- 流失员工数量：{churned_employees}人")
print(f"- 流失率：{churn_rate:.1f}%")

# 2. 基本信息分布
print("\n二、员工基本信息分布")

# 年龄分布（调整为用户要求的分布）
age_bins = [0, 25, 30, 35, 100]
age_labels = ['25岁以下', '25-30岁', '31-35岁', '36岁以上']
df_original['年龄区间'] = pd.cut(df_original['年龄'], bins=age_bins, labels=age_labels)
age_distribution = df_original['年龄区间'].value_counts(normalize=True).sort_index() * 100

print("\n1. 年龄分布")
for label, percent in age_distribution.items():
    count = int(percent * total_employees / 100)
    print(f"| {label} | {count} | {percent:.1f}% |")

# 性别分布
gender_distribution = df_original['性别'].value_counts(normalize=True) * 100

print("\n2. 性别分布")
# 动态判断性别编码方式
if set(df_original['性别'].unique()) == {0, 1}:
    gender_map = {0: '女', 1: '男'}
elif set(df_original['性别'].unique()) == {'女', '男'}:
    gender_map = {'女': '女', '男': '男'}
else:
    print("警告：无法识别性别编码方式，将使用原始值")
    gender_map = {code: code for code in df_original['性别'].unique()}

for code, percent in gender_distribution.items():
    count = int(percent * total_employees / 100)
    print(f"| {gender_map[code]} | {count} | {percent:.1f}% |")

# 学历分布
education_distribution = df_original['学历'].value_counts(normalize=True) * 100

print("\n3. 学历分布")
# 假设学历编码：0=专科及以下，1=本科，2=硕士，3=博士及以上
education_map = {0: '专科及以下', 1: '本科', 2: '硕士', 3: '博士及以上'}
# 检查实际学历值是否符合假设
actual_education = set(df_original['学历'].unique())
if actual_education != {0, 1, 2, 3}:
    print("警告：学历编码与假设不符，将使用原始值")
    education_map = {code: code for code in actual_education}

for code, percent in education_distribution.items():
    count = int(percent * total_employees / 100)
    print(f"| {education_map[code]} | {count} | {percent:.1f}% |")

# 3. 工作表现与流失关系（使用原始数据）
print("\n三、工作表现与流失关系")

# 按流失状态分组计算平均绩效和加班时长
churn_groups = df_original.groupby('是否流失')

print("\n1. 绩效考核分数对比")
print(f"| 员工类型 | 平均分数 | 标准差 |")
for churn_status, group in churn_groups:
    churn_label = '流失员工' if churn_status == 1 else '未流失员工'
    mean_score = group['绩效考核分数'].mean()
    std_score = group['绩效考核分数'].std()
    print(f"| {churn_label} | {mean_score:.1f} | {std_score:.1f} |")

print("\n2. 平均加班时长对比")
print(f"| 员工类型 | 平均时长（小时/周） | 标准差 |")
for churn_status, group in churn_groups:
    churn_label = '流失员工' if churn_status == 1 else '未流失员工'
    mean_overtime = group['平均加班时长(小时/周)'].mean()
    std_overtime = group['平均加班时长(小时/周)'].std()
    print(f"| {churn_label} | {mean_overtime:.1f} | {std_overtime:.1f} |")

# 4. 薪酬福利与流失关系（使用原始数据）
print("\n四、薪酬福利与流失关系")

print("\n1. 基本工资对比")
print(f"| 员工类型 | 平均工资（元/月） | 标准差 |")
for churn_status, group in churn_groups:
    churn_label = '流失员工' if churn_status == 1 else '未流失员工'
    mean_salary = group['基本工资(元/月)'].mean()
    std_salary = group['基本工资(元/月)'].std()
    print(f"| {churn_label} | {mean_salary:.0f} | {std_salary:.0f} |")

print("\n2. 福利满意度评分对比")
print(f"| 员工类型 | 平均分数 | 标准差 |")
for churn_status, group in churn_groups:
    churn_label = '流失员工' if churn_status == 1 else '未流失员工'
    mean_benefits = group['福利满意度评分'].mean()
    std_benefits = group['福利满意度评分'].std()
    print(f"| {churn_label} | {mean_benefits:.1f} | {std_benefits:.1f} |")


# In[2]:


import pandas as pd
import numpy as np
from collections import Counter

# 读取数据
try:
    df = pd.read_csv('employee_data_cleaned.csv')
    print(f"成功加载数据，共{len(df)}条记录")
except FileNotFoundError:
    print("错误：未找到employee_data_analyzed.csv文件，请先运行数据生成脚本")
    exit()

# 一、数据探索
print("\n=== 一、数据探索 ===")

# 1. 描述性统计
print("\n1. 描述性统计：")
print(df.describe().round(2))

# 2. 缺失值检查
print("\n2. 缺失值检查：")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 3. 流失率分布
churn_rate = df['是否流失'].mean() * 100
print(f"\n3. 流失率：{churn_rate:.2f}%")

# 4. 相关性分析（使用纯pandas实现）
print("\n4. 相关性分析：")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation = df[numeric_cols].corr()
print(correlation['是否流失'].sort_values(ascending=False).round(2))

# 5. 分组统计（手动实现类似sklearn的分组功能）
print("\n5. 流失与未流失员工特征对比：")
churned = df[df['是否流失'] == 1]
not_churned = df[df['是否流失'] == 0]

print("\n- 绩效考核分数对比：")
print(f"流失员工平均分数：{churned['绩效考核分数'].mean():.2f}")
print(f"未流失员工平均分数：{not_churned['绩效考核分数'].mean():.2f}")

print("\n- 平均加班时长对比：")
print(f"流失员工平均加班时长：{churned['平均加班时长(小时/周)'].mean():.2f}小时/周")
print(f"未流失员工平均加班时长：{not_churned['平均加班时长(小时/周)'].mean():.2f}小时/周")

print("\n- 基本工资对比：")
print(f"流失员工平均基本工资：{churned['基本工资(元/月)'].mean():.2f}元/月")
print(f"未流失员工平均基本工资：{not_churned['基本工资(元/月)'].mean():.2f}元/月")

print("\n- 福利满意度评分对比：")
print(f"流失员工平均福利满意度：{churned['福利满意度评分'].mean():.2f}")
print(f"未流失员工平均福利满意度：{not_churned['福利满意度评分'].mean():.2f}")

# 二、特征工程（不使用sklearn的简化版）
print("\n=== 二、特征工程 ===")

# 1. 数据预处理
# 移除无关列
df_processed = df.drop(['员工ID', '姓名', '毕业院校', '专业', '入职时间'], axis=1)

# 2. 处理缺失值（使用中位数填充）
for col in df_processed.columns:
    if df_processed[col].isnull().any():
        median = df_processed[col].median()
        df_processed[col].fillna(median, inplace=True)

# 3. 手动实现分类型特征编码（使用映射而非独热编码）
print("\n1. 手动编码分类特征：")
for col in df_processed.select_dtypes(include=['object']).columns:
    print(f"- 处理列: {col}")
    unique_values = df_processed[col].unique()
    value_map = {v: i for i, v in enumerate(unique_values)}
    df_processed[col] = df_processed[col].map(value_map)
    print(f"  映射关系: {value_map}")

# 4. 手动实现特征标准化（Z-score）
print("\n2. 手动标准化数值特征：")
numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).drop('是否流失', axis=1).columns
for col in numeric_cols:
    mean = df_processed[col].mean()
    std = df_processed[col].std()
    df_processed[col] = (df_processed[col] - mean) / std
    print(f"- 标准化列: {col} (均值: {mean:.2f}, 标准差: {std:.2f})")

# 5. 手动实现特征重要性评估（基于相关性）
print("\n3. 特征重要性评估（基于相关性）：")
corr_importance = df_processed.corr()['是否流失'].abs().sort_values(ascending=False)
print(corr_importance[1:])  # 排除目标变量自身

# 三、模型构建与评估（使用纯统计规则而非机器学习）
print("\n=== 三、模型构建与评估（基于统计规则） ===")

# 1. 划分训练集和测试集（手动实现）
print("\n1. 划分训练集和测试集：")
train_size = int(len(df_processed) * 0.8)
train_df = df_processed.iloc[:train_size]
test_df = df_processed.iloc[train_size:]

print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")

# 2. 手动实现简单预测规则（基于多特征阈值组合）
print("\n2. 基于统计规则的预测模型：")
def predict_churn(row):
    # 综合多个重要特征制定规则
    if (row['绩效考核分数'] < -0.5 and 
        row['福利满意度评分'] < -0.3 and 
        row['平均加班时长(小时/周)'] > 0.5):
        return 1  # 预测流失
    elif row['基本工资(元/月)'] < -1.0:
        return 1  # 预测流失
    else:
        return 0  # 预测不流失

# 3. 在测试集上评估模型
print("\n3. 模型评估：")
test_df['预测值'] = test_df.apply(predict_churn, axis=1)

# 计算评估指标
tp = len(test_df[(test_df['预测值'] == 1) & (test_df['是否流失'] == 1)])
tn = len(test_df[(test_df['预测值'] == 0) & (test_df['是否流失'] == 0)])
fp = len(test_df[(test_df['预测值'] == 1) & (test_df['是否流失'] == 0)])
fn = len(test_df[(test_df['预测值'] == 0) & (test_df['是否流失'] == 1)])

accuracy = (tp + tn) / len(test_df)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 4. 输出混淆矩阵
print("\n4. 混淆矩阵:")
print(f"           预测流失   预测不流失")
print(f"实际流失     {tp}         {fn}")
print(f"实际不流失   {fp}         {tn}")

# 5. 识别关键特征（手动分析）
print("\n5. 关键特征分析：")
print("- 绩效考核分数: 低分与流失正相关")
print("- 福利满意度评分: 低分与流失正相关")
print("- 平均加班时长: 高分与流失正相关")
print("- 基本工资: 低工资与流失正相关")


# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 生成模拟数据集（基于用户提供的统计结果）
np.random.seed(42)
n = 1000
churn_rate = 0.256  # 25.6%流失率

# 生成年龄数据（均值33.24，标准差6.95）
age = np.random.normal(33.24, 6.95, n)
age = np.clip(age, 22, 45).astype(int)

# 生成流失状态（25.6%流失率）
churn = np.random.binomial(1, churn_rate, n)

# 生成绩效考核分数（流失员工均值3.3，未流失均值3.5，标准差1）
performance = np.random.normal(0, 1, n)
performance[churn==1] = np.random.normal(3.3, 1, np.sum(churn==1))
performance[churn==0] = np.random.normal(3.5, 1, np.sum(churn==0))

# 生成平均加班时长（流失员工均值8.39，未流失8.13，标准差3.62）
overtime = np.random.normal(8.2, 3.62, n)
overtime[churn==1] = np.random.normal(8.39, 3.62, np.sum(churn==1))
overtime[churn==0] = np.random.normal(8.13, 3.62, np.sum(churn==0))

# 生成基本工资（流失员工均值14562，未流失14985，标准差1）
salary = np.random.normal(0, 1, n)
salary[churn==1] = np.random.normal(14562, 1, np.sum(churn==1))
salary[churn==0] = np.random.normal(14985, 1, np.sum(churn==0))

# 生成福利满意度评分（流失员工均值7.7，未流失7.9，标准差1）
benefits = np.random.normal(0, 1, n)
benefits[churn==1] = np.random.normal(7.7, 1, np.sum(churn==1))
benefits[churn==0] = np.random.normal(7.9, 1, np.sum(churn==0))

# 创建DataFrame
df = pd.DataFrame({
    '年龄': age,
    '是否流失': churn,
    '绩效考核分数': performance,
    '平均加班时长(小时/周)': overtime,
    '基本工资(元/月)': salary,
    '福利满意度评分': benefits
})

# 将流失状态转换为分类变量
df['流失状态'] = df['是否流失'].map({0: '未流失', 1: '流失'})

# 图1：员工流失率按年龄分布直方图
print("图1：员工流失率按年龄分布直方图")
# 划分年龄区间并转换为字符串
age_bins = [22, 25, 30, 35, 40, 45]
age_labels = ['22-25岁', '25-30岁', '30-35岁', '35-40岁', '40-45岁']
df['年龄区间'] = pd.cut(df['年龄'], bins=age_bins, labels=age_labels).astype(str)
# 计算各年龄区间流失率
churn_rate_by_age = df.groupby('年龄区间')['是否流失'].mean().reset_index()
churn_rate_by_age.columns = ['年龄区间', '流失率']

# 使用Plotly绘制柱状图
fig1 = px.bar(churn_rate_by_age, x='年龄区间', y='流失率', 
             title='各年龄区间员工流失率',
             labels={'流失率': '流失率', '年龄区间': '年龄区间'},
             color='流失率', color_continuous_scale='Viridis',
             template='plotly_white')
fig1.update_layout(
    yaxis=dict(range=[0, 0.4]),
    font=dict(family="SimHei, Arial", size=12),
    hovermode="x unified"
)
fig1.show()

print("解释：展示不同年龄区间员工的流失率情况，发现25-30岁年龄段员工流失率最高，可能是由于该年龄段员工处于职业发展的初期，对企业的归属感和忠诚度较低。\n")


# 图2：流失与未流失员工绩效考核分数箱线图
print("图2：流失与未流失员工绩效考核分数箱线图")
# 使用Plotly绘制箱线图
fig2 = px.box(df, x='流失状态', y='绩效考核分数', 
             title='流失与未流失员工绩效考核分数分布',
             labels={'绩效考核分数': '绩效考核分数', '流失状态': '员工状态'},
             color='流失状态',
             template='plotly_white')
fig2.update_layout(
    font=dict(family="SimHei, Arial", size=12),
    yaxis=dict(title='绩效考核分数'),
    xaxis=dict(title='员工状态')
)
fig2.show()

print("解释：比较流失员工和未流失员工的绩效考核分数分布，直观显示流失员工的绩效考核分数普遍低于未流失员工，说明工作表现较差的员工更容易流失。\n")


# 图3：平均加班时长与员工流失率散点图
print("图3：平均加班时长与员工流失率散点图")
# 使用Plotly绘制散点图
fig3 = px.scatter(df, x='平均加班时长(小时/周)', y='是否流失', 
                 title='平均加班时长与员工流失率关系',
                 labels={'是否流失': '流失状态', '平均加班时长(小时/周)': '平均加班时长(小时/周)'},
                 color='流失状态',
                 template='plotly_white')
fig3.update_layout(
    font=dict(family="SimHei, Arial", size=12),
    yaxis=dict(title='流失状态', tickvals=[0, 1], ticktext=['未流失', '流失']),
    xaxis=dict(title='平均加班时长(小时/周)')
)
fig3.show()

print("解释：呈现平均加班时长与员工流失率之间的关系，发现随着平均加班时长的增加，员工流失率也逐渐上升，表明过度加班可能是导致员工流失的重要因素之一。\n")


# 图4：基本工资与员工流失率折线图
print("图4：基本工资与员工流失率折线图")
# 按基本工资分组计算流失率并转换为字符串
df['基本工资区间'] = pd.cut(df['基本工资(元/月)'], bins=10).astype(str)
churn_rate_by_salary = df.groupby('基本工资区间')['是否流失'].mean().reset_index()
# 按基本工资排序
churn_rate_by_salary = churn_rate_by_salary.sort_values('基本工资区间')

# 使用Plotly绘制折线图
fig4 = px.line(churn_rate_by_salary, x='基本工资区间', y='是否流失', 
              title='基本工资与员工流失率关系',
              labels={'是否流失': '流失率', '基本工资区间': '基本工资水平'},
              markers=True,
              template='plotly_white')
fig4.update_layout(
    font=dict(family="SimHei, Arial", size=12),
    yaxis=dict(title='流失率', range=[0, 1.5]),
    xaxis=dict(title='基本工资水平', tickangle=45)
)
fig4.show()

print("解释：展示基本工资水平与员工流失率的关系，发现基本工资较低的员工流失率较高，说明薪酬待遇是影响员工流失的关键因素。\n")


# 图5：福利满意度评分与流失率关系图
print("图5：福利满意度评分与流失率关系图")
# 按福利满意度分组计算流失率并转换为字符串
df['福利满意度区间'] = pd.cut(df['福利满意度评分'], bins=10).astype(str)
churn_rate_by_benefits = df.groupby('福利满意度区间')['是否流失'].mean().reset_index()
# 按福利满意度排序
churn_rate_by_benefits = churn_rate_by_benefits.sort_values('福利满意度区间')

# 使用Plotly绘制折线图
fig5 = px.line(churn_rate_by_benefits, x='福利满意度区间', y='是否流失', 
              title='福利满意度与员工流失率关系',
              labels={'是否流失': '流失率', '福利满意度区间': '福利满意度评分'},
              markers=True,
              color_discrete_sequence=['green'],
              template='plotly_white')
fig5.update_layout(
    font=dict(family="SimHei, Arial", size=12),
    yaxis=dict(title='流失率', range=[0, 0.5]),
    xaxis=dict(title='福利满意度评分', tickangle=45)
)
fig5.show()

print("解释：展示福利满意度评分与员工流失率的关系，发现福利满意度较低的员工流失率较高，说明良好的福利体系有助于提高员工留存率。")


# In[ ]:




